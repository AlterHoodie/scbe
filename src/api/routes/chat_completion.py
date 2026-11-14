"""Chat and upload routes."""

from uuid import uuid4

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.api.models import (
    ChatRequest,
    SessionDetailResponse,
    SessionListResponse,
    SessionProgress,
    SessionSummary,
    TranscriptAnalysisMessage,
    TranscriptTextMessage,
    UploadCompleteRequest,
    UploadCompleteResponse,
    UploadRequest,
    UploadResponse,
)
from src.logger import get_logger
from src.models.job import Job
from src.models.session import Message, Session
from src.repo.jobs import JobsRepo
from src.repo.sessions import SessionsRepo
from src.services.aws import get_s3_client
from src.services.llm import chat_branch, chat_events, ensure_job_tool_messages, ready_payload, series_frames, sse
from src.services.mongodb import get_mongodb_client
from src.services.rabbitmq import get_rabbitmq

router = APIRouter()
log = get_logger("server")

_UPLOAD_TYPES = {
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/webm": ".webm",
}

_PREVIEW_LEN = 80


def _preview_text(messages: list[Message]) -> str | None:
    for item in messages:
        if item.role == "user" and item.content and item.content.strip():
            text = " ".join(item.content.split())
            if len(text) > _PREVIEW_LEN:
                return f"{text[:_PREVIEW_LEN]}…"
            return text
    return None


def _session_phase(job: Job | None) -> str:
    branch = chat_branch(job)
    if branch == "resume":
        return "queued" if job and job.status == "queued" else "running"
    if branch == "coach":
        return "coaching"
    if branch == "failed":
        return "failed"
    if job and job.status == "failed":
        return "failed"
    if job and job.status == "done":
        return "done"
    return "idle"


async def _transcript_for_ui(
    session: Session,
    jobs: JobsRepo,
) -> list[TranscriptTextMessage | TranscriptAnalysisMessage]:
    items: list[TranscriptTextMessage | TranscriptAnalysisMessage] = []
    s3 = get_s3_client()
    for message in session.messages:
        if message.role == "system":
            continue
        if message.role == "tool" and message.job_id:
            job = await jobs.get(message.job_id)
            if job is None or job.status not in ("done", "failed"):
                continue
            # Match the chat contract: show a results card once coaching finished,
            # or immediately for a failed analysis.
            if job.status == "done" and job.reply is None:
                continue
            payload = ready_payload(job, series_frames(session, job.job_id))
            items.append(
                TranscriptAnalysisMessage(id=job.job_id, job=payload),
            )
            continue
        if message.role not in ("user", "assistant"):
            continue
        if message.tool_calls and not message.content:
            continue
        if not message.content and not message.video_keys:
            continue
        items.append(
            TranscriptTextMessage(
                id=str(uuid4()),
                role=message.role,  # type: ignore[arg-type]
                text=message.content or "",
                video_urls=[s3.presign_get(key) for key in message.video_keys],
            )
        )
    return items


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions() -> SessionListResponse:
    """List recent chat sessions, newest first."""
    db = await get_mongodb_client()
    sessions = SessionsRepo(db)
    rows = await sessions.list_all()
    return SessionListResponse(
        sessions=[
            SessionSummary(
                session_id=row.session_id,
                created_at=row.created_at,
                updated_at=row.updated_at,
                preview=_preview_text(row.messages),
            )
            for row in rows
        ]
    )


@router.get("/sessions/{session_id}", response_model=SessionDetailResponse)
async def get_session(session_id: str) -> SessionDetailResponse:
    """Return one session's transcript so the client can open that chatbox."""
    db = await get_mongodb_client()
    sessions = SessionsRepo(db)
    jobs = JobsRepo(db)
    session = await sessions.get(session_id)
    if session is None:
        log.warning("Rejected session get", session_id=session_id, reason="unknown_session")
        raise HTTPException(status_code=404, detail="Unknown session")

    job = await jobs.get_latest_for_session(session_id)
    phase = _session_phase(job)
    progress = None
    if job and job.status == "running":
        progress = SessionProgress(frame=job.progress.frame, frame_count=job.progress.frame_count)
    elif phase == "idle" and any(
        item.role in ("user", "assistant") and item.content for item in session.messages
    ):
        phase = "done"

    messages = await _transcript_for_ui(session, jobs)
    return SessionDetailResponse(
        session_id=session.session_id,
        created_at=session.created_at,
        updated_at=session.updated_at,
        phase=phase,  # type: ignore[arg-type]
        messages=messages,
        progress=progress,
    )


@router.post("/uploads", response_model=UploadResponse)
async def create_upload(body: UploadRequest) -> UploadResponse:
    """Return a presigned PUT for a squat video. Creates a session if none is given."""
    db = await get_mongodb_client()
    sessions = SessionsRepo(db)
    if body.session_id:
        session = await sessions.get(body.session_id)
        if session is None:
            log.warning("Rejected upload", session_id=body.session_id, reason="unknown_session")
            raise HTTPException(status_code=404, detail="Unknown session")
    else:
        session = Session()
        await sessions.insert_one(session)

    extension = _UPLOAD_TYPES[body.content_type]
    key = f"uploads/{session.session_id}/{uuid4().hex}{extension}"
    upload_url = get_s3_client().presign_put(key, body.content_type)
    log.info("Prepared upload", session_id=session.session_id, key=key, content_type=body.content_type)
    return UploadResponse(
        session_id=session.session_id,
        upload_url=upload_url,
        key=key,
        content_type=body.content_type,
    )


@router.post("/uploads/complete", response_model=UploadCompleteResponse)
async def complete_upload(body: UploadCompleteRequest) -> UploadCompleteResponse:
    """Queue analysis for a video that has finished uploading to S3."""
    db = await get_mongodb_client()
    sessions = SessionsRepo(db)
    jobs = JobsRepo(db)
    session = await sessions.get(body.session_id)
    if session is None:
        log.warning("Rejected upload complete", session_id=body.session_id, reason="unknown_session")
        raise HTTPException(status_code=404, detail="Unknown session")

    job = Job(session_id=body.session_id, input_key=body.key, status="queued")
    await jobs.insert_one(job)
    await sessions.append_message(
        body.session_id,
        Message(role="user", video_keys=[body.key]),
    )
    await ensure_job_tool_messages(sessions, body.session_id, job.job_id)
    try:
        rabbit = await get_rabbitmq()
        await rabbit.publish(job.job_id)
    except Exception as exc:
        log.exception("Failed to publish analysis", job_id=job.job_id, session_id=body.session_id)
        await jobs.mark_failed(job.job_id, "Could not queue analysis")
        raise HTTPException(status_code=502, detail="Could not queue analysis") from exc
    log.info("Queued analysis", job_id=job.job_id, session_id=body.session_id)
    return UploadCompleteResponse(job_id=job.job_id)


@router.post("/chat")
async def chat(body: ChatRequest) -> StreamingResponse:
    """Stream one chat turn. Blocks on and reports the latest job's progress and coaching
    before taking a new message: the user can't do anything else on this session until a
    queued video has been analysed."""
    db = await get_mongodb_client()
    sessions = SessionsRepo(db)

    async def events():
        session_id = body.session_id
        if not session_id:
            session = Session()
            await sessions.insert_one(session)
            session_id = session.session_id
            yield sse("session", {"session_id": session_id})
        async for frame in chat_events(session_id, body.text):
            yield frame

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

"""OpenRouter streaming and session chat/coaching helpers."""

import asyncio
import base64
import json
from typing import AsyncIterator, Optional

import cv2
import numpy as np
from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError, AsyncOpenAI

from src.config import get_settings
from src.logger import get_logger
from src.models.job import Job
from src.models.session import Message, Session, ToolCall
from src.repo.jobs import JobsRepo
from src.repo.sessions import SessionsRepo
from src.services.aws import get_s3_client
from src.services.mongodb import get_mongodb_client

settings = get_settings()
log = get_logger("server")

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=settings.openrouter_api_key or "missing",
)


SYSTEM_PROMPT = """You are an experienced, practical strength-training coach
specializing in squat technique, chatting directly with the lifter.

Chat normally about anything related to training, filming, or the user's
squat. If the user wants a form check, tell them to upload a video; you do
not trigger or control uploads yourself.

When a tool message shows {"status":"queued"}, tell the user their video is
queued for analysis and ask them to wait. Do not invent form feedback yet.

When the conversation includes analysis results (rep measurements and
images attached to a tool message), assess the user's squat using ONLY
those numerical measurements and images. Prioritize meaningful form
problems over biomechanical perfection.

INPUT:
Each rep contains down_ms, bottom_ms, up_ms and measurements at
standing, bottom, drive and finish:
knee_deg, hip_deg, back_from_vertical_deg, hip_below_knee,
hip_knee_gap, knee_forward, nose_below_ear, confidence.

==================================================
1. FORM CHECKLIST
==================================================

Evaluate every repetition independently, then compare repetitions.

DEPTH:
- Use hip_below_knee, hip_knee_gap and knee_deg.
- Knee angle below 90: substantial flexion.
- 90-100: moderate flexion.
- 100-110: potentially shallow.
- Above 110: strong indication of limited knee flexion.
- Look for insufficient or progressively reduced depth.
- Never use knee angle alone to determine depth.

BACK INCLINATION (back_from_vertical_deg):
- 0-30: generally reasonable.
- 30-45: noticeable forward inclination.
- 45-60: significant forward inclination.
- Above 60: strong warning requiring closer assessment.
- Look for sudden changes and excessive inclination.
- Consider body proportions and squat style. Do not enforce a vertical back.

KNEES AND HIPS:
- Check progressive flexion during descent and extension during ascent.
- Compare bottom and finish positions.
- Look for incomplete extension, abrupt changes or inconsistent movement.
- Normal forward knee travel is acceptable.

FINISH:
- Compare standing and finish knee/hip angles.
- Check whether the user returns to their initial standing position.

REP SPEED:
Use down_ms, bottom_ms and up_ms.
- Check descent control and upward drive.
- Identify unusually rushed descents or significantly slower drives.
- Slower reps alone do not indicate bad form.
- check for relative speed for between rep analysis, dont be nitpicky on very less difference only notify when the speed significantly changes in the order of seconds.

WEIGHT IN THE PICTURES:
- Look at the images for a barbell, plates, dumbbells, or kettlebells.
- If none are visible, the set is bodyweight. Do not tell the user to add, reduce, or maintain a load.
- If weights are visible, give load advice from the form. Do not guess the number on the plates.
- If the images are unclear, say you cannot tell whether they used weight.

==================================================
2. REP CONSISTENCY AND FATIGUE
==================================================

Compare the first and last repetitions, especially:
- Squat depth.
- Bottom knee and hip angles.
- Back inclination.
- Finish position.
- Upward drive duration.
- Overall movement control.

GOOD:
Repetitions remain reasonably consistent without meaningful deterioration.

WARNING:
One noticeable deviation appears, but movement control is maintained.

BAD:
Multiple indicators deteriorate together, suggesting technique breakdown.

Consistency does not automatically mean good form. Repeating the same
poor movement is still poor form.

==================================================
3. MEASUREMENT RELIABILITY
==================================================

- Use confidence to judge measurement reliability.
- Low-confidence measurements are not definitive.
- Ignore small differences caused by measurement noise.
- Do not assume the sign conventions of hip_knee_gap or knee_forward
  unless explicitly established.
- Missing or unreliable data is UNKNOWN, never PASS.
- Do not invent observations.
- If visual and numerical evidence disagree, acknowledge uncertainty.

The numerical ranges are screening heuristics, not absolute
biomechanical or medical thresholds.

==================================================
4. OVERALL CLASSIFICATION (do not include in chat this is for your reference only)
==================================================

GOOD:
Controlled movement, acceptable depth and posture, consistent reps,
and no meaningful technique breakdown.

ACCEPTABLE:
Generally good technique with minor deviations. Maintain current weight.

NEEDS IMPROVEMENT:
Meaningful technique problems requiring correction. Reduce weight
if necessary to restore control.

POOR:
Multiple significant failures or clear loss of control. Recommend
stopping the set or substantially reducing weight.

DECISION RULES:
- Do not default to GOOD.
- Do not let good measurements cancel significant failures.
- Evaluate individual reps before judging the set.
- Multiple concerning indicators increase severity.
- Do not penalize normal human variation.
- Do not manufacture problems or praise.

==================================================
5. COACHING RECOMMENDATIONS
==================================================

GOOD:
Acknowledge good technique. If reps are controlled and consistent,
suggest a small weight increase next session.

ACCEPTABLE:
Acknowledge overall performance. Mention minor issues to monitor
and recommend maintaining weight.

NEEDS IMPROVEMENT:
Identify the most important corrections. Recommend reducing weight
if technique deteriorates under the current load.

POOR:
Explain the breakdown and recommend stopping or reducing weight.

Never recommend increasing weight simply because all reps were completed.
Load advice applies only when weights are visible in the pictures. For a bodyweight set, coach the movement instead.

==================================================
6. COMMUNICATION
==================================================

- Use plain, simple English.
- Be direct, honest and constructive.
- Never expose angles, milliseconds, percentages or raw measurements.
- Explain what happened and what to fix.
- Mention only 1-2 important observations.
- Do not nitpick minor deviations or minute speed changes.
- Do not manufacture praise.
- If evidence is insufficient, say so.

Never mention stance, bar path, toe angle, heel lift or a rounded lower back.

Your objective is to help the user progressively lift heavier while
maintaining controlled, consistent and safe technique.
"""


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def series_frames(session: Session | None, job_id: str) -> list[dict]:
    if session is None:
        return []
    for item in session.messages:
        if item.role == "tool" and item.job_id == job_id:
            return item.series
    return []


def ready_payload(job: Job, series: list[dict] | None = None) -> dict:
    s3 = get_s3_client()
    overlay = s3.presign_get(job.artifacts.overlay) if job.artifacts.overlay else None
    images = [s3.presign_get(key) for key in job.artifacts.images]
    return {
        "job_id": job.job_id,
        "status": job.status,
        "error": job.error,
        "reps": job.reps,
        "series": series or [],
        "overlay_url": overlay,
        "image_urls": images,
        "reply": job.reply,
    }


async def ensure_system(sessions: SessionsRepo, session: Session, content: str) -> None:
    """Keep the system prompt as the first stored session message."""
    if session.messages and session.messages[0].role == "system":
        if session.messages[0].content == content:
            return
        await sessions.set_system(session.session_id, content)
        session.messages[0].content = content
        return
    message = Message(role="system", content=content)
    await sessions.prepend_message(session.session_id, message)
    session.messages.insert(0, message)


_MODEL_IMAGE_HEIGHT = 480


def _fit_height(image: np.ndarray, max_height: int) -> np.ndarray:
    height, width = image.shape[:2]
    if height <= max_height:
        return image
    scale = max_height / height
    return cv2.resize(image, (max(1, int(width * scale)), max_height), interpolation=cv2.INTER_AREA)


def image_data_url(key: str, body: bytes) -> str:
    """JPEG data URL, scaled down so a vision request stays under provider size limits."""
    decoded = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        encoded = base64.b64encode(body).decode("ascii")
        media = "image/png" if key.endswith(".png") else "image/jpeg"
        return f"data:{media};base64,{encoded}"
    ok, jpg = cv2.imencode(
        ".jpg",
        _fit_height(decoded, _MODEL_IMAGE_HEIGHT),
        [int(cv2.IMWRITE_JPEG_QUALITY), 80],
    )
    if not ok:
        encoded = base64.b64encode(body).decode("ascii")
        media = "image/png" if key.endswith(".png") else "image/jpeg"
        return f"data:{media};base64,{encoded}"
    encoded = base64.b64encode(jpg.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


_STILL_ORDER = (
    ("_standing.png", "Standing"),
    ("_down.png", "Down"),
    ("_seated.png", "Seated"),
    ("_drive.png", "Drive up"),
)


def _still_rank(key: str) -> int:
    lower = key.lower()
    for index, (suffix, _label) in enumerate(_STILL_ORDER):
        if lower.endswith(suffix):
            return index
    return len(_STILL_ORDER)


def _still_label(key: str) -> str:
    lower = key.lower()
    for suffix, label in _STILL_ORDER:
        if lower.endswith(suffix):
            return label
    return "Rep image"


def _first_rep_keys(keys: list[str]) -> list[str]:
    first = [key for key in keys if "_rep1_" in key]
    chosen = first or list(keys[:4])
    return sorted(chosen, key=_still_rank)


async def history_for_model(messages: list[Message]) -> list[dict]:
    """Session messages for the model. Reps and images only; series stays off this list."""
    rendered: list[dict] = []
    s3 = None
    for item in messages:
        if item.role == "tool":
            rendered.append(
                {
                    "role": "tool",
                    "tool_call_id": item.tool_call_id or "",
                    "content": item.content or "",
                }
            )
            if item.image_keys:
                if s3 is None:
                    s3 = get_s3_client()
                parts: list[dict] = []
                for key in _first_rep_keys(item.image_keys):
                    parts.append({"type": "text", "text": _still_label(key)})
                    url = image_data_url(key, await s3.get_bytes(key))
                    parts.append({"type": "image_url", "image_url": {"url": url}})
                rendered.append({"role": "user", "content": parts})
            continue
        if item.role == "user" and not item.content:
            continue
        message: dict = {"role": item.role, "content": item.content}
        if item.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in item.tool_calls
            ]
            if not item.content:
                message["content"] = None
        rendered.append(message)
    return rendered


def chat_branch(job: Job | None) -> str:
    if job and job.status in ("queued", "running"):
        return "resume"
    if job and job.status == "done" and not job.reply:
        return "coach"
    if job and job.status == "failed" and job.error and not job.reply:
        return "failed"
    return "turn"


def _tool_call_id(job_id: str) -> str:
    return f"call_{job_id.replace('-', '')[:24]}"


async def ensure_job_tool_messages(sessions: SessionsRepo, session_id: str, job_id: str) -> None:
    """Make sure the assistant tool call + queued tool result exist for this job.

    /uploads/complete auto-queues analysis without an LLM tool call, but coaching and
    series updates still target the tool message by job_id. Create the placeholder once.
    """
    session = await sessions.get(session_id)
    if session is None:
        return
    if any(item.role == "tool" and item.job_id == job_id for item in session.messages):
        return
    call_id = _tool_call_id(job_id)
    assistant = Message(
        role="assistant",
        tool_calls=[ToolCall(id=call_id, name="start_analysis", arguments="{}")],
    )
    tool = Message(
        role="tool",
        content=json.dumps({"status": "queued"}),
        tool_call_id=call_id,
        job_id=job_id,
    )
    await sessions.append_message(session_id, assistant)
    await sessions.append_message(session_id, tool)


def _tool_pair_start(messages: list[Message], job_id: str) -> int | None:
    """Index of the assistant tool_calls message for this job, or the tool message itself."""
    for index, item in enumerate(messages):
        if item.role != "tool" or item.job_id != job_id:
            continue
        if index > 0 and messages[index - 1].role == "assistant" and messages[index - 1].tool_calls:
            return index - 1
        return index
    return None


def needs_wait_ack(session: Session, job_id: str) -> bool:
    """True until a wait text exists for this job (just before or after its tool pair)."""
    start = _tool_pair_start(session.messages, job_id)
    if start is None:
        return False
    if start > 0:
        prior = session.messages[start - 1]
        if prior.role == "assistant" and prior.content and not prior.tool_calls:
            return False
    tool_idx = next(
        (i for i, item in enumerate(session.messages) if item.role == "tool" and item.job_id == job_id),
        start,
    )
    for item in session.messages[tool_idx + 1 :]:
        if item.role == "tool":
            break
        if item.role == "assistant" and item.content and not item.tool_calls:
            return False
    return True


async def place_wait_before_tools(sessions: SessionsRepo, session_id: str, job_id: str) -> None:
    """Keep history as wait → tool_calls → tool → analysis (tool sits in between)."""
    session = await sessions.get(session_id)
    if session is None:
        return
    messages = list(session.messages)
    start = _tool_pair_start(messages, job_id)
    if start is None:
        return
    tool_idx = next(
        (i for i, item in enumerate(messages) if item.role == "tool" and item.job_id == job_id),
        None,
    )
    if tool_idx is None:
        return
    wait_idx = None
    for index in range(tool_idx + 1, len(messages)):
        item = messages[index]
        if item.role == "tool":
            break
        if item.role == "assistant" and item.content and not item.tool_calls:
            wait_idx = index
            break
    if wait_idx is None or wait_idx < start:
        return
    wait = messages.pop(wait_idx)
    messages.insert(start, wait)
    await sessions.set_messages(session_id, messages)


async def save_user_note(
    sessions: SessionsRepo,
    session: Session,
    text: str,
    job: Job | None,
) -> Session:
    """Persist text sent with a video upload even when /chat is blocked on that job.

    /uploads/complete already wrote the tool pair, so insert the note just before it
    (and before any wait-ack sitting in front of it).
    """
    trimmed = text.strip() if text else ""
    if not trimmed:
        return session
    messages = list(session.messages)

    def fill_video_note(at: int) -> bool:
        """Put the note on the upload's user message instead of adding a second bubble."""
        if at <= 0:
            return False
        prior = messages[at - 1]
        if prior.role != "user" or prior.content or not prior.video_keys:
            return False
        prior.content = trimmed
        return True

    def already_present(at: int) -> bool:
        return at > 0 and messages[at - 1].role == "user" and messages[at - 1].content == trimmed

    insert_at: int | None = None
    if job is not None:
        start = _tool_pair_start(messages, job.job_id)
        if start is not None:
            if fill_video_note(start):
                await sessions.set_messages(session.session_id, messages)
                session.messages = messages
                return session
            if already_present(start):
                return session
            prior = messages[start - 1] if start > 0 else None
            if (
                prior
                and prior.role == "assistant"
                and prior.content
                and not prior.tool_calls
            ):
                insert_at = start - 1
            else:
                insert_at = start
            if fill_video_note(insert_at):
                await sessions.set_messages(session.session_id, messages)
                session.messages = messages
                return session
            if already_present(insert_at):
                return session

    if insert_at is None:
        if messages and messages[-1].role == "user" and messages[-1].content == trimmed:
            return session
        message = Message(role="user", content=trimmed)
        await sessions.append_message(session.session_id, message)
        session.messages.append(message)
        return session

    message = Message(role="user", content=trimmed)
    messages.insert(insert_at, message)
    await sessions.set_messages(session.session_id, messages)
    session.messages = messages
    return session


async def chat_events(session_id: str, text: str) -> AsyncIterator[str]:
    if not settings.openrouter_api_key:
        log.error("Rejected chat", session_id=session_id, reason="openrouter_missing")
        yield sse("error", {"detail": "OpenRouter is not configured"})
        return

    db = await get_mongodb_client()
    sessions = SessionsRepo(db)
    jobs = JobsRepo(db)
    session = await sessions.get(session_id)
    if session is None:
        log.warning("Rejected chat", session_id=session_id, reason="unknown_session")
        yield sse("error", {"detail": "Unknown session"})
        return

    job = await jobs.get_latest_for_session(session_id)
    branch = chat_branch(job)
    log.info("Handling chat", session_id=session_id, branch=branch, job_id=job.job_id if job else None)

    # Block here while a video is being analysed. The user cannot do anything else
    # with this session until the job finishes (or fails). Still save any note they
    # sent with the upload so it shows up in history and coaching context.
    if branch in ("resume", "coach", "failed"):
        session = await save_user_note(sessions, session, text, job)
    if branch == "resume":
        async for frame in follow_job(job.job_id, jobs, sessions, session_id):
            yield frame
        return
    if branch == "coach":
        async for frame in coach_stream(session, job, sessions):
            yield frame
        return
    if branch == "failed":
        await reveal_analysis(sessions, session_id, job)
        yield sse("job", {"job_id": job.job_id, "status": "failed", "error": job.error})
        yield sse("ready", {"status": "failed", "error": job.error})
        return

    async for frame in model_turn(session, text, sessions):
        yield frame


async def wait_ack_stream(
    session: Session,
    job: Job,
    sessions: SessionsRepo,
) -> AsyncIterator[str]:
    """Ask the model to acknowledge the queued analysis before we block on progress."""
    await ensure_job_tool_messages(sessions, session.session_id, job.job_id)
    fresh = await sessions.get(session.session_id)
    if fresh is None or not needs_wait_ack(fresh, job.job_id):
        return
    await ensure_system(sessions, fresh, SYSTEM_PROMPT)
    messages = await history_for_model(fresh.messages)
    parts: list[str] = []
    try:
        async for text_piece in token_texts(messages):
            parts.append(text_piece)
            yield sse("delta", {"text": text_piece})
    except APIError as exc:
        yield sse("error", {"detail": _model_failure(exc)[1]})
        return
    reply = "".join(parts)
    if reply:
        await sessions.append_message(session.session_id, Message(role="assistant", content=reply))
        await place_wait_before_tools(sessions, session.session_id, job.job_id)
        log.info("Queued wait acknowledged", session_id=session.session_id, job_id=job.job_id)


async def follow_job(job_id: str, jobs: JobsRepo, sessions: SessionsRepo, session_id: str) -> AsyncIterator[str]:
    await ensure_job_tool_messages(sessions, session_id, job_id)
    job = await jobs.get(job_id)
    if job is None:
        log.warning("Job stream ended", job_id=job_id, reason="missing")
        yield sse("error", {"detail": "Job missing"})
        return

    yield sse(
        "job",
        {
            "job_id": job.job_id,
            "status": job.status,
            "progress": job.progress.model_dump(),
            "error": job.error,
        },
    )
    if job.status in ("queued", "running"):
        session = await sessions.get(session_id)
        if session is not None:
            async for frame in wait_ack_stream(session, job, sessions):
                yield frame
                if frame.startswith("event: error"):
                    return

    last: Optional[tuple] = (job.status, job.progress.frame, job.progress.frame_count)
    if job.status == "done":
        log.info("Job stream finished", job_id=job_id, status=job.status)
        session = await sessions.get(session_id)
        async for frame in coach_stream(session, job, sessions):
            yield frame
        return
    if job.status == "failed":
        log.info("Job stream finished", job_id=job_id, status=job.status)
        await reveal_analysis(sessions, session_id, job)
        yield sse("ready", {"status": "failed", "error": job.error})
        return

    while True:
        job = await jobs.get(job_id)
        if job is None:
            log.warning("Job stream ended", job_id=job_id, reason="missing")
            yield sse("error", {"detail": "Job missing"})
            return
        snap = (job.status, job.progress.frame, job.progress.frame_count)
        if snap != last:
            last = snap
            yield sse(
                "job",
                {
                    "job_id": job.job_id,
                    "status": job.status,
                    "progress": job.progress.model_dump(),
                    "error": job.error,
                },
            )
            if job.status == "done":
                log.info("Job stream finished", job_id=job_id, status=job.status)
                session = await sessions.get(session_id)
                async for frame in coach_stream(session, job, sessions):
                    yield frame
                return
            if job.status == "failed":
                log.info("Job stream finished", job_id=job_id, status=job.status)
                await reveal_analysis(sessions, session_id, job)
                yield sse("ready", {"status": "failed", "error": job.error})
                return
        else:
            yield ": ping\n\n"
        await asyncio.sleep(0.5)


async def coach_stream(
    session: Session | None,
    job: Job,
    sessions: SessionsRepo,
) -> AsyncIterator[str]:
    if session is None:
        log.warning("Coaching stopped", job_id=job.job_id, reason="unknown_session")
        yield sse("error", {"detail": "Unknown session"})
        return
    await reveal_analysis(sessions, session.session_id, job)
    fresh = await sessions.get(session.session_id)
    if fresh is None:
        log.warning("Coaching stopped", session_id=session.session_id, reason="unknown_session")
        yield sse("error", {"detail": "Unknown session"})
        return
    await ensure_system(sessions, fresh, SYSTEM_PROMPT)
    messages = await history_for_model(fresh.messages)
    parts: list[str] = []
    try:
        async for text_piece in token_texts(messages):
            parts.append(text_piece)
            yield sse("delta", {"text": text_piece})
    except APIError as exc:
        yield sse("error", {"detail": _model_failure(exc)[1]})
        return
    reply = "".join(parts)
    if reply:
        await sessions.append_message(session.session_id, Message(role="assistant", content=reply))
    db = await get_mongodb_client()
    await JobsRepo(db).update_one({"job_id": job.job_id}, {"$set": {"reply": reply}})
    job.reply = reply
    yield sse("ready", ready_payload(job, series_frames(fresh, job.job_id)))


async def model_turn(
    session: Session,
    text: str,
    sessions: SessionsRepo,
) -> AsyncIterator[str]:
    """One plain conversational turn. Only reached when there is no job to resume or coach."""
    await ensure_system(sessions, session, SYSTEM_PROMPT)
    if text:
        message = Message(role="user", content=text)
        await sessions.append_message(session.session_id, message)
        session.messages.append(message)

    messages = await history_for_model(session.messages)

    parts: list[str] = []
    try:
        async for text_piece in token_texts(messages):
            parts.append(text_piece)
            yield sse("delta", {"text": text_piece})
    except APIError as exc:
        yield sse("error", {"detail": _model_failure(exc)[1]})
        return

    reply = "".join(parts)
    if reply:
        await sessions.append_message(session.session_id, Message(role="assistant", content=reply))
    log.info("Chat finished", session_id=session.session_id)
    yield sse("ready", {"status": "done"})


async def reveal_analysis(sessions: SessionsRepo, session_id: str, job: Job) -> None:
    """Point the tool message at reps and images. Series stays on that same message."""
    await ensure_job_tool_messages(sessions, session_id, job.job_id)
    if job.status == "done":
        reps = job.reps or {"reps": []}
        reason = reps.get("reason") if isinstance(reps, dict) else None
        if not reason:
            found = bool(isinstance(reps, dict) and reps.get("reps"))
            reason = "success" if found else "not enough confident knee angles to segment reps"
        content = json.dumps({"reason": reason, "reps": reps})
        image_keys = list(job.artifacts.images)
    else:
        content = json.dumps({"reason": job.error, "error": job.error})
        image_keys = []
    await sessions.set_tool_result(session_id, job.job_id, content, image_keys)


# OpenRouter error_type values that are safe to retry. Everything else is a bad request,
# auth, credits, or a policy block, and another attempt will fail the same way.
_RETRYABLE_TYPES = {
    "rate_limit_exceeded",
    "provider_overloaded",
    "provider_unavailable",
    "server",
    "timeout",
    "unmapped",
}
_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
_MAX_RETRY_WAIT = 8


def _status_code(body: dict, exc: APIError) -> int | None:
    code = body.get("code")
    if isinstance(code, int):
        return code
    if isinstance(code, str) and code.isdigit():
        return int(code)
    status = getattr(exc, "status_code", None)
    return status if isinstance(status, int) else None


def _retry_after(exc: APIError) -> float | None:
    response = getattr(exc, "response", None)
    if response is None:
        return None
    try:
        seconds = float(response.headers.get("retry-after"))
    except (TypeError, ValueError):
        return None
    if seconds <= 0:
        return None
    return seconds


def _error_type(metadata: dict) -> str | None:
    error_type = metadata.get("error_type")
    if isinstance(error_type, str) and error_type:
        return error_type
    return None


def _log_reason(body: dict, metadata: dict, exc: APIError, status: int | None, error_type: str | None) -> str:
    if error_type:
        return error_type
    raw = metadata.get("raw")
    if isinstance(raw, str):
        text = " ".join(raw.split())
        if text:
            return text[:160]
    message = body.get("message")
    if isinstance(message, str) and message.strip():
        text = " ".join(message.split())
        provider = metadata.get("provider_name")
        if isinstance(provider, str) and provider and provider not in text:
            text = f"{text} ({provider})"
        return text[:160]
    if isinstance(exc, APITimeoutError):
        return "timeout"
    if isinstance(exc, APIConnectionError):
        return "connection"
    if status is not None:
        return f"http_{status}"
    return "stream"


def _model_failure(exc: APIError) -> tuple[int | None, str, float | None]:
    """Status, log reason, and Retry-After seconds from an SDK error."""
    body = exc.body if isinstance(exc.body, dict) else {}
    metadata = body.get("metadata") if isinstance(body.get("metadata"), dict) else {}
    status = _status_code(body, exc)
    error_type = _error_type(metadata)
    return status, _log_reason(body, metadata, exc, status, error_type), _retry_after(exc)


def _can_retry(exc: APIError, status: int | None, retry_after: float | None) -> bool:
    body = exc.body if isinstance(exc.body, dict) else {}
    metadata = body.get("metadata") if isinstance(body.get("metadata"), dict) else {}
    error_type = _error_type(metadata)
    if isinstance(exc, APIConnectionError):
        return True
    # 402 is out of credits, except the in-flight budget case, which sends Retry-After.
    if error_type == "payment_required" or status == 402:
        return retry_after is not None
    if error_type in _RETRYABLE_TYPES:
        return True
    if error_type:
        return False
    if status in _RETRYABLE_STATUS:
        return True
    # Mid-stream SSE error with no typed reason. OpenRouter already returned HTTP 200.
    return not isinstance(exc, APIStatusError) and status is None


async def raw_deltas(messages: list[dict], tools=None, tool_choice=None):
    kwargs: dict = {
        "model": settings.openrouter_model,
        "messages": messages,
        "stream": True,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice or "auto"
    delay = 0.5
    for attempt in range(3):
        started = False
        try:
            stream = await client.chat.completions.create(**kwargs)
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content or delta.tool_calls:
                    started = True
                yield delta
            return
        except APIError as exc:
            status, reason, retry_after = _model_failure(exc)
            if started or attempt == 2 or not _can_retry(exc, status, retry_after):
                log.warning("Model request failed", model=settings.openrouter_model, status=status, reason=reason)
                raise
            wait = delay if retry_after is None else min(retry_after, _MAX_RETRY_WAIT)
            log.warning("Retrying model request", attempt=attempt + 1, status=status, reason=reason)
            await asyncio.sleep(wait)
            delay *= 2


async def token_texts(messages: list[dict]) -> AsyncIterator[str]:
    async for delta in raw_deltas(messages):
        if delta.content:
            yield delta.content

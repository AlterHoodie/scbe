"""Consume analysis jobs and run RTMPose."""

import asyncio
import json
import tempfile
from pathlib import Path

from src.config import get_settings
from src.logger import configure, get_logger
from src.pose import RTMPoseRunner, process_video
from src.repo.jobs import JobsRepo
from src.repo.sessions import SessionsRepo
from src.services.aws import get_s3_client
from src.services.mongodb import get_mongodb_client
from src.services.rabbitmq import get_rabbitmq

log = get_logger("worker")


def series_from_context(payload: dict) -> list[dict]:
    side = payload.get("side") or "left"
    series = []
    for frame in payload.get("frames") or []:
        item = frame.get(side) or {}
        series.append(
            {
                "t_ms": frame.get("t_ms"),
                "knee_deg": item.get("knee_deg"),
                "hip_deg": item.get("hip_deg"),
                "back_from_vertical_deg": item.get("back_from_vertical_deg"),
            }
        )
    return series


def _content_type(path: Path) -> str:
    if path.suffix == ".mp4":
        return "video/mp4"
    if path.suffix == ".png":
        return "image/png"
    return "application/json"


async def handle_job(job_id: str, runner: RTMPoseRunner, jobs: JobsRepo, sessions: SessionsRepo) -> None:
    if not await jobs.claim(job_id):
        log.info("Skipped job", job_id=job_id, reason="not_queued")
        return
    job = await jobs.get(job_id)
    if job is None:
        log.info("Skipped job", job_id=job_id, reason="missing")
        return
    log.info("Started analysis", job_id=job_id, session_id=job.session_id)
    s3 = get_s3_client()
    loop = asyncio.get_running_loop()

    def on_progress(frame: int, frame_count: int) -> None:
        asyncio.run_coroutine_threadsafe(jobs.update_progress(job_id, frame, frame_count), loop)

    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suffix = Path(job.input_key).suffix.lower()
            if suffix not in {".mp4", ".mov", ".webm"}:
                suffix = ".mp4"
            local = root / f"input{suffix}"
            out = root / "out"
            await s3.download_file(job.input_key, local)
            await asyncio.to_thread(process_video, local, runner, out, on_progress)

            overlay = next(out.glob("*_rtmpose.mp4"))
            context_path = next(out.glob("*_rtmpose_context.json"))
            reps_path = next(out.glob("*_rtmpose_reps.json"), None)
            images = sorted(out.glob("*_rtmpose_rep*.png"))

            overlay_key = f"jobs/{job_id}/overlay.mp4"
            context_key = f"jobs/{job_id}/context.json"
            image_keys = [f"jobs/{job_id}/{path.name}" for path in images]
            await s3.upload_file(overlay, overlay_key, _content_type(overlay))
            await s3.upload_file(context_path, context_key, "application/json")
            for path, key in zip(images, image_keys):
                await s3.upload_file(path, key, "image/png")

            context = json.loads(context_path.read_text(encoding="utf-8"))
            reps = json.loads(reps_path.read_text(encoding="utf-8")) if reps_path else {"reps": []}
            await sessions.set_tool_series(job.session_id, job_id, series_from_context(context))
            rep_count = len(reps.get("reps") or []) if isinstance(reps, dict) else 0
            await jobs.mark_done(
                job_id,
                {
                    "artifacts": {
                        "overlay": overlay_key,
                        "images": image_keys,
                        "context": context_key,
                    },
                    "reps": reps,
                    "progress": {"frame": context.get("frame_count") or 0, "frame_count": context.get("frame_count") or 0},
                },
            )
            log.info(
                "Finished analysis",
                job_id=job_id,
                frames=context.get("frame_count") or 0,
                reps=rep_count,
            )
    except Exception as exc:
        log.exception("Analysis failed", job_id=job_id)
        await jobs.mark_failed(job_id, str(exc))


async def run_worker() -> None:
    configure()
    db = await get_mongodb_client()
    jobs = JobsRepo(db)
    sessions = SessionsRepo(db)
    rabbit = await get_rabbitmq()
    runner = RTMPoseRunner()
    log.info("Worker ready", queue=get_settings().rabbitmq_queue)
    try:
        await rabbit.consume(lambda job_id: handle_job(job_id, runner, jobs, sessions))
    finally:
        runner.close()

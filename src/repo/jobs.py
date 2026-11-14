"""Jobs collection."""

import time

from src.models.job import Job
from src.repo.base import BaseRepo
from src.services.mongodb import MongoDBClient


class JobsRepo(BaseRepo):
    def __init__(self, db_client: MongoDBClient):
        super().__init__(db_client, "jobs", Job)

    async def get(self, job_id: str) -> Job | None:
        return await self.find_one({"job_id": job_id})

    async def get_latest_for_session(self, session_id: str) -> Job | None:
        jobs = await self.find({"session_id": session_id}, sort=[("created_at", -1)], limit=1)
        return jobs[0] if jobs else None

    async def claim(self, job_id: str) -> bool:
        """Move queued -> running. True only when this worker won the update."""
        collection = await self._get_collection()
        result = await collection.update_one(
            {"job_id": job_id, "status": "queued"},
            {"$set": {"status": "running", "updated_at": int(time.time())}},
        )
        return result.modified_count == 1

    async def update_progress(self, job_id: str, frame: int, frame_count: int) -> None:
        await self.update_one(
            {"job_id": job_id, "status": "running"},
            {"$set": {"progress": {"frame": frame, "frame_count": frame_count}}},
        )

    async def mark_done(self, job_id: str, fields: dict) -> None:
        await self.update_one(
            {"job_id": job_id},
            {"$set": {"status": "done", "error": None, **fields}},
        )

    async def mark_failed(self, job_id: str, error: str) -> None:
        await self.update_one(
            {"job_id": job_id},
            {"$set": {"status": "failed", "error": error}},
        )

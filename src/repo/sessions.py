"""Sessions collection."""

import time

from src.models.session import Message, Session
from src.repo.base import BaseRepo
from src.services.mongodb import MongoDBClient


class SessionsRepo(BaseRepo):
    def __init__(self, db_client: MongoDBClient):
        super().__init__(db_client, "sessions", Session)

    async def get(self, session_id: str) -> Session | None:
        return await self.find_one({"session_id": session_id})

    async def list_all(self, limit: int = 100) -> list[Session]:
        return await self.find({}, sort=[("updated_at", -1)], limit=limit)

    async def append_message(self, session_id: str, message: Message) -> None:
        await self.update_one(
            {"session_id": session_id},
            {"$push": {"messages": message.persisted()}},
        )

    async def set_messages(self, session_id: str, messages: list[Message]) -> None:
        await self.update_one(
            {"session_id": session_id},
            {"$set": {"messages": [message.persisted() for message in messages]}},
        )

    async def prepend_message(self, session_id: str, message: Message) -> None:
        await self.update_one(
            {"session_id": session_id},
            {"$push": {"messages": {"$each": [message.persisted()], "$position": 0}}},
        )

    async def set_system(self, session_id: str, content: str) -> None:
        collection = await self._get_collection()
        await collection.update_one(
            {"session_id": session_id, "messages.0.role": "system"},
            {"$set": {"messages.0.content": content, "updated_at": int(time.time())}},
        )

    async def set_tool_series(self, session_id: str, job_id: str, frames: list[dict]) -> None:
        collection = await self._get_collection()
        await collection.update_one(
            {"session_id": session_id},
            {
                "$set": {
                    "messages.$[m].series": frames,
                    "updated_at": int(time.time()),
                }
            },
            array_filters=[{"m.role": "tool", "m.job_id": job_id}],
        )

    async def set_tool_result(
        self,
        session_id: str,
        job_id: str,
        content: str,
        image_keys: list[str],
    ) -> None:
        collection = await self._get_collection()
        await collection.update_one(
            {"session_id": session_id},
            {
                "$set": {
                    "messages.$[m].content": content,
                    "messages.$[m].image_keys": image_keys,
                    "updated_at": int(time.time()),
                }
            },
            array_filters=[{"m.role": "tool", "m.job_id": job_id}],
        )

"""MongoDB client."""

from typing import Optional

from pymongo import AsyncMongoClient

from src.config import get_settings

settings = get_settings()


class MongoDBClient:
    def __init__(self) -> None:
        self.client: Optional[AsyncMongoClient] = None
        self._db = None
        self.db_name = settings.mongodb_database

    async def connect(self) -> None:
        self.client = AsyncMongoClient(settings.mongodb_uri)
        self._db = self.client[self.db_name]

    async def disconnect(self) -> None:
        if self.client:
            await self.client.close()
            self.client = None
            self._db = None

    async def get_collection(self, collection_name: str):
        if self._db is None:
            await self.connect()
        return self._db[collection_name]


_mongodb_client: Optional[MongoDBClient] = None


async def get_mongodb_client() -> MongoDBClient:
    global _mongodb_client
    if _mongodb_client is None:
        _mongodb_client = MongoDBClient()
        await _mongodb_client.connect()
    return _mongodb_client

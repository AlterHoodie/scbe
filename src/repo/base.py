"""
Base repository: generic async CRUD over a MongoDB collection.
Strips _id before returning. Adds created_at/updated_at on insert and updated_at on update.
"""

import time
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from pydantic import BaseModel

from src.services.mongodb import MongoDBClient

T = TypeVar("T", bound=BaseModel)


class BaseRepo:
    def __init__(
        self,
        db_client: MongoDBClient,
        collection_name: str,
        model_class: Type[T],
    ):
        self.db_client = db_client
        self.collection_name = collection_name
        self.model_class = model_class

    async def _get_collection(self):
        return await self.db_client.get_collection(self.collection_name)

    def _to_model(self, doc: Optional[Dict[str, Any]]) -> Optional[T]:
        if doc is None:
            return None
        doc = dict(doc)
        doc.pop("_id", None)
        return self.model_class(**doc)

    async def find_one(self, query: dict) -> Optional[T]:
        collection = await self._get_collection()
        doc = await collection.find_one(query)
        return self._to_model(doc)

    async def find(
        self,
        query: dict,
        sort: Optional[List[tuple]] = None,
        limit: Optional[int] = None,
        skip: Optional[int] = None,
    ) -> List[T]:
        collection = await self._get_collection()
        cursor = collection.find(query)
        if sort:
            cursor = cursor.sort(sort)
        if skip is not None and skip > 0:
            cursor = cursor.skip(skip)
        if limit is not None:
            cursor = cursor.limit(limit)
        docs = await cursor.to_list(length=None)
        return [self._to_model(doc) for doc in docs]  # type: ignore[misc]

    def _payload_with_timestamps(self, doc: Union[T, Dict[str, Any]], now: int) -> Dict[str, Any]:
        if isinstance(doc, dict):
            payload = dict(doc)
        else:
            payload = doc.model_dump()
        if "created_at" not in payload:
            payload["created_at"] = now
        if "updated_at" not in payload:
            payload["updated_at"] = now
        return payload

    async def insert_one(self, doc: Union[T, Dict[str, Any]]) -> bool:
        collection = await self._get_collection()
        now = int(time.time())
        payload = self._payload_with_timestamps(doc, now)
        result = await collection.insert_one(payload)
        return result.acknowledged

    def _with_updated_at(self, update: dict) -> dict:
        now = int(time.time())
        update = dict(update)
        if "$set" not in update:
            update["$set"] = {}
        update["$set"] = dict(update["$set"])
        update["$set"]["updated_at"] = now
        return update

    async def update_one(self, query: dict, update: dict) -> bool:
        collection = await self._get_collection()
        update = self._with_updated_at(update)
        result = await collection.update_one(query, update)
        return result.acknowledged

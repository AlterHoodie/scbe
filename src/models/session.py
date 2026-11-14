"""Chat session document stored in the sessions collection."""

import time
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: str = "{}"


class Message(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: Optional[str] = None
    job_id: Optional[str] = None
    image_keys: list[str] = Field(default_factory=list)
    video_keys: list[str] = Field(default_factory=list)
    series: list[dict] = Field(default_factory=list)

    def persisted(self) -> dict:
        data: dict = {"role": self.role}
        if self.content is not None:
            data["content"] = self.content
        if self.tool_calls:
            data["tool_calls"] = [call.model_dump() for call in self.tool_calls]
        if self.tool_call_id:
            data["tool_call_id"] = self.tool_call_id
        if self.job_id:
            data["job_id"] = self.job_id
        if self.image_keys:
            data["image_keys"] = self.image_keys
        if self.video_keys:
            data["video_keys"] = self.video_keys
        if self.series:
            data["series"] = self.series
        return data


class Session(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    messages: list[Message] = Field(default_factory=list)
    created_at: int = Field(default_factory=lambda: int(time.time()))
    updated_at: int = Field(default_factory=lambda: int(time.time()))

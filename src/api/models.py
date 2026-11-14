"""Request and response models."""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class UploadRequest(BaseModel):
    session_id: Optional[str] = None
    content_type: Literal["video/mp4", "video/quicktime", "video/webm"] = "video/mp4"


class UploadResponse(BaseModel):
    session_id: str
    upload_url: str
    key: str
    content_type: Literal["video/mp4", "video/quicktime", "video/webm"]


class UploadCompleteRequest(BaseModel):
    session_id: str
    key: str


class UploadCompleteResponse(BaseModel):
    job_id: str


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    text: str = Field(default="")


class SessionSummary(BaseModel):
    session_id: str
    created_at: int
    updated_at: int
    preview: Optional[str] = None


class SessionListResponse(BaseModel):
    sessions: list[SessionSummary]


class TranscriptTextMessage(BaseModel):
    id: str
    kind: Literal["text"] = "text"
    role: Literal["user", "assistant"]
    text: str
    video_urls: list[str] = Field(default_factory=list)


class TranscriptAnalysisMessage(BaseModel):
    id: str
    kind: Literal["analysis"] = "analysis"
    job: dict[str, Any]


class SessionProgress(BaseModel):
    frame: int
    frame_count: int


class SessionDetailResponse(BaseModel):
    session_id: str
    created_at: int
    updated_at: int
    phase: Literal["idle", "queued", "running", "coaching", "done", "failed"]
    messages: list[TranscriptTextMessage | TranscriptAnalysisMessage]
    progress: Optional[SessionProgress] = None

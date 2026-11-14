"""Job document stored in the jobs collection."""

import time
from typing import Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class Progress(BaseModel):
    frame: int = 0
    frame_count: int = 0


class Artifacts(BaseModel):
    overlay: Optional[str] = None
    images: list[str] = Field(default_factory=list)
    context: Optional[str] = None


class Job(BaseModel):
    job_id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    status: Literal["queued", "running", "done", "failed"] = "queued"
    progress: Progress = Field(default_factory=Progress)
    input_key: str
    artifacts: Artifacts = Field(default_factory=Artifacts)
    reps: Optional[dict] = None
    reply: Optional[str] = None
    error: Optional[str] = None
    created_at: int = Field(default_factory=lambda: int(time.time()))
    updated_at: int = Field(default_factory=lambda: int(time.time()))

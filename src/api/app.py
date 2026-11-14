"""FastAPI application."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes.chat_completion import router as chat_router
from src.config import get_settings
from src.logger import configure, get_logger
from src.services.aws import get_s3_client
from src.services.mongodb import get_mongodb_client
from src.services.rabbitmq import get_rabbitmq

settings = get_settings()
log = get_logger("server")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    configure()
    await get_mongodb_client()
    get_s3_client()
    await get_rabbitmq()
    log.info("API ready")
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Squat analysis", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.cors_origin],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    app.include_router(chat_router)
    return app


app = create_app()

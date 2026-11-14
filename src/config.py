"""Environment settings for the API and the worker."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_database: str = "assignment"

    rabbitmq_url: str = "amqp://guest:guest@localhost:5672/"
    rabbitmq_queue: str = "analysis"

    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"
    aws_endpoint_url: str = ""
    s3_bucket: str = ""

    openrouter_api_key: str = ""
    openrouter_model: str = "openai/gpt-4.1-mini"

    cors_origin: str = "http://localhost:3000"


@lru_cache
def get_settings() -> Settings:
    return Settings()

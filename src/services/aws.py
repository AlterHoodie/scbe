"""S3 uploads, downloads, and presigned URLs."""

import asyncio
from pathlib import Path
from typing import Optional

import boto3
from botocore.config import Config

from src.config import get_settings

settings = get_settings()


class S3Client:
    def __init__(self) -> None:
        self.client = boto3.client(
            "s3",
            aws_access_key_id=settings.aws_access_key_id or None,
            aws_secret_access_key=settings.aws_secret_access_key or None,
            region_name=settings.aws_region,
            endpoint_url=settings.aws_endpoint_url or None,
            config=Config(s3={"addressing_style": "path"}),
        )
        self.bucket = settings.s3_bucket
        if settings.aws_endpoint_url:
            self.allow_browser_uploads()

    def allow_browser_uploads(self) -> None:
        """Let the frontend PUT a video and GET presigned results from this origin."""
        self.client.put_bucket_cors(
            Bucket=self.bucket,
            CORSConfiguration={
                "CORSRules": [
                    {
                        "AllowedOrigins": [settings.cors_origin],
                        "AllowedMethods": ["GET", "HEAD", "PUT"],
                        "AllowedHeaders": ["*"],
                        "ExposeHeaders": ["ETag"],
                        "MaxAgeSeconds": 3600,
                    }
                ]
            },
        )

    def presign_put(self, key: str, content_type: str = "video/mp4") -> str:
        return self.client.generate_presigned_url(
            "put_object",
            Params={"Bucket": self.bucket, "Key": key, "ContentType": content_type},
            ExpiresIn=3600,
        )

    def presign_get(self, key: str) -> str:
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=3600,
        )

    async def get_bytes(self, key: str) -> bytes:
        def read() -> bytes:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
            return response["Body"].read()

        return await asyncio.to_thread(read)

    async def download_file(self, key: str, path: Path) -> None:
        await asyncio.to_thread(self.client.download_file, self.bucket, key, str(path))

    async def upload_file(self, path: Path, key: str, content_type: str) -> None:
        await asyncio.to_thread(
            self.client.upload_file,
            str(path),
            self.bucket,
            key,
            ExtraArgs={"ContentType": content_type},
        )


_s3: Optional[S3Client] = None


def get_s3_client() -> S3Client:
    global _s3
    if _s3 is None:
        _s3 = S3Client()
    return _s3

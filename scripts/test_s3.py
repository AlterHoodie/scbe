"""Put and get a small object against the LocalStack S3 endpoint in .env.dev."""

import os
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

ENV_FILE = Path(__file__).resolve().parents[1] / ".env.dev"
NEEDED = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_REGION",
    "AWS_ENDPOINT_URL",
    "S3_BUCKET",
)


def load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key in NEEDED:
            values[key] = value
    missing = [key for key in NEEDED if not values.get(key)]
    if missing:
        raise SystemExit(f"missing in {path.name}: {', '.join(missing)}")
    return values


def main() -> None:
    env = load_env(ENV_FILE)
    endpoint = env["AWS_ENDPOINT_URL"]
    bucket = env["S3_BUCKET"]
    client = boto3.client(
        "s3",
        aws_access_key_id=env["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=env["AWS_SECRET_ACCESS_KEY"],
        region_name=env["AWS_REGION"],
        endpoint_url=endpoint,
        config=Config(s3={"addressing_style": "path"}),
    )

    print(f"endpoint {endpoint}")
    names = {item["Name"] for item in client.list_buckets().get("Buckets", [])}
    if bucket not in names:
        client.create_bucket(Bucket=bucket)
        print(f"created bucket {bucket}")
    else:
        print(f"bucket {bucket} exists")

    key = "_healthcheck/ping.txt"
    body = b"ok"
    client.put_object(Bucket=bucket, Key=key, Body=body, ContentType="text/plain")
    got = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    client.delete_object(Bucket=bucket, Key=key)
    if got != body:
        raise SystemExit("round trip returned unexpected bytes")
    print("put/get/delete ok")


if __name__ == "__main__":
    try:
        main()
    except (BotoCoreError, ClientError, OSError) as exc:
        raise SystemExit(f"localstack check failed: {exc}") from exc

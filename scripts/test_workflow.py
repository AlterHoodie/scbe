"""Walk one clip through the live API: upload, analysis, then the coach reply.

The API and worker must already be running:

    uv run -m src.main server
    uv run -m src.main worker
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--video", type=Path, help="mp4 to upload; a short clip is generated if omitted")
    parser.add_argument("--timeout", type=float, default=300, help="seconds allowed between SSE events")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    timeout = args.timeout

    health = get_json(f"{base}/health", timeout)
    if health.get("status") != "healthy":
        raise SystemExit(f"health check failed: {health}")
    print("health ok")

    upload = post_json(f"{base}/uploads", {}, timeout)
    session_id = upload["session_id"]
    upload_url = upload["upload_url"]
    print(f"session {session_id}")
    print(f"key {upload['key']}")

    video = args.video
    if video is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.close()
        video = Path(tmp.name)
        write_clip(video)
        print(f"generated {video}")
    elif not video.is_file():
        raise SystemExit(f"video not found: {video}")

    put_bytes(upload_url, video.read_bytes(), timeout)
    print(f"uploaded {video.stat().st_size} bytes")

    print("chat")
    ready = stream_chat(base, session_id, "Check this squat.", timeout)
    if ready.get("status") == "failed":
        raise SystemExit(f"analysis failed: {ready.get('error')}")
    if ready.get("status") != "done":
        raise SystemExit(f"analysis did not finish: {ready.get('status')}")
    print(f"job {ready.get('job_id')} done")

    print("coach")
    parts: list[str] = []
    ready = stream_chat(base, session_id, "", timeout, parts)
    reply = "".join(parts).strip()
    if not reply:
        raise SystemExit("coach reply was empty")
    print(reply)
    print("workflow ok")


def write_clip(path: Path) -> None:
    import cv2
    import numpy as np

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (320, 240))
    if not writer.isOpened():
        raise SystemExit("could not write a sample mp4")
    frame = np.full((240, 320, 3), (40, 80, 160), dtype=np.uint8)
    try:
        for _ in range(8):
            writer.write(frame)
    finally:
        writer.release()


def get_json(url: str, timeout: float) -> dict:
    return read_json(Request(url, method="GET"), timeout)


def post_json(url: str, body: dict, timeout: float) -> dict:
    payload = json.dumps(body).encode()
    request = Request(
        url,
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    return read_json(request, timeout)


def put_bytes(url: str, body: bytes, timeout: float) -> None:
    request = Request(url, data=body, method="PUT", headers={"Content-Type": "video/mp4"})
    try:
        with urlopen(request, timeout=timeout) as response:
            if response.status not in (200, 204):
                raise SystemExit(f"upload failed: HTTP {response.status}")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"upload failed: HTTP {exc.code} {detail}") from exc
    except URLError as exc:
        raise SystemExit(f"upload failed: {exc.reason}") from exc


def read_json(request: Request, timeout: float) -> dict:
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"{request.full_url} failed: HTTP {exc.code} {detail}") from exc
    except URLError as exc:
        raise SystemExit(f"{request.full_url} failed: {exc.reason}") from exc


def stream_chat(
    base: str,
    session_id: str,
    text: str,
    timeout: float,
    parts: list[str] | None = None,
) -> dict:
    payload = json.dumps({"session_id": session_id, "text": text}).encode()
    request = Request(
        f"{base}/chat",
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    ready: dict | None = None
    try:
        with urlopen(request, timeout=timeout) as response:
            for event, data in iter_sse(response, timeout):
                if event == "delta" and data.get("text"):
                    chunk = data["text"]
                    if parts is not None:
                        parts.append(chunk)
                    print(chunk, end="", flush=True)
                    continue
                print(f"{event}: {json.dumps(data)}")
                if event == "error":
                    raise SystemExit(f"chat error: {data.get('detail')}")
                if event == "ready":
                    ready = data
                    break
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"chat failed: HTTP {exc.code} {detail}") from exc
    except URLError as exc:
        raise SystemExit(f"chat failed: {exc.reason}") from exc
    if ready is None:
        raise SystemExit("chat ended before ready")
    return ready


def iter_sse(response, timeout: float):
    event = "message"
    data_lines: list[str] = []
    last_event = time.monotonic()
    while True:
        if time.monotonic() - last_event >= timeout:
            raise SystemExit("timed out waiting for the next event")
        raw = response.readline()
        if not raw:
            if data_lines:
                yield event, json.loads("\n".join(data_lines))
            return
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        if line == "":
            if data_lines:
                last_event = time.monotonic()
                yield event, json.loads("\n".join(data_lines))
            event = "message"
            data_lines = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            data_lines.append(line.split(":", 1)[1].lstrip())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)

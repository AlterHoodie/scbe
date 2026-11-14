"""Run the API server or the analysis worker."""

import asyncio
import sys

from src.logger import configure, get_logger

log = get_logger("server")


def main() -> None:
    configure()
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "server":
        import uvicorn

        log.info("Starting API server", host="0.0.0.0", port=8000)
        uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, log_config=None)
    elif mode == "worker":
        from src.worker import run_worker

        asyncio.run(run_worker())
    else:
        raise SystemExit("Usage: python -m src.main server|worker")


if __name__ == "__main__":
    main()

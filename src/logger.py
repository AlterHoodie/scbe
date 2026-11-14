"""Logs with one sentence and a few key=value pairs."""

import logging
import sys

_CONFIGURED = "_assignment_configured"


def _line(sentence: str, fields: dict) -> str:
    pairs = [f"{key}={value}" for key, value in fields.items() if value is not None]
    if not pairs:
        return sentence
    return f"{sentence} | {' '.join(pairs)}"


class Logger:
    def __init__(self, name: str) -> None:
        self._log = logging.getLogger(name)

    def info(self, sentence: str, **fields) -> None:
        self._log.info(_line(sentence, fields))

    def warning(self, sentence: str, **fields) -> None:
        self._log.warning(_line(sentence, fields))

    def error(self, sentence: str, **fields) -> None:
        self._log.error(_line(sentence, fields))

    def exception(self, sentence: str, **fields) -> None:
        self._log.exception(_line(sentence, fields))


def get_logger(name: str) -> Logger:
    return Logger(name)


def configure() -> None:
    root = logging.getLogger()
    if getattr(root, _CONFIGURED, False):
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    setattr(root, _CONFIGURED, True)

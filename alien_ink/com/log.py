"""Central alien-ink logger for CLI, local runs, and notebooks.

Progress lines stay human-readable (no timestamps/noise). INFO goes to
stdout; WARNING and above go to stderr. Configure once via
``configure()`` or env ``ALIEN_INK_LOG_LEVEL`` (default INFO).
"""

from __future__ import annotations

import logging
import os
import sys
from typing import TextIO

LOGGER_NAME = "alien_ink"
DEFAULT_LEVEL = "INFO"
_ENV_LEVEL = "ALIEN_INK_LOG_LEVEL"

_configured = False


class CleanFormatter(logging.Formatter):
    """Message-only for INFO/DEBUG; level prefix for WARNING+."""

    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()
        if record.levelno >= logging.WARNING:
            return f"{record.levelname}: {msg}"
        return msg


class _MaxLevelFilter(logging.Filter):
    """Pass records strictly below ``max_level`` (exclusive)."""

    def __init__(self, max_level: int) -> None:
        super().__init__()
        self.max_level = max_level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno < self.max_level


def _resolve_level(level: int | str | None) -> int:
    if level is None:
        raw = os.getenv(_ENV_LEVEL, DEFAULT_LEVEL)
        return getattr(logging, str(raw).upper(), logging.INFO)
    if isinstance(level, int):
        return level
    return getattr(logging, str(level).upper(), logging.INFO)


def configure(
    level: int | str | None = None,
    *,
    stream: TextIO | None = None,
    force: bool = False,
) -> logging.Logger:
    """Configure the package logger (idempotent unless ``force=True``).

    Parameters
    ----------
    level:
        Log level name or int. Defaults to ``ALIEN_INK_LOG_LEVEL`` or INFO.
    stream:
        If set, send all levels to this stream (handy in tests). Otherwise
        INFO/DEBUG → stdout and WARNING+ → stderr.
    force:
        Replace existing handlers even if already configured.
    """
    global _configured
    logger = logging.getLogger(LOGGER_NAME)

    if _configured and not force and logger.handlers:
        if level is not None:
            logger.setLevel(_resolve_level(level))
        return logger

    logger.handlers.clear()
    logger.setLevel(_resolve_level(level))
    logger.propagate = False

    fmt = CleanFormatter()

    if stream is not None:
        handler = logging.StreamHandler(stream)
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    else:
        out = logging.StreamHandler(sys.stdout)
        out.setLevel(logging.DEBUG)
        out.addFilter(_MaxLevelFilter(logging.WARNING))
        out.setFormatter(fmt)
        logger.addHandler(out)

        err = logging.StreamHandler(sys.stderr)
        err.setLevel(logging.WARNING)
        err.setFormatter(fmt)
        logger.addHandler(err)

    _configured = True
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return the package logger, or a child like ``alien_ink.hf.ds``."""
    configure()
    if name:
        child = name if name.startswith(LOGGER_NAME) else f"{LOGGER_NAME}.{name}"
        return logging.getLogger(child)
    return logging.getLogger(LOGGER_NAME)


def header(*, logger: logging.Logger | None = None) -> None:
    """Log the Alien Ink multi-line brand banner (starfield + wordmark)."""
    # Local import keeps ``alien_ink.__init__`` free of logging side effects.
    from alien_ink import HEADER

    log = logger or get_logger()
    for line in HEADER.strip("\n").splitlines():
        log.info(line.rstrip())
    log.info("")


def banner(title: str, *, logger: logging.Logger | None = None) -> None:
    """Section header: rule / ``:: title`` / rule."""
    log = logger or get_logger()
    log.info("----------------------------------------------------------------------")
    log.info(f":: {title}")
    log.info("----------------------------------------------------------------------")


def step(msg: str, *, logger: logging.Logger | None = None) -> None:
    """Progress step: ``>> msg``."""
    (logger or get_logger()).info(f">> {msg}")


def detail(msg: str, *, logger: logging.Logger | None = None) -> None:
    """Indented detail under a step."""
    (logger or get_logger()).info(f"   {msg}")


def blank(*, logger: logging.Logger | None = None) -> None:
    """Blank line (keeps narrative spacing in notebooks and terminals)."""
    (logger or get_logger()).info("")

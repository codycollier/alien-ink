"""Tests for the central alien-ink logger."""

from __future__ import annotations

import io
import logging

import pytest

from alien_ink import HEADER
from alien_ink.com import log as ink_log
from alien_ink.com.log import banner, blank, configure, detail, get_logger, header, step


@pytest.fixture(autouse=True)
def _reset_logger():
    """Isolate logger configuration across tests."""
    stream = io.StringIO()
    configure(level="DEBUG", stream=stream, force=True)
    yield stream
    root = logging.getLogger(ink_log.LOGGER_NAME)
    root.handlers.clear()
    ink_log._configured = False


def test_get_logger_returns_package_and_child():
    assert get_logger().name == "alien_ink"
    assert get_logger("hf.ds").name == "alien_ink.hf.ds"
    assert get_logger("alien_ink.wb").name == "alien_ink.wb"


def test_step_detail_banner_blank_style(_reset_logger):
    stream = _reset_logger
    log = get_logger("test")
    banner("Title", logger=log)
    step("hello", logger=log)
    detail("world", logger=log)
    blank(logger=log)

    text = stream.getvalue()
    assert ":: Title" in text
    assert ">> hello" in text
    assert "   world" in text
    assert "-" * 79 in text
    # blank line between narrative blocks
    assert "\n\n" in text or text.endswith("\n")


def test_header_emits_brand_wordmark(_reset_logger):
    stream = _reset_logger
    header(logger=get_logger("test"))
    text = stream.getvalue()
    assert "_    _" in text  # ALIEN wordmark top
    assert "|_ _|" in text  # INK wordmark fragment
    assert text.count("\n") >= len(HEADER.strip("\n").splitlines())


def test_warning_and_error_include_level_prefix(_reset_logger):
    stream = _reset_logger
    log = get_logger("test")
    log.warning("soft fail")
    log.error("hard fail")
    text = stream.getvalue()
    assert "WARNING: soft fail" in text
    assert "ERROR: hard fail" in text


def test_configure_respects_level(_reset_logger):
    stream = io.StringIO()
    configure(level="WARNING", stream=stream, force=True)
    log = get_logger("test")
    step("hidden", logger=log)
    log.warning("visible")
    text = stream.getvalue()
    assert ">> hidden" not in text
    assert "WARNING: visible" in text


def test_configure_idempotent_unless_force(_reset_logger):
    first = io.StringIO()
    configure(level="INFO", stream=first, force=True)
    configure(level="INFO", stream=io.StringIO(), force=False)
    get_logger().info("ping")
    assert "ping" in first.getvalue()


def test_env_level(monkeypatch, _reset_logger):
    monkeypatch.setenv("ALIEN_INK_LOG_LEVEL", "ERROR")
    stream = io.StringIO()
    configure(stream=stream, force=True)
    log = get_logger("test")
    log.warning("nope")
    log.error("yep")
    text = stream.getvalue()
    assert "nope" not in text
    assert "ERROR: yep" in text

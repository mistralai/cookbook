"""Tests for the best-effort browser opener.

``open_browser`` redirects file descriptor 2 around ``webbrowser.open`` so a
browserless launcher (xdg-open with no browser installed) cannot spew warnings
into the user's terminal. The redirect must be at the fd level -- the launcher
is a child process and inherits fd 2, not Python's ``sys.stderr`` wrapper -- and
it must always be undone, even when the open fails.
"""

from __future__ import annotations

import os

import pytest

from extract_bench.utils import browser
from extract_bench.utils.browser import open_browser


def _fd2_identity() -> tuple[int, int]:
    stat = os.fstat(2)
    return (stat.st_dev, stat.st_ino)


def _open_fd_count() -> int:
    """Number of open fds, or -1 where the platform does not expose them."""
    try:
        return len(os.listdir("/dev/fd"))
    except OSError:
        return -1


def test_suppresses_launcher_output_written_to_fd_2(
    capfd: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Noise a launcher writes straight to fd 2 never reaches the terminal."""

    def noisy_open(url: str) -> bool:
        os.write(2, b"xdg-open: no method available\n")
        return False

    monkeypatch.setattr(browser.webbrowser, "open", noisy_open)

    open_browser("http://localhost:8080")

    assert capfd.readouterr().err == ""


def test_restores_stderr_after_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """fd 2 points back at the original stream, and no fd is leaked."""
    monkeypatch.setattr(browser.webbrowser, "open", lambda url: True)
    before_identity = _fd2_identity()
    before_open_fds = _open_fd_count()

    for _ in range(10):
        open_browser("http://localhost:8080")

    assert _fd2_identity() == before_identity
    assert _open_fd_count() == before_open_fds


def test_restores_stderr_when_open_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A launcher blowing up is swallowed, and fd 2 still gets restored."""

    def exploding_open(url: str) -> bool:
        raise OSError("no browser")

    monkeypatch.setattr(browser.webbrowser, "open", exploding_open)
    before_identity = _fd2_identity()

    open_browser("http://localhost:8080")

    assert _fd2_identity() == before_identity


def test_still_opens_when_stderr_cannot_be_duplicated(monkeypatch: pytest.MonkeyPatch) -> None:
    """A closed fd 2 must not turn a best-effort open into a crash."""
    opened: list[str] = []
    monkeypatch.setattr(browser.webbrowser, "open", lambda url: opened.append(url) or True)

    def no_such_fd(fd: int) -> int:
        raise OSError(9, "Bad file descriptor")

    monkeypatch.setattr(browser.os, "dup", no_such_fd)

    open_browser("http://localhost:8080")

    assert opened == ["http://localhost:8080"]

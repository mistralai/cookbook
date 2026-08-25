"""Thin wrapper around the stdlib webbrowser module."""

import os
import sys
import webbrowser


def open_browser(url: str) -> None:
    """Best-effort browser open. Callers should print the URL first.

    Never raises: a failed open is not worth failing the command over.

    Not safe to call while another thread may be writing to stderr, since the
    suppression below is process-wide and lasts as long as webbrowser.open()
    does -- which for a foreground BROWSER is until the browser itself exits.
    """
    # webbrowser.open() shells out to a platform launcher (e.g., xdg-open on
    # Linux). When no browser is available, the launcher often prints noisy
    # warnings to stderr even though BackgroundBrowser returns immediately.
    # To properly suppress this, temporarily redirect file descriptor 2 (stderr)
    # to /dev/null for the call. redirect_stderr() is not enough: the child
    # process inherits the real fd 2, not Python's sys.stderr wrapper.
    try:
        sys.stderr.flush()
        stderr_fd = os.dup(2)
    except (OSError, ValueError):
        # No usable stderr to redirect (e.g., the caller closed fd 2), so there
        # is nothing to suppress either.
        _open_quietly(url)
        return

    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), 2)
            _open_quietly(url)
    finally:
        os.dup2(stderr_fd, 2)
        os.close(stderr_fd)


def _open_quietly(url: str) -> None:
    """Open `url`, swallowing any launcher failure."""
    try:
        webbrowser.open(url)
    except Exception:
        pass


__all__ = ["open_browser"]

"""
utils/encoding_guard.py
=======================
Force UTF-8 for stdout, stderr, and logging on Windows.

Usage — add to the TOP of every pipeline script, before any prints or logging:

    from utils.encoding_guard import ensure_utf8
    ensure_utf8()

Safe to call on Linux/Mac (no-op when encoding is already UTF-8).
"""
from __future__ import annotations

import io
import logging
import sys


def ensure_utf8() -> None:
    """Reconfigure stdout/stderr to UTF-8 if they aren't already."""
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None:
            continue
        if not hasattr(stream, "buffer"):
            continue                                     # already wrapped
        current_enc = getattr(stream, "encoding", "") or ""
        if current_enc.lower().replace("-", "") != "utf8":
            setattr(
                sys,
                name,
                io.TextIOWrapper(
                    stream.buffer,
                    encoding="utf-8",
                    errors="replace",
                    line_buffering=True,
                ),
            )

    # Re-attach the root logging StreamHandler to the (now UTF-8) stderr
    root = logging.getLogger()
    for handler in root.handlers:
        if isinstance(handler, logging.StreamHandler):
            if handler.stream in (sys.__stderr__, sys.__stdout__):
                handler.setStream(sys.stderr)

"""
utils/file_guard.py
-------------------
Guard against the two file-tool bugs in Cowork / Claude Code:
  1. Silent tail truncation when file content exceeds the tool buffer (~12 KB)
  2. Null-byte padding appended after Edit operations

Usage
-----
    from utils.file_guard import safe_write, check_file

    # Write a large file safely (bypasses the buffer limit)
    safe_write(path, content)

    # After any Edit/Write tool call, call this to strip nulls + verify AST
    check_file(path)
"""

from __future__ import annotations
import ast
import sys
from pathlib import Path


def strip_nulls(path: str | Path) -> int:
    """
    Remove trailing null bytes from a file in-place.
    Returns the number of bytes removed.
    """
    p = Path(path)
    raw = p.read_bytes()
    clean = raw.rstrip(b'\x00')
    removed = len(raw) - len(clean)
    if removed:
        p.write_bytes(clean)
    return removed


def check_file(path: str | Path, fix: bool = True) -> bool:
    """
    Verify a Python source file is intact:
      - Strip null bytes (if fix=True)
      - Parse with ast.parse
      - Warn if file ends suspiciously (truncated mid-line)

    Returns True if file is clean, False if problems found.
    """
    p = Path(path)
    if fix:
        removed = strip_nulls(p)
        if removed:
            print(f"[file_guard] Stripped {removed} null bytes from {p.name}")

    src = p.read_text(encoding="utf-8", errors="replace")
    lines = src.splitlines()

    # Check for truncation: last non-empty line should not end mid-statement
    last = next((l for l in reversed(lines) if l.strip()), "")
    suspicious = last.strip() and not last.strip().endswith((":", ")", "]", "}", '"""', "'''", '"""'))

    try:
        ast.parse(src)
        if suspicious:
            print(f"[file_guard] WARNING: {p.name} passes AST but last line looks truncated:")
            print(f"  -> {last!r}")
        else:
            print(f"[file_guard] OK: {p.name} ({len(lines)} lines, {len(p.read_bytes())} bytes)")
        return True
    except SyntaxError as e:
        print(f"[file_guard] SYNTAX ERROR in {p.name} line {e.lineno}: {e.msg}")
        print(f"  -> {lines[e.lineno - 1] if e.lineno <= len(lines) else '(past EOF)'!r}")
        return False


def safe_write(path: str | Path, content: str) -> None:
    """
    Write content to path via Python directly — bypasses the tool buffer limit.
    Always use this for files larger than ~10 KB.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    check_file(p, fix=False)


if __name__ == "__main__":
    # CLI: python -m utils.file_guard path/to/file.py [path2 ...]
    targets = sys.argv[1:] or []
    if not targets:
        print("Usage: python utils/file_guard.py <file.py> [file2.py ...]")
        sys.exit(0)
    ok = all(check_file(t) for t in targets)
    sys.exit(0 if ok else 1)

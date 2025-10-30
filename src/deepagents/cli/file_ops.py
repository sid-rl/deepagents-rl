"""Helpers for tracking file operations and computing diffs for CLI display."""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


FileOpStatus = Literal["pending", "success", "error"]


def _safe_read(path: Path) -> str | None:
    """Read file content, returning None on failure."""
    try:
        return path.read_text()
    except (OSError, UnicodeDecodeError):
        return None


def _count_lines(text: str) -> int:
    """Count lines in text, treating empty strings as zero lines."""
    if not text:
        return 0
    return len(text.splitlines())


def _compute_diff(before: str, after: str, display_path: str, *, max_lines: int = 400) -> str | None:
    """Compute a unified diff between before and after content."""
    before_lines = before.splitlines()
    after_lines = after.splitlines()
    diff_lines = list(
        difflib.unified_diff(
            before_lines,
            after_lines,
            fromfile=f"{display_path} (before)",
            tofile=f"{display_path} (after)",
            lineterm="",
        )
    )
    if not diff_lines:
        return None
    if len(diff_lines) > max_lines:
        truncated = diff_lines[: max_lines - 1]
        truncated.append("... (diff truncated)")
        return "\n".join(truncated)
    return "\n".join(diff_lines)


@dataclass
class FileOpMetrics:
    """Line and byte level metrics for a file operation."""

    lines_read: int = 0
    start_line: int | None = None
    end_line: int | None = None
    lines_written: int = 0
    lines_added: int = 0
    lines_removed: int = 0
    bytes_written: int = 0


@dataclass
class FileOperationRecord:
    """Track a single filesystem tool call."""

    tool_name: str
    display_path: str
    physical_path: Path | None
    tool_call_id: str | None
    args: dict[str, Any] = field(default_factory=dict)
    status: FileOpStatus = "pending"
    error: str | None = None
    metrics: FileOpMetrics = field(default_factory=FileOpMetrics)
    diff: str | None = None
    before_content: str | None = None
    after_content: str | None = None
    read_output: str | None = None


class FileOpTracker:
    """Collect file operation metrics during a CLI interaction."""

    def __init__(self, *, assistant_id: str | None) -> None:
        self.assistant_id = assistant_id
        self.agent_dir: Path | None = (
            Path.home() / ".deepagents" / assistant_id if assistant_id else None
        )
        self.active: dict[str | None, FileOperationRecord] = {}
        self.completed: list[FileOperationRecord] = []

    def _physical_path(self, path_str: str | None) -> Path | None:
        if path_str is None:
            return None
        if path_str.startswith("/memories/") and self.agent_dir:
            suffix = path_str.removeprefix("/memories/")
            return self.agent_dir / suffix.lstrip("/")
        try:
            path = Path(path_str)
        except TypeError:
            return None
        if path.is_absolute():
            return path
        return (Path.cwd() / path).resolve()

    def _display_path(self, path_str: str | None) -> str:
        if not path_str:
            return "(unknown)"
        try:
            path = Path(path_str)
            if path.is_absolute():
                return path.name or str(path)
            return str(path)
        except (OSError, ValueError):
            return str(path_str)

    def start_operation(self, tool_name: str, args: dict[str, Any], tool_call_id: str | None) -> None:
        if tool_name not in {"read_file", "write_file", "edit_file"}:
            return
        path_str = str(args.get("file_path") or args.get("path") or "")
        display_path = self._display_path(path_str)
        record = FileOperationRecord(
            tool_name=tool_name,
            display_path=display_path,
            physical_path=self._physical_path(path_str),
            tool_call_id=tool_call_id,
            args=args,
        )
        if tool_name in {"write_file", "edit_file"} and record.physical_path:
            record.before_content = _safe_read(record.physical_path) or ""
        self.active[tool_call_id] = record

    def complete_with_message(self, tool_message: Any) -> FileOperationRecord | None:
        tool_call_id = getattr(tool_message, "tool_call_id", None)
        record = self.active.get(tool_call_id)
        if record is None:
            return None

        content = tool_message.content
        if isinstance(content, list):
            # Some tool messages may return list segments; join them for analysis.
            joined = []
            for item in content:
                if isinstance(item, str):
                    joined.append(item)
                else:
                    joined.append(str(item))
            content_text = "\n".join(joined)
        else:
            content_text = str(content) if content is not None else ""

        if getattr(tool_message, "status", "success") != "success" or content_text.lower().startswith("error"):
            record.status = "error"
            record.error = content_text
            self._finalize(record)
            return record

        record.status = "success"

        if record.tool_name == "read_file":
            record.read_output = content_text
            lines = _count_lines(content_text)
            record.metrics.lines_read = lines
            offset = record.args.get("offset")
            limit = record.args.get("limit")
            if isinstance(offset, int):
                record.metrics.start_line = offset + 1
                if lines:
                    record.metrics.end_line = offset + lines
            elif lines:
                record.metrics.start_line = 1
                record.metrics.end_line = lines
            if isinstance(limit, int) and lines > limit:
                record.metrics.end_line = (record.metrics.start_line or 1) + limit - 1
        else:
            self._populate_after_content(record)
            if record.after_content is None:
                record.status = "error"
                record.error = "Could not read updated file content."
                self._finalize(record)
                return record
            record.metrics.lines_written = _count_lines(record.after_content)
            before_lines = _count_lines(record.before_content or "")
            if record.tool_name == "write_file" and (record.before_content or "") == "":
                record.metrics.lines_added = record.metrics.lines_written
            else:
                diff = _compute_diff(
                    record.before_content or "",
                    record.after_content,
                    record.display_path,
                )
                record.diff = diff
                if diff:
                    additions = sum(1 for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++"))
                    deletions = sum(1 for line in diff.splitlines() if line.startswith("-") and not line.startswith("---"))
                    record.metrics.lines_added = additions
                    record.metrics.lines_removed = deletions
            record.metrics.bytes_written = len(record.after_content.encode("utf-8"))
            if record.diff is None and (record.before_content or "") != record.after_content:
                record.diff = _compute_diff(
                    record.before_content or "",
                    record.after_content,
                    record.display_path,
                )
            if record.diff is None and before_lines != record.metrics.lines_written:
                record.metrics.lines_added = max(record.metrics.lines_written - before_lines, 0)

        self._finalize(record)
        return record

    def _populate_after_content(self, record: FileOperationRecord) -> None:
        if record.physical_path is None:
            record.after_content = None
            return
        record.after_content = _safe_read(record.physical_path)

    def _finalize(self, record: FileOperationRecord) -> None:
        self.completed.append(record)
        self.active.pop(record.tool_call_id, None)

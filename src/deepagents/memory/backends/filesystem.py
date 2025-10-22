"""FilesystemBackend: Read and write files directly from the filesystem."""

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


class FilesystemBackend:
    """Backend that reads and writes files directly from the filesystem.

    Files are accessed using their actual filesystem paths. Relative paths are
    resolved relative to the current working directory. Content is read/written
    as plain text, and metadata (timestamps) are derived from filesystem stats.
    """

    def __init__(self) -> None:
        """Initialize filesystem backend."""
        self.cwd = Path.cwd()

    @property
    def uses_state(self) -> bool:
        """False for FilesystemBackend - stores directly to disk."""
        return False

    def _resolve_path(self, key: str) -> Path:
        """Resolve a file path relative to cwd if not absolute.

        Args:
            key: File path (absolute or relative)

        Returns:
            Resolved absolute Path object
        """
        path = Path(key)
        if path.is_absolute():
            return path
        return self.cwd / path

    def get(self, key: str) -> Optional[dict[str, Any]]:
        """Read file from filesystem.

        Args:
            key: File path (absolute or relative to cwd, e.g., "notes.txt" or "/home/user/notes.txt")

        Returns:
            FileData dict with content (list of lines) and timestamps, or None if not found.
        """
        file_path = self._resolve_path(key)

        if not file_path.exists() or not file_path.is_file():
            return None

        try:
            # Read content as lines
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().splitlines()

            # Get timestamps from filesystem
            stat = file_path.stat()
            created_at = datetime.fromtimestamp(stat.st_ctime).isoformat() + "Z"
            modified_at = datetime.fromtimestamp(stat.st_mtime).isoformat() + "Z"

            return {
                "content": content,
                "created_at": created_at,
                "modified_at": modified_at,
            }
        except (OSError, UnicodeDecodeError):
            return None

    def put(self, key: str, value: dict[str, Any]) -> None:
        """Write file to filesystem.

        Args:
            key: File path (absolute or relative to cwd, e.g., "notes.txt" or "/home/user/notes.txt")
            value: FileData dict with "content" (list of lines)
        """
        file_path = self._resolve_path(key)

        # Create parent directories if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write content (join lines with newlines)
        content = value.get("content", [])
        text = "\n".join(content)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)

    def ls(self, prefix: Optional[str] = None) -> list[str]:
        """List files from filesystem.

        Args:
            prefix: Optional directory path to list files from (absolute or relative to cwd).
                   Defaults to current working directory if not provided.

        Returns:
            List of absolute file paths.
        """
        if prefix is None:
            # Default to current working directory
            dir_path = self.cwd
        else:
            dir_path = self._resolve_path(prefix)

        if not dir_path.exists() or not dir_path.is_dir():
            return []

        results: list[str] = []

        # Walk the directory tree
        try:
            for path in dir_path.rglob("*"):
                if path.is_file():
                    results.append(str(path))
        except (OSError, PermissionError):
            pass

        return sorted(results)

    def delete(self, key: str) -> None:
        """Delete file from filesystem.

        Args:
            key: File path to delete (absolute or relative to cwd)
        """
        file_path = self._resolve_path(key)

        if file_path.exists() and file_path.is_file():
            file_path.unlink()

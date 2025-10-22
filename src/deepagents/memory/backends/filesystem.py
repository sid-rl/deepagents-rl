"""FilesystemBackend: Read and write files directly from the filesystem."""

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from langgraph.types import Command

from .utils import check_empty_content, format_content_with_line_numbers, perform_string_replacement


class FilesystemBackend:
    """Backend that reads and writes files directly from the filesystem.

    Files are accessed using their actual filesystem paths. Relative paths are
    resolved relative to the current working directory. Content is read/written
    as plain text, and metadata (timestamps) are derived from filesystem stats.
    """

    def __init__(
        self, 
        root_dir: Optional[str | Path] = None,
        virtual_mode: bool = False
    ) -> None:
        """Initialize filesystem backend.
        
        Args:
            root_dir: Optional root directory for file operations. If provided,
                     all file paths will be resolved relative to this directory.
                     If not provided, uses the current working directory.
        """
        self.cwd = Path(root_dir) if root_dir else Path.cwd()
        self.virtual_mode = virtual_mode

    @property
    def uses_state(self) -> bool:
        """False for FilesystemBackend - stores directly to disk."""
        return False

    def get_system_prompt_addition(self) -> Optional[str]:
        """Provide CWD information for the system prompt.

        Returns:
            System prompt text explaining the current working directory.
        """
        return f"""
### Current Working Directory

The filesystem backend is currently operating in: `{self.cwd}`

When using filesystem tools (ls, read_file, write_file, edit_file):
- Relative paths (e.g., "notes.txt", "data/config.json") will be resolved relative to the current working directory
- Absolute paths (e.g., "/home/user/file.txt") will be used as-is
- To list files in the current directory, use `ls()` with no arguments or `ls(".")`
"""

    def _resolve_path(self, key: str) -> Path:
        """Resolve a file path relative to cwd if not absolute.

        Args:
            key: File path (absolute or relative)

        Returns:
            Resolved absolute Path object
        """
        if self.virtual_mode:
            return self.cwd / key.lstrip('/')
        path = Path(key)
        if path.is_absolute():
            return path
        return self.cwd / path

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

        # Convert cwd to string for comparison
        cwd_str = str(self.cwd)
        if not cwd_str.endswith("/"):
            cwd_str += "/"

        # Walk the directory tree
        try:
            for path in dir_path.rglob("*"):
                if path.is_file():
                    abs_path = str(path)
                    if not self.virtual_mode:
                        results.append(abs_path)
                        continue
                    # Strip the cwd prefix if present
                    if abs_path.startswith(cwd_str):
                        relative_path = abs_path[len(cwd_str):]
                    elif abs_path.startswith(str(self.cwd)):
                        # Handle case where cwd doesn't end with /
                        relative_path = abs_path[len(str(self.cwd)):].lstrip("/")
                    else:
                        # Path is outside cwd, return as-is or skip
                        relative_path = abs_path

                    results.append("/" + relative_path)
        except (OSError, PermissionError):
            pass

        return sorted(results)
    
    def read(
        self, 
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """Read file content with line numbers.
        
        Args:
            file_path: Absolute or relative file path
            offset: Line offset to start reading from (0-indexed)
            limit: Maximum number of lines to read
        
        Returns:
            Formatted file content with line numbers, or error message.
        """
        resolved_path = self._resolve_path(file_path)
        
        if not resolved_path.exists() or not resolved_path.is_file():
            return f"Error: File '{file_path}' not found"
        
        try:
            with open(resolved_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            empty_msg = check_empty_content(content)
            if empty_msg:
                return empty_msg
            
            lines = content.splitlines()
            start_idx = offset
            end_idx = min(start_idx + limit, len(lines))
            
            if start_idx >= len(lines):
                return f"Error: Line offset {offset} exceeds file length ({len(lines)} lines)"
            
            selected_lines = lines[start_idx:end_idx]
            return format_content_with_line_numbers(selected_lines, start_line=start_idx + 1)
        except (OSError, UnicodeDecodeError) as e:
            return f"Error reading file '{file_path}': {e}"
    
    def write(
        self, 
        file_path: str,
        content: str,
    ) -> Command | str:
        """Create a new file with content.
        
        Args:
            file_path: Absolute or relative file path
            content: File content as a string
        
        Returns:
            Success message or error if file already exists.
        """
        resolved_path = self._resolve_path(file_path)
        
        if resolved_path.exists():
            return f"Cannot write to {file_path} because it already exists. Read and then make an edit, or write to a new path."
        
        try:
            # Create parent directories if needed
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(resolved_path, "w", encoding="utf-8") as f:
                f.write(content)
            
            return f"Updated file {file_path}"
        except (OSError, UnicodeEncodeError) as e:
            return f"Error writing file '{file_path}': {e}"
    
    def edit(
        self, 
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> Command | str:
        """Edit a file by replacing string occurrences.
        
        Args:
            file_path: Absolute or relative file path
            old_string: String to find and replace
            new_string: Replacement string
            replace_all: If True, replace all occurrences
        
        Returns:
            Success message or error message on failure.
        """
        resolved_path = self._resolve_path(file_path)
        
        if not resolved_path.exists() or not resolved_path.is_file():
            return f"Error: File '{file_path}' not found"
        
        try:
            with open(resolved_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            result = perform_string_replacement(content, old_string, new_string, replace_all)
            
            if isinstance(result, str):
                return result
            
            new_content, occurrences = result
            
            with open(resolved_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            
            return f"Successfully replaced {occurrences} instance(s) of the string in '{file_path}'"
        except (OSError, UnicodeDecodeError, UnicodeEncodeError) as e:
            return f"Error editing file '{file_path}': {e}"

    def delete(self, file_path: str) -> Command | None:
        """Delete file from filesystem.

        Args:
            file_path: File path to delete (absolute or relative to cwd)
        
        Returns:
            None (direct filesystem modification)
        """
        resolved_path = self._resolve_path(file_path)

        if resolved_path.exists() and resolved_path.is_file():
            resolved_path.unlink()
        
        return None
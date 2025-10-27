"""Type definitions for backend interface.

This module defines the unified Backend interface that all backend implementations
must follow. Backends can store files in different locations (state, filesystem,
database, etc.) and provide a uniform interface for file operations.
"""

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from langchain.tools import ToolRuntime


@dataclass
class WriteResult:
    """Result of a file write operation (creating a new file).

    This class provides a consistent return type for write operations across all
    backends, regardless of whether they manage their own storage or rely on
    framework-managed storage.

    Attributes:
        error:
            None on success.
            String error message on failure.

        path:
            Absolute path of the file that was written.
            None on failure.

        content:
            Full file content that was written.
            May be None if the write failed, or if the backend
            chooses not to return file contents.

        files_update:
            For checkpoint storage backends (e.g., StateBackend):
                A {file_path: file_data} mapping representing the new canonical
                state for those files. The tool layer is responsible for merging
                this into LangGraph state (persisted via checkpoints).

            For external storage backends (e.g., FilesystemBackend, S3Backend, StoreBackend):
                None, because the backend has already committed the change to
                its external storage system (disk, S3, database, BaseStore, etc.).

    Examples:
        Checkpoint storage backend (StateBackend):
        >>> WriteResult(path="/notes.txt", content="Hello world", files_update={"/notes.txt": {"content": [...], "created_at": ...}})

        External storage backend (FilesystemBackend):
        >>> WriteResult(
        ...     path="/notes.txt",
        ...     content="Hello world",
        ...     files_update=None,  # Already written to disk
        ... )

        Error case:
        >>> WriteResult(error="File already exists")
    """

    error: str | None = None
    path: str | None = None
    content: str | None = None
    files_update: dict[str, Any] | None = None


@dataclass
class EditResult:
    """Result of a file edit operation (modifying an existing file).

    This class provides a consistent return type for edit operations across all
    backends, regardless of whether they manage their own storage or rely on
    framework-managed storage.

    Attributes:
        error:
            None on success.
            String error message on failure.

        path:
            Absolute path of the file that was edited.
            None on failure.

        content:
            Full file content after the edit.
            May be None if the edit failed, or if the backend
            chooses not to return file contents.

        files_update:
            For checkpoint storage backends (e.g., StateBackend):
                A {file_path: file_data} mapping representing the new canonical
                state for those files. The tool layer is responsible for merging
                this into LangGraph state (persisted via checkpoints).

            For external storage backends (e.g., FilesystemBackend, S3Backend, StoreBackend):
                None, because the backend has already committed the change to
                its external storage system (disk, S3, database, BaseStore, etc.).

        occurrences:
            Number of string occurrences that were replaced.
            None on failure or if not applicable.

    Examples:
        Checkpoint storage backend (StateBackend):
        >>> EditResult(path="/notes.txt", content="Hello world", files_update={"/notes.txt": {"content": [...], "modified_at": ...}}, occurrences=1)

        External storage backend (FilesystemBackend):
        >>> EditResult(
        ...     path="/notes.txt",
        ...     content="Hello world",
        ...     files_update=None,  # Already written to disk
        ...     occurrences=3,
        ... )

        Error case:
        >>> EditResult(error="File not found")
    """

    error: str | None = None
    path: str | None = None
    content: str | None = None
    files_update: dict[str, Any] | None = None
    occurrences: int | None = None


class Backend(ABC):
    """Abstract base class for pluggable memory backends.

    Backends can store files in different locations (state, filesystem,
    database, etc.) and provide a uniform interface for file operations.

    Storage Models:
    --------------
    There are two categories of backends based on how they handle storage:

    1. Checkpoint Storage (e.g., StateBackend):
       - Files are stored as data structures in LangGraph state
       - Persisted via LangGraph's checkpoint system (Postgres, Redis, in-memory, etc.)
       - write() and edit() return WriteResult/EditResult with files_update populated
       - The tool layer converts files_update into LangGraph state updates via Command

       Example: StateBackend stores files in LangGraph state, which is persisted
       through LangGraph's checkpointing. The backend returns files_update, and
       the tool layer wraps it in a Command object for state mutation.

    2. External Storage (e.g., FilesystemBackend, S3Backend, StoreBackend, DatabaseBackend):
       - Files are stored in external storage systems (filesystem, S3, BaseStore, database, etc.)
       - Backend persists changes directly to external storage
       - write() and edit() return WriteResult/EditResult with files_update=None
       - Tool layer just reports success/error

       Example: FilesystemBackend writes directly to disk, then returns a
       WriteResult indicating success with files_update=None.

    File Data Format:
    ----------------
    Checkpoint storage backends represent files as dicts with:
    {
        "content": list[str],      # Lines of text content
        "created_at": str,         # ISO format timestamp
        "modified_at": str,        # ISO format timestamp
    }

    External storage backends handle their own file format internally.
    """

    @abstractmethod
    def ls(self, path: str) -> list[str]:
        """List all file paths in a directory.

        Args:
            path: Absolute path to directory (e.g., "/", "/subdir/", "/memories/")

        Returns:
            List of absolute file paths in the specified directory.
            Returns empty list if directory doesn't exist or is empty.
        """
        ...

    @abstractmethod
    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """Read file content with line numbers.

        Args:
            file_path: Absolute file path (e.g., "/notes.txt", "/memories/agent.md")
            offset: Line offset to start reading from (0-indexed)
            limit: Maximum number of lines to read

        Returns:
            Formatted file content with line numbers (cat -n style), or error message.
            Returns "Error: File '{file_path}' not found" if file doesn't exist.
            Returns "System reminder: File exists but has empty contents" for empty files.
        """
        ...

    @abstractmethod
    def write(self, file_path: str, content: str) -> WriteResult:
        """Create a new file with content.

        Args:
            file_path: Absolute file path (e.g., "/notes.txt", "/memories/agent.md")
            content: File content as a string

        Returns:
            WriteResult with:
            - On success:
                - error=None
                - path=file_path
                - content=content (or None)
                - files_update={...} for checkpoint storage backends
                - files_update=None for external storage backends
            - On failure:
                - error="error message"
                - path=None
                - content=None
                - files_update=None

        Error cases:
            - File already exists (should use edit instead)
            - Permission denied
            - Path traversal attempts
            - I/O errors
        """
        ...

    @abstractmethod
    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Edit a file by replacing string occurrences.

        Args:
            file_path: Absolute file path (e.g., "/notes.txt", "/memories/agent.md")
            old_string: String to find and replace
            new_string: Replacement string
            replace_all: If True, replace all occurrences; if False, require unique match

        Returns:
            EditResult with:
            - On success:
                - error=None
                - path=file_path
                - content=updated content (or None)
                - files_update={...} for checkpoint storage backends
                - files_update=None for external storage backends
                - occurrences=number of replacements made
            - On failure:
                - error="error message"
                - path=None
                - content=None
                - files_update=None
                - occurrences=None

        Error cases:
            - "Error: File '{file_path}' not found" if file doesn't exist
            - "Error: String not found in file: '{old_string}'" if string not found
            - "Error: String '{old_string}' appears {n} times. Use replace_all=True..."
              if multiple matches found and replace_all=False
        """
        ...

    @abstractmethod
    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        output_mode: str = "files_with_matches",
    ) -> str:
        """Search for a pattern in files.

        TODO: This implementation is significantly less capable than Claude Code's Grep tool.
        Missing features to add in the future:
        - Context lines: -A (after), -B (before), -C (context) parameters
        - Line numbers: -n parameter to show line numbers in output
        - Case sensitivity: -i parameter for case-insensitive search
        - Output limiting: head_limit parameter for large result sets
        - File type filter: type parameter (e.g., "py", "js")
        - Multiline support: multiline parameter for cross-line pattern matching
        - Pattern semantics: Clarify if pattern is regex or literal string

        Args:
            pattern: String pattern to search for (currently literal string)
            path: Path to search in (default "/")
            glob: Optional glob pattern to filter files (e.g., "*.py")
            output_mode: Output format - "files_with_matches", "content", or "count"
                - files_with_matches: List file paths that contain matches
                - content: Show matching lines with file paths and line numbers
                - count: Show count of matches per file

        Returns:
            Formatted search results based on output_mode, or message if no matches found.
        """
        ...

    @abstractmethod
    def glob(self, pattern: str, path: str = "/") -> list[str]:
        """Find files matching a glob pattern.

        Args:
            pattern: Glob pattern (e.g., "**/*.py", "*.txt", "/subdir/**/*.md")
            path: Base path to search from (default: "/")

        Returns:
            List of absolute file paths matching the pattern.
            Returns empty list if no matches found.
        """
        ...


# Type alias for backend factory functions
# A backend provider is any callable that takes a ToolRuntime and returns a Backend instance
BackendProvider = Callable[[ToolRuntime], Backend]
AsyncBackendProvider = Callable[[ToolRuntime], Awaitable[Backend]]

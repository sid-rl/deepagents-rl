"""Protocol definition for pluggable memory backends.

This module defines the BackendProtocol that all backend implementations
must follow. Backends can store files in different locations (state, filesystem,
database, etc.) and provide a uniform interface for file operations.
"""

from typing import TYPE_CHECKING, Optional, Protocol, runtime_checkable, Callable, TypeAlias, Any
from langchain.tools import ToolRuntime
from deepagents.backends.utils import FileInfo, GrepMatch

from dataclasses import dataclass


@dataclass
class WriteResult:
    """Result from backend write operations.
    Attributes:
        error: Error message on failure, None on success.
        path: Absolute path of written file, None on failure.
        files_update: State update dict for checkpoint backends, None for external storage.
            Checkpoint backends populate this with {file_path: file_data} for LangGraph state.
            External backends set None (already persisted to disk/S3/database/etc).
    Examples:
        >>> # Checkpoint storage
        >>> WriteResult(path="/f.txt", files_update={"/f.txt": {...}})
        >>> # External storage
        >>> WriteResult(path="/f.txt", files_update=None)
        >>> # Error
        >>> WriteResult(error="File exists")
    """

    error: str | None = None
    path: str | None = None
    files_update: dict[str, Any] | None = None


@dataclass
class EditResult:
    """Result from backend edit operations.
    Attributes:
        error: Error message on failure, None on success.
        path: Absolute path of edited file, None on failure.
        files_update: State update dict for checkpoint backends, None for external storage.
            Checkpoint backends populate this with {file_path: file_data} for LangGraph state.
            External backends set None (already persisted to disk/S3/database/etc).
        occurrences: Number of replacements made, None on failure.
    Examples:
        >>> # Checkpoint storage
        >>> EditResult(path="/f.txt", files_update={"/f.txt": {...}}, occurrences=1)
        >>> # External storage
        >>> EditResult(path="/f.txt", files_update=None, occurrences=2)
        >>> # Error
        >>> EditResult(error="File not found")
    """

    error: str | None = None
    path: str | None = None
    files_update: dict[str, Any] | None = None
    occurrences: int | None = None

@runtime_checkable
class _BackendProtocol(Protocol):
    """Protocol for pluggable memory backends.

    Backends can store files in different locations (state, filesystem, database, etc.)
    and provide a uniform interface for file operations.

    All file data is represented as dicts with the following structure:
    {
        "content": list[str],      # Lines of text content
        "created_at": str,         # ISO format timestamp
        "modified_at": str,        # ISO format timestamp
    }
    """
    
    def ls_info(self, path: str) -> list["FileInfo"]:
        """Structured listing with file metadata.

        Returns a list of FileInfo-like dicts: at minimum includes "path";
        may include fields such as "is_dir", "size", and "modified_at" depending on backend.
        """
        ...
    
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

    
    def grep_raw(
        self,
        pattern: str,
        path: Optional[str] = None,
        glob: Optional[str] = None,
    ) -> list["GrepMatch"] | str:
        """Structured search results.

        Returns a list of GrepMatch-like dicts {path, line, text} on success,
        or a string when the input is invalid (e.g., invalid regex pattern) or a
        backend needs to surface a user-actionable error without throwing.

        Rationale for union return type:
        - Tool contexts generally prefer non-throwing backends; returning a
          string preserves error information end-to-end instead of raising and
          losing context at tool boundaries.
        - Backends may rely on external engines (e.g., ripgrep) and need to
          propagate their validation errors verbatim.
        - Normal "no matches" is represented as an empty list; only input/
          validation errors use the string form.

        Prefer this method for composition and apply formatting at the tool layer.
        """
        ...
    
    def glob_info(self, pattern: str, path: str = "/") -> list["FileInfo"]:
        """Structured glob matching.

        Returns a list of dicts with at least {"path"}; may include extra fields.
        """
        ...


class BackendProtocol(_BackendProtocol):
    def write(
            self,
            file_path: str,
            content: str,
    ) -> WriteResult:
        """Create a new file with content.
        Returns WriteResult with either files_update for state backends or None for external storage.
        Error is populated on failure.
        """
        ...

    def edit(
            self,
            file_path: str,
            old_string: str,
            new_string: str,
            replace_all: bool = False,
    ) -> EditResult:
        """Edit a file by replacing string occurrences.
        Returns EditResult with occurrences and optional files_update for state backends.
        Error is populated on failure.
        """
        ...


BackendFactory: TypeAlias = Callable[[ToolRuntime], BackendProtocol]
StateBackendFactory: TypeAlias = Callable[[ToolRuntime], BackendProtocol]

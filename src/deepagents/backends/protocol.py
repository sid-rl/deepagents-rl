"""Protocol definition for pluggable memory backends.

This module defines the BackendProtocol that all backend implementations
must follow. Backends can store files in different locations (state, filesystem,
database, etc.) and provide a uniform interface for file operations.
"""

from typing import TYPE_CHECKING, Optional, Protocol, runtime_checkable, Callable, TypeAlias
from langgraph.types import Command
from langchain.tools import ToolRuntime

if TYPE_CHECKING:
    # TypedDicts for structured returns
    from deepagents.backends.utils import FileInfo, GrepMatch

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
    ) -> str:
        """Create a new file with content.

        Args:
            file_path: Absolute file path (e.g., "/notes.txt", "/memories/agent.md")
            content: File content as a string

        Returns:
            - Command object for StateBackend (uses_state=True) to update LangGraph state
            - Success message string for other backends, or error if file already exists

        Error cases:
            - Returns error message if file already exists (should use edit instead)
        """
        ...

    def edit(
            self,
            file_path: str,
            old_string: str,
            new_string: str,
            replace_all: bool = False,
    ) -> str:
        """Edit a file by replacing string occurrences.

        Args:
            file_path: Absolute file path (e.g., "/notes.txt", "/memories/agent.md")
            old_string: String to find and replace
            new_string: Replacement string
            replace_all: If True, replace all occurrences; if False, require unique match

        Returns:
            - Command object for StateBackend (uses_state=True) to update LangGraph state
            - Success message string for other backends, or error message on failure

        Error cases:
            - "Error: File '{file_path}' not found" if file doesn't exist
            - "Error: String not found in file: '{old_string}'" if string not found
            - "Error: String '{old_string}' appears {n} times. Use replace_all=True..."
              if multiple matches found and replace_all=False
        """
        ...


@runtime_checkable
class BackendProvider(Protocol):

    def get_backend(self, runtime: ToolRuntime) -> BackendProtocol:
        """Get the backend."""
        ...

# Callable factory alternative to provider classes
BackendFactory: TypeAlias = Callable[[ToolRuntime], BackendProtocol]


class StateBackendProtocol(_BackendProtocol):

    def write(
            self,
            file_path: str,
            content: str,
    ) -> Command | str:
        """Create a new file with content.

        Args:
            file_path: Absolute file path (e.g., "/notes.txt", "/memories/agent.md")
            content: File content as a string

        Returns:
            - Command object for StateBackend (uses_state=True) to update LangGraph state
            - Success message string for other backends, or error if file already exists

        Error cases:
            - Returns error message if file already exists (should use edit instead)
        """
        ...

    def edit(
            self,
            file_path: str,
            old_string: str,
            new_string: str,
            replace_all: bool = False,
    ) -> Command | str:
        """Edit a file by replacing string occurrences.

        Args:
            file_path: Absolute file path (e.g., "/notes.txt", "/memories/agent.md")
            old_string: String to find and replace
            new_string: Replacement string
            replace_all: If True, replace all occurrences; if False, require unique match

        Returns:
            - Command object for StateBackend (uses_state=True) to update LangGraph state
            - Success message string for other backends, or error message on failure

        Error cases:
            - "Error: File '{file_path}' not found" if file doesn't exist
            - "Error: String not found in file: '{old_string}'" if string not found
            - "Error: String '{old_string}' appears {n} times. Use replace_all=True..."
              if multiple matches found and replace_all=False
        """
        ...

@runtime_checkable
class StateBackendProvider(Protocol):

    def get_backend(self, runtime: ToolRuntime) -> StateBackendProtocol:
        """Get the backend."""
        ...

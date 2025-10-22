"""Protocol definition for pluggable memory backends.

This module defines the MemoryBackend protocol that all backend implementations
must follow. Backends can store files in different locations (state, filesystem,
database, etc.) and provide a uniform interface for file operations.
"""

from typing import Optional, Protocol, runtime_checkable
from langgraph.types import Command


@runtime_checkable
class MemoryBackend(Protocol):
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

    @property
    def uses_state(self) -> bool:
        """Flag indicating if this backend uses agent state.

        When True, write() and edit() operations should return Command objects 
        instead of None. This is required for backends that store data in LangGraph 
        state, which must be updated via Command objects for proper checkpointing.

        Default: False (most backends modify external storage directly)
        """
        ...

    def get_system_prompt_addition(self) -> Optional[str]:
        """Get additional system prompt text for this backend.

        Backends can provide context-specific information to be injected into
        the system prompt (e.g., current working directory for FilesystemBackend).

        Returns:
            Optional string to append to system prompt, or None if no addition needed.
        """
        ...
    
    def ls(self, prefix: Optional[str] = None) -> list[str]:
        """List all file paths, optionally filtered by prefix.
        
        Args:
            prefix: Optional path prefix to filter results (e.g., "/subdir/", "/memories/")
                   If None, returns all files.
        
        Returns:
            List of absolute file paths matching the prefix.
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
    
    def delete(self, file_path: str) -> Command | None:
        """Delete a file by path.
        
        Args:
            file_path: Absolute file path to delete
        
        Returns:
            - None for backends that modify storage directly (uses_state=False)
            - Command object for StateBackend (uses_state=True) to update LangGraph state
        """
        ...
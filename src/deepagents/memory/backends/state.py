"""StateBackend: Store files in LangGraph agent state (ephemeral)."""

from typing import Any, Optional

from langgraph.runtime import get_runtime
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from .utils import (
    create_file_data,
    update_file_data,
    file_data_to_string,
    format_read_response,
    perform_string_replacement,
)


class StateBackend:
    """Backend that stores files in agent state (ephemeral).
    
    Uses LangGraph's state management and checkpointing. Files persist within
    a conversation thread but not across threads. State is automatically
    checkpointed after each agent step.
    
    Special handling: Since LangGraph state must be updated via Command objects
    (not direct mutation), put() operations return Command objects instead of None.
    This is indicated by the uses_state=True flag.
    """
    
    @property
    def uses_state(self) -> bool:
        """Always True for StateBackend - must return Commands for writes."""
        return True

    def get_system_prompt_addition(self) -> Optional[str]:
        """No system prompt addition needed for StateBackend."""
        return None
    
    def ls(self, prefix: Optional[str] = None) -> list[str]:
        """List files from state.
        
        Args:
            prefix: Optional path prefix to filter results.
        
        Returns:
            List of file paths.
        """
        runtime = get_runtime()
        files = runtime.state.get("files", {})
        keys = list(files.keys())
        
        if prefix is not None:
            keys = [k for k in keys if k.startswith(prefix)]
        
        return keys
    
    def read(
        self, 
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """Read file content with line numbers.
        
        Args:
            file_path: Absolute file path
            offset: Line offset to start reading from (0-indexed)
            limit: Maximum number of lines to read
        
        Returns:
            Formatted file content with line numbers, or error message.
        """
        runtime = get_runtime()
        files = runtime.state.get("files", {})
        file_data = files.get(file_path)
        
        if file_data is None:
            return f"Error: File '{file_path}' not found"
        
        return format_read_response(file_data, offset, limit)
    
    def write(
        self, 
        file_path: str,
        content: str,
    ) -> Command | str:
        """Create a new file with content.
        
        Args:
            file_path: Absolute file path
            content: File content as a string
        
        Returns:
            Command object to update state, or error message if file exists.
        """
        runtime = get_runtime()
        files = runtime.state.get("files", {})
        
        if file_path in files:
            return f"Cannot write to {file_path} because it already exists. Read and then make an edit, or write to a new path."
        
        new_file_data = create_file_data(content)
        tool_call_id = runtime.tool_call_id
        
        return Command(
            update={
                "files": {file_path: new_file_data},
                "messages": [
                    ToolMessage(
                        content=f"Updated file {file_path}",
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )
    
    def edit(
        self, 
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> Command | str:
        """Edit a file by replacing string occurrences.
        
        Args:
            file_path: Absolute file path
            old_string: String to find and replace
            new_string: Replacement string
            replace_all: If True, replace all occurrences
        
        Returns:
            Command object to update state, or error message on failure.
        """
        runtime = get_runtime()
        files = runtime.state.get("files", {})
        file_data = files.get(file_path)
        
        if file_data is None:
            return f"Error: File '{file_path}' not found"
        
        content = file_data_to_string(file_data)
        result = perform_string_replacement(content, old_string, new_string, replace_all)
        
        if isinstance(result, str):
            return result
        
        new_content, occurrences = result
        new_file_data = update_file_data(file_data, new_content)
        tool_call_id = runtime.tool_call_id
        
        return Command(
            update={
                "files": {file_path: new_file_data},
                "messages": [
                    ToolMessage(
                        content=f"Successfully replaced {occurrences} instance(s) of the string in '{file_path}'",
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )
    
    def delete(self, file_path: str) -> Command | None:
        """Delete file from state via Command.
        
        Args:
            file_path: File path to delete
        
        Returns:
            Command object to update state (sets file to None for deletion).
        """
        runtime = get_runtime()
        tool_call_id = runtime.tool_call_id
        return Command(
            update={
                "files": {file_path: None},
                "messages": [
                    ToolMessage(
                        content=f"Deleted file {file_path}",
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )
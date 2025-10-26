"""StateBackend: Store files in LangGraph agent state (ephemeral)."""

import re
from typing import Any, Literal, Optional, TYPE_CHECKING

from langchain.tools import ToolRuntime

from langchain_core.messages import ToolMessage
from langgraph.types import Command

from .utils import (
    create_file_data,
    update_file_data,
    file_data_to_string,
    format_read_response,
    perform_string_replacement,
    truncate_if_too_long,
    _glob_search_files,
    grep_matches_from_files,
    format_grep_matches,
)


class StateBackend:
    """Backend that stores files in agent state (ephemeral).
    
    Uses LangGraph's state management and checkpointing. Files persist within
    a conversation thread but not across threads. State is automatically
    checkpointed after each agent step.
    
    Special handling: Since LangGraph state must be updated via Command objects
    (not direct mutation), operations return Command objects instead of None.
    This is indicated by the uses_state=True flag.
    """
    
    def __init__(self, runtime: "ToolRuntime"):
        """Initialize StateBackend with runtime.
        
        Args:"""
        self.runtime = runtime
    
    def ls_info(self, path: str) -> list[dict]:
        """List files from state.
        
        Args:
            path: Absolute path to directory.
        
        Returns:
            List of FileInfo-like dicts.
        """
        files = self.runtime.state.get("files", {})
        infos: list[dict] = []
        for k, fd in files.items():
            if not k.startswith(path):
                continue
            size = len("\n".join(fd.get("content", [])))
            infos.append({
                "path": k,
                "is_dir": False,
                "size": int(size),
                "modified_at": fd.get("modified_at", ""),
            })
        infos.sort(key=lambda x: x.get("path", ""))
        return infos

    def ls(self, path: str) -> list[str]:
        infos = self.ls_info(path)
        return [fi["path"] for fi in infos]
    
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
            limit: Maximum number of lines to readReturns:
            Formatted file content with line numbers, or error message.
        """
        files = self.runtime.state.get("files", {})
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
            content: File content as a stringReturns:
            Command object to update state, or error message if file exists.
        """
        files = self.runtime.state.get("files", {})
        
        if file_path in files:
            return f"Cannot write to {file_path} because it already exists. Read and then make an edit, or write to a new path."
        
        new_file_data = create_file_data(content)
        tool_call_id = self.runtime.tool_call_id
        
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
            replace_all: If True, replace all occurrencesReturns:
            Command object to update state, or error message on failure.
        """
        files = self.runtime.state.get("files", {})
        file_data = files.get(file_path)
        
        if file_data is None:
            return f"Error: File '{file_path}' not found"
        
        content = file_data_to_string(file_data)
        result = perform_string_replacement(content, old_string, new_string, replace_all)
        
        if isinstance(result, str):
            return result
        
        new_content, occurrences = result
        new_file_data = update_file_data(file_data, new_content)
        tool_call_id = self.runtime.tool_call_id
        
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
    
    def grep(
        self,
        pattern: str,
        path: str = "/",
        glob: Optional[str] = None,
        output_mode: Literal["files_with_matches", "content", "count"] = "files_with_matches",
    ) -> str:
        """Search for a pattern in files.
        
        Args:
            pattern: String pattern to search for
            path: Path to search in (default "/")
            glob: Optional glob pattern to filter files (e.g., "*.py")
            output_mode: Output format - "files_with_matches", "content", or "count"Returns:
            Formatted search results based on output_mode.
        """
        files = self.runtime.state.get("files", {})
        matches_or_err = grep_matches_from_files(files, pattern, path, glob)
        if isinstance(matches_or_err, str):
            return matches_or_err
        formatted = format_grep_matches(matches_or_err, output_mode)
        return truncate_if_too_long(formatted)  # type: ignore[arg-type]

    def grep_raw(
        self,
        pattern: str,
        path: str = "/",
        glob: Optional[str] = None,
    ) -> list[dict] | str:
        files = self.runtime.state.get("files", {})
        return grep_matches_from_files(files, pattern, path, glob)
    
    def glob(self, pattern: str, path: str = "/") -> list[str]:
        """Find files matching a glob pattern.
        
        Args:
            pattern: Glob pattern (e.g., "**/*.py", "*.txt", "/subdir/**/*.md")
            path: Base path to search from (default "/")Returns:
            List of absolute file paths matching the pattern.
        """
        files = self.runtime.state.get("files", {})
        
        result = _glob_search_files(files, pattern, path)
        if result == "No files found":
            return []
        return truncate_if_too_long(result.split("\n"))

class StateBackendProvider:

    def get_backend(self, runtime: ToolRuntime):
        return StateBackend(runtime)

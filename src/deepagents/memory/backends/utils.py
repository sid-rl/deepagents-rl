"""Shared utility functions for memory backend implementations."""

from datetime import UTC, datetime
from typing import Any

EMPTY_CONTENT_WARNING = "System reminder: File exists but has empty contents"
MAX_LINE_LENGTH = 2000
LINE_NUMBER_WIDTH = 6


def format_content_with_line_numbers(
    content: str | list[str],
    start_line: int = 1,
) -> str:
    """Format file content with line numbers (cat -n style).
    
    Args:
        content: File content as string or list of lines
        start_line: Starting line number (default: 1)
    
    Returns:
        Formatted content with line numbers
    """
    if isinstance(content, str):
        lines = content.split("\n")
        if lines and lines[-1] == "":
            lines = lines[:-1]
    else:
        lines = content
    
    return "\n".join(
        f"{i + start_line:{LINE_NUMBER_WIDTH}d}\t{line[:MAX_LINE_LENGTH]}" 
        for i, line in enumerate(lines)
    )


def check_empty_content(content: str) -> str | None:
    """Check if content is empty and return warning message.
    
    Args:
        content: Content to check
    
    Returns:
        Warning message if empty, None otherwise
    """
    if not content or content.strip() == "":
        return EMPTY_CONTENT_WARNING
    return None


def file_data_to_string(file_data: dict[str, Any]) -> str:
    """Convert FileData to plain string content.
    
    Args:
        file_data: FileData dict with 'content' key
    
    Returns:
        Content as string with lines joined by newlines
    """
    return "\n".join(file_data["content"])


def create_file_data(content: str, created_at: str | None = None) -> dict[str, Any]:
    """Create a FileData object with timestamps.
    
    Args:
        content: File content as string
        created_at: Optional creation timestamp (ISO format)
    
    Returns:
        FileData dict with content and timestamps
    """
    lines = content.split("\n") if isinstance(content, str) else content
    now = datetime.now(UTC).isoformat()
    
    return {
        "content": lines,
        "created_at": created_at or now,
        "modified_at": now,
    }


def update_file_data(file_data: dict[str, Any], content: str) -> dict[str, Any]:
    """Update FileData with new content, preserving creation timestamp.
    
    Args:
        file_data: Existing FileData dict
        content: New content as string
    
    Returns:
        Updated FileData dict
    """
    lines = content.split("\n") if isinstance(content, str) else content
    now = datetime.now(UTC).isoformat()
    
    return {
        "content": lines,
        "created_at": file_data["created_at"],
        "modified_at": now,
    }


def format_read_response(
    file_data: dict[str, Any],
    offset: int,
    limit: int,
) -> str:
    """Format file data for read response with line numbers.
    
    Args:
        file_data: FileData dict
        offset: Line offset (0-indexed)
        limit: Maximum number of lines
    
    Returns:
        Formatted content or error message
    """
    content = file_data_to_string(file_data)
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


def perform_string_replacement(
    content: str,
    old_string: str,
    new_string: str,
    replace_all: bool,
) -> tuple[str, int] | str:
    """Perform string replacement with occurrence validation.
    
    Args:
        content: Original content
        old_string: String to replace
        new_string: Replacement string
        replace_all: Whether to replace all occurrences
    
    Returns:
        Tuple of (new_content, occurrences) on success, or error message string
    """
    occurrences = content.count(old_string)
    
    if occurrences == 0:
        return f"Error: String not found in file: '{old_string}'"
    
    if occurrences > 1 and not replace_all:
        return f"Error: String '{old_string}' appears {occurrences} times in file. Use replace_all=True to replace all instances, or provide a more specific string with surrounding context."
    
    new_content = content.replace(old_string, new_string)
    return new_content, occurrences

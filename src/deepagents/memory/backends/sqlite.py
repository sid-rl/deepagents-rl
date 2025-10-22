"""SQLiteBackend: Store files in SQLite database."""

import json
import sqlite3
from pathlib import Path
from typing import Any, Optional
from langgraph.types import Command

from .utils import (
    create_file_data,
    format_content_with_line_numbers,
    check_empty_content,
    perform_string_replacement,
)


class SQLiteBackend:
    """Backend that stores files in a SQLite database.
    
    Creates a table with columns:
    - key (TEXT PRIMARY KEY): File path
    - content (TEXT): JSON-serialized list of lines
    - created_at (TEXT): ISO format timestamp
    - modified_at (TEXT): ISO format timestamp
    - agent_id (TEXT): Optional agent identifier for isolation
    
    The database file is created automatically if it doesn't exist.
    """
    
    def __init__(self, db_path: str, agent_id: Optional[str] = None) -> None:
        """Initialize SQLite backend.
        
        Args:
            db_path: Path to SQLite database file.
            agent_id: Optional agent ID for isolation. When provided, only files
                     belonging to this agent are accessible.
        """
        self.db_path = Path(db_path)
        self.agent_id = agent_id
        
        # Create parent directories if needed
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_db()
    
    @property
    def uses_state(self) -> bool:
        """False for SQLiteBackend - stores directly to database."""
        return False

    def get_system_prompt_addition(self) -> Optional[str]:
        """No system prompt addition needed for SQLiteBackend."""
        return None

    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS files (
                    key TEXT NOT NULL,
                    agent_id TEXT NOT NULL DEFAULT '',
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    modified_at TEXT NOT NULL,
                    PRIMARY KEY (key, agent_id)
                )
            """)
            
            # Create index for faster ls operations
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_files_agent_key 
                ON files(agent_id, key)
            """)
            
            conn.commit()
    
    def _get_agent_id(self) -> str:
        """Get agent_id for queries (empty string if None)."""
        return self.agent_id or ""
    
    def ls(self, prefix: Optional[str] = None) -> list[str]:
        """List files from database.
        
        Args:
            prefix: Optional path prefix to filter results.
        
        Returns:
            List of file paths.
        """
        with sqlite3.connect(self.db_path) as conn:
            if prefix is not None:
                # Use LIKE for prefix matching (escape special chars)
                like_pattern = prefix.replace("%", "\\%").replace("_", "\\_") + "%"
                cursor = conn.execute(
                    "SELECT key FROM files WHERE agent_id = ? AND key LIKE ? ESCAPE '\\\\'",
                    (self._get_agent_id(), like_pattern)
                )
            else:
                cursor = conn.execute(
                    "SELECT key FROM files WHERE agent_id = ?",
                    (self._get_agent_id(),)
                )
            
            return sorted([row[0] for row in cursor.fetchall()])
    
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
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT content FROM files WHERE key = ? AND agent_id = ?",
                (file_path, self._get_agent_id())
            )
            row = cursor.fetchone()
            
            if row is None:
                return f"Error: File '{file_path}' not found"
            
            try:
                content_lines = json.loads(row["content"])
                content = "\n".join(content_lines)
                
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
            except json.JSONDecodeError:
                return f"Error: Corrupted file data for '{file_path}'"
    
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
            Success message or error if file already exists.
        """
        # Check if file exists
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT 1 FROM files WHERE key = ? AND agent_id = ?",
                (file_path, self._get_agent_id())
            )
            if cursor.fetchone() is not None:
                return f"Cannot write to {file_path} because it already exists. Read and then make an edit, or write to a new path."
        
        # Create new file
        file_data = create_file_data(content)
        content_json = json.dumps(file_data["content"])
        agent_id = self._get_agent_id()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO files (key, agent_id, content, created_at, modified_at) VALUES (?, ?, ?, ?, ?)",
                (file_path, agent_id, content_json, file_data["created_at"], file_data["modified_at"])
            )
            conn.commit()
        
        return f"Updated file {file_path}"
    
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
            Success message or error message on failure.
        """
        # Get existing file
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT content, created_at FROM files WHERE key = ? AND agent_id = ?",
                (file_path, self._get_agent_id())
            )
            row = cursor.fetchone()
            
            if row is None:
                return f"Error: File '{file_path}' not found"
            
            try:
                content_lines = json.loads(row["content"])
                content = "\n".join(content_lines)
            except json.JSONDecodeError:
                return f"Error: Corrupted file data for '{file_path}'"
        
        # Perform replacement
        result = perform_string_replacement(content, old_string, new_string, replace_all)
        
        if isinstance(result, str):
            return result
        
        new_content, occurrences = result
        
        # Update file
        file_data = create_file_data(new_content, created_at=row["created_at"])
        content_json = json.dumps(file_data["content"])
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE files SET content = ?, modified_at = ? WHERE key = ? AND agent_id = ?",
                (content_json, file_data["modified_at"], file_path, self._get_agent_id())
            )
            conn.commit()
        
        return f"Successfully replaced {occurrences} instance(s) of the string in '{file_path}'"
    
    def delete(self, file_path: str) -> Command | None:
        """Delete file from database.
        
        Args:
            file_path: File path to delete
        
        Returns:
            None (direct database modification)
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM files WHERE key = ? AND agent_id = ?",
                (file_path, self._get_agent_id())
            )
            conn.commit()
        
        return None
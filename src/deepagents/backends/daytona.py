"""Daytona sandbox backend implementation."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

from daytona import Daytona, DaytonaConfig
from daytona.common.filesystem import FileUpload as DaytonaFileUpload

from deepagents.backends.fs import FileInfo, FileSystem, FileSystemCapabilities
from deepagents.backends.pagination import PageResults, PaginationCursor
from deepagents.backends.process import ExecuteResponse, Process, ProcessCapabilities
from deepagents.backends.sandbox import Sandbox, SandboxCapabilities, SandboxMetadata, SandboxProvider

if TYPE_CHECKING:
    from daytona import Sandbox as DaytonaSandboxClient


class DaytonaFileSystem(FileSystem):
    """Daytona filesystem implementation."""

    def __init__(self, sandbox: DaytonaSandboxClient) -> None:
        """Initialize the DaytonaFileSystem with a Daytona sandbox client."""
        self._sandbox = sandbox

    def ls(self, prefix: Optional[str] = None) -> list[FileInfo]:
        """List all file paths, optionally filtered by prefix."""
        path = prefix or "/"
        files = self._sandbox.fs.list_files(path)

        result: list[FileInfo] = []
        for file in files:
            # Convert Daytona format to our FileInfo format
            file_info: FileInfo = {
                "path": file.get("name", ""),
            }

            # Add optional fields if present
            if "is_dir" in file:
                file_info["kind"] = "dir" if file["is_dir"] else "file"
            if "size" in file:
                file_info["size"] = int(file["size"])
            if "mod_time" in file:
                file_info["modified_at"] = file["mod_time"]

            result.append(file_info)

        return result

    def upload_file(self, file: bytes, path: str, *, timeout: int = 30 * 60) -> None:
        """Upload a file to the sandbox."""
        self._sandbox.fs.upload_file(file, path, timeout=timeout)

    def download_file(self, path: str, *, timeout: int = 30 * 60) -> bytes:
        """Download a file from the sandbox."""
        return self._sandbox.fs.download_file(path, timeout=timeout)

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """Read file content with line numbers."""
        try:
            content = self.download_file(file_path)
            lines = content.decode("utf-8").splitlines()

            if not lines:
                return "System reminder: File exists but has empty contents"

            # Apply offset and limit
            selected_lines = lines[offset : offset + limit]

            # Format with line numbers (1-indexed)
            result = []
            for i, line in enumerate(selected_lines, start=offset + 1):
                result.append(f"{i:6d}\t{line}")

            return "\n".join(result)
        except Exception as e:
            return f"Error: File '{file_path}' not found"

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> str:
        """Edit a file by replacing string occurrences."""
        try:
            content = self.download_file(file_path)
            text = content.decode("utf-8")

            # Count occurrences
            count = text.count(old_string)

            if count == 0:
                return f"Error: String not found in file: '{old_string}'"

            if count > 1 and not replace_all:
                return (
                    f"Error: String '{old_string}' appears {count} times. "
                    f"Use replace_all=True to replace all occurrences."
                )

            # Perform replacement
            if replace_all:
                new_text = text.replace(old_string, new_string)
            else:
                new_text = text.replace(old_string, new_string, 1)

            # Upload the modified file
            self.upload_file(new_text.encode("utf-8"), file_path)

            return f"Successfully replaced {count} occurrence(s) in {file_path}"
        except Exception as e:
            return f"Error: File '{file_path}' not found"

    def delete(self, file_path: str) -> None:
        """Delete a file by path."""
        # Daytona doesn't have a direct delete method in the reference,
        # we would need to use exec or the filesystem API
        raise NotImplementedError("Delete not yet implemented for Daytona backend")

    def grep(
        self,
        pattern: str,
        path: str = "/",
        include: Optional[str] = None,
        output_mode: str = "files_with_matches",
    ) -> str:
        """Search for a pattern in files."""
        # This would require executing grep command in the sandbox
        raise NotImplementedError("Grep not yet implemented for Daytona backend")

    def glob(self, pattern: str, path: str = "/") -> list[str]:
        """Find files matching a glob pattern."""
        # This would require executing find command or similar in the sandbox
        raise NotImplementedError("Glob not yet implemented for Daytona backend")

    @property
    def get_capabilities(self) -> FileSystemCapabilities:
        """Get the filesystem capabilities."""
        return {
            "can_upload": True,
            "can_download": True,
            "can_list_files": True,
            "can_read": True,
            "can_edit": True,
            "can_delete": False,
            "can_grep": False,
            "can_glob": False,
        }


class DaytonaProcess(Process):
    """Daytona process implementation."""

    def __init__(self, sandbox: DaytonaSandboxClient) -> None:
        """Initialize the DaytonaProcess with a Daytona sandbox client."""
        self._sandbox = sandbox
        self._session_id = "main-exec-session"

    def execute(
        self,
        command: str,
        cwd: Optional[str] = None,
        *,
        timeout: int = 30 * 60,
    ) -> ExecuteResponse:
        """Execute a command in the process.

        Args:
            command: Command to execute as a string.
            cwd: Working directory to execute the command in.
            timeout: Maximum execution time in seconds (default: 30 minutes).
        """
        try:
            response = self._sandbox.process.execute_session_command(
                self._session_id, {"command": command}
            )
            return ExecuteResponse(
                result=response.get("result", ""),
                exit_code=response.get("exit_code"),
            )
        except Exception as e:
            return ExecuteResponse(
                result=f"Error executing command: {str(e)}",
                exit_code=1,
            )

    def get_capabilities(self) -> ProcessCapabilities:
        """Get the process capabilities."""
        return {
            "supports_exec": True,
        }


class DaytonaSandbox(Sandbox):
    """Daytona sandbox implementation."""

    def __init__(self, sandbox: DaytonaSandboxClient) -> None:
        """Initialize the DaytonaSandbox with a Daytona sandbox client."""
        self._sandbox = sandbox
        self._fs = DaytonaFileSystem(sandbox)
        self._process = DaytonaProcess(sandbox)

    @property
    def fs(self) -> FileSystem:
        """Filesystem backend."""
        return self._fs

    @property
    def process(self) -> Process:
        """Process backend."""
        return self._process

    @property
    def get_capabilities(self) -> SandboxCapabilities:
        """Get the capabilities of the sandbox backend."""
        return {
            "fs": self._fs.get_capabilities,
            "process": self._process.get_capabilities(),
        }

    @property
    def id(self) -> str:
        """Get the sandbox ID."""
        return self._sandbox.id


class DaytonaSandboxProvider(SandboxProvider):
    """Daytona sandbox provider implementation."""

    def __init__(
        self, *, daytona_client: Optional[Daytona] = None, api_key: Optional[str] = None
    ) -> None:
        """Initialize the DaytonaSandboxProvider with a Daytona client.

        Args:
            daytona_client: An existing Daytona client instance
            api_key: API key for creating a new Daytona client
        """
        if daytona_client and api_key:
            raise ValueError("Provide either daytona_client or api_key, not both.")

        if daytona_client is None:
            api_key = api_key or os.environ.get("DAYTONA_API_KEY")
            if api_key is None:
                raise ValueError("Either daytona_client or api_key must be provided.")
            config = DaytonaConfig(api_key=api_key)
            daytona_client = Daytona(config)

        self._client = daytona_client

    def get_or_create(self, id: str | None) -> Sandbox:
        """Get or create a sandbox instance by ID.

        If id is None, creates a new sandbox.
        If id is provided, retrieves the existing sandbox.
        """
        if id is None:
            # Create a new sandbox
            sandbox_client = self._client.create()
            # Create the main execution session
            sandbox_client.process.create_session("main-exec-session")
            return DaytonaSandbox(sandbox_client)
        else:
            # Get existing sandbox
            sandbox_client = self._client.get(id)
            return DaytonaSandbox(sandbox_client)

    def delete(self, id: str) -> None:
        """Delete a sandbox instance by ID.

        Do not raise an error if the sandbox does not exist.
        """
        try:
            sandbox = self._client.get(id)
            self._client.delete(sandbox)
        except Exception:
            # Silently ignore if sandbox doesn't exist
            pass

    def list(self, *, cursor: PaginationCursor | None, **kwargs) -> PageResults[SandboxMetadata]:
        """List all sandbox IDs.

        Note: Daytona's list() method returns a simple list of IDs,
        so we don't support pagination at the API level.
        """
        # Daytona's list returns list[str] of sandbox IDs
        sandbox_ids = self._client.list()

        # Convert to SandboxMetadata format
        items: list[SandboxMetadata] = [{"id": sid} for sid in sandbox_ids]

        # Since Daytona doesn't support pagination, we return all items
        return PageResults(
            items=items,
            cursor=PaginationCursor(
                next_cursor=None,
                has_more=False,
            ),
        )

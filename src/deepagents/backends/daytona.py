"""Daytona sandbox backend implementation."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

from daytona import Daytona, DaytonaConfig

from deepagents.backends.fs import FileInfo, FileSystem, FileSystemCapabilities
from deepagents.backends.pagination import PageResults, PaginationCursor
from deepagents.backends.process import ExecuteResponse, Process, ProcessCapabilities
from deepagents.backends.sandbox import Sandbox, SandboxCapabilities, SandboxMetadata, \
    SandboxProvider

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
        """Read file content with line numbers using a single shell command."""
        try:
            # Single command that checks file, reads lines, and formats with line numbers
            # tail -n +N starts from line N, head limits output, nl adds line numbers
            start_line = offset + 1
            cmd = (
                f"if [ ! -f '{file_path}' ]; then "
                f"echo 'Error: File not found'; exit 1; "
                f"elif [ ! -s '{file_path}' ]; then "
                f"echo 'System reminder: File exists but has empty contents'; "
                f"else "
                f"tail -n +{start_line} '{file_path}' | head -n {limit} | nl -ba -nln -w6 -s$'\\t' -v{start_line}; "
                f"fi"
            )
            result = self._sandbox.process.exec({"command": cmd})

            output = result.get("result", "").rstrip()
            exit_code = result.get("exit_code", 0)

            if exit_code != 0 or "Error: File not found" in output:
                return f"Error: File '{file_path}' not found"

            return output
        except Exception as e:
            return f"Error: File '{file_path}' not found"

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> str:
        """Edit a file by replacing string occurrences using a single shell command."""
        try:
            # Escape single quotes in the strings for shell safety
            old_escaped = old_string.replace("'", "'\\''")
            new_escaped = new_string.replace("'", "'\\''")

            # Use Python one-liner for complex string replacement logic
            python_code = (
                f"import sys; "
                f"text = open('{file_path}', 'r').read(); "
                f"old = '''{old_escaped}'''; "
                f"new = '''{new_escaped}'''; "
                f"count = text.count(old); "
                f"sys.exit(1) if count == 0 else (sys.exit(2) if count > 1 and not {replace_all} else None); "
                f"result = text.replace(old, new) if {replace_all} else text.replace(old, new, 1); "
                f"open('{file_path}', 'w').write(result); "
                f"print(count)"
            )

            cmd = f'python3 -c "{python_code}" 2>&1'
            result = self._sandbox.process.exec({"command": cmd})

            exit_code = result.get("exit_code", 0)
            output = result.get("result", "").strip()

            if exit_code == 1:
                return f"Error: String not found in file: '{old_string}'"
            elif exit_code == 2:
                # Get count from before the error
                return f"Error: String '{old_string}' appears multiple times. Use replace_all=True to replace all occurrences."
            elif exit_code != 0:
                return f"Error: File '{file_path}' not found"

            count = output
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
        """Search for a pattern in files using a single shell command."""
        try:
            # Build grep command based on output_mode
            grep_opts = "-r"  # recursive

            if output_mode == "files_with_matches":
                grep_opts += "l"  # files with matches
            elif output_mode == "count":
                grep_opts += "c"  # count per file

            # Add include pattern if specified
            include_pattern = ""
            if include:
                include_pattern = f"--include='{include}'"

            # Escape pattern for shell
            pattern_escaped = pattern.replace("'", "'\\''")

            cmd = f"grep {grep_opts} {include_pattern} -e '{pattern_escaped}' '{path}' 2>/dev/null || true"
            result = self._sandbox.process.exec({"command": cmd})

            return result.get("result", "").rstrip()
        except Exception as e:
            return ""

    def glob(self, pattern: str, path: str = "/") -> list[str]:
        """Find files matching a glob pattern using a single shell command."""
        try:
            # Escape pattern for shell
            pattern_escaped = pattern.replace("'", "'\\''")

            # Use Python's glob module for proper glob pattern matching
            python_code = (
                f"import glob; "
                f"import os; "
                f"os.chdir('{path}'); "
                f"results = sorted(glob.glob('{pattern_escaped}', recursive=True)); "
                f"print('\\n'.join(results))"
            )

            cmd = f'python3 -c "{python_code}" 2>/dev/null'
            result = self._sandbox.process.exec({"command": cmd})

            output = result.get("result", "").strip()
            if not output:
                return []

            return output.split("\n")
        except Exception as e:
            return []

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
            "can_grep": True,
            "can_glob": True,
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
        response = self._sandbox.process.exec(self._session_id, {"command": command})
        return ExecuteResponse(
            result=response.get("result", ""),
            exit_code=response.get("exit_code"),
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

    def __init__(self, *, client: Optional[Daytona] = None, api_key: Optional[str] = None) -> None:
        """Initialize the DaytonaSandboxProvider with a Daytona client.

        Args:
            client: An existing Daytona client instance
            api_key: API key for creating a new Daytona client
        """
        if client and api_key:
            raise ValueError("Provide either daytona_client or api_key, not both.")

        if client is None:
            api_key = api_key or os.environ.get("DAYTONA_API_KEY")
            if api_key is None:
                raise ValueError("Either daytona_client or api_key must be provided.")
            config = DaytonaConfig(api_key=api_key)
            client = Daytona(config)

        self._client = client

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

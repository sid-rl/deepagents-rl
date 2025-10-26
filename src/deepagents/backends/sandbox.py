import abc
from deepagents.backends.fs import FileSystem
from typing import TypedDict, NotRequired


class ExecuteResponse(TypedDict):
    """Result of code execution."""

    result: str
    """The output of the executed code.
    This will usually be the standard output of the command executed.
    """
    exit_code: NotRequired[int]
    """The exit code of the executed code, if applicable."""


class Sandbox(FileSystem, abc.ABC):
    """Abstract class for sandbox backends.
    
    Extends FileSystem to provide both filesystem and process execution capabilities.
    """

    @abc.abstractmethod
    def execute(
        self,
        command: str,
        cwd: str | None = None,
        *,
        timeout: int = 30 * 60,
    ) -> ExecuteResponse:
        """Execute a command in the sandbox.

        Args:
            command: Command to execute as a string.
            cwd: Working directory to execute the command in.
            timeout: Maximum execution time in seconds (default: 30 minutes).
        """
        ...

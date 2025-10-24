"""Abstraction for modeling a process."""

import abc
from typing import NotRequired

from typing import TypedDict


class ProcessCapabilities(TypedDict):
    """Capabilities of the process backend."""

    supports_exec: bool


class ExecuteResponse(TypedDict):
    """Result of code execution."""

    result: str
    """The output of the executed code.

    This will usually be the standard output of the command executed.
    """
    exit_code: NotRequired[int]
    """The exit code of the executed code, if applicable."""


class Process(abc.ABC):
    @abc.abstractmethod
    def execute(self, command: str) -> ExecuteResponse:
        """Execute a command in the process."""
        pass

    @abc.abstractmethod
    def get_capabilities(self) -> ProcessCapabilities: ...

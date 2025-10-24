import abc
from typing import TypedDict

from deepagents.backends.fs import FileSystem, FileSystemCapabilities
from deepagents.backends.pagination import PaginationCursor, PageResults
from deepagents.backends.process import Process, ProcessCapabilities


class SandboxCapabilities(TypedDict):
    """Capabilities of the sandbox backend."""

    fs: FileSystemCapabilities
    process: ProcessCapabilities


class Sandbox(abc.ABC):
    """Abstract class for sandbox backends."""

    fs: FileSystem
    """Filesystem backend."""
    process: Process
    """Process backend."""

    @property
    def get_capabilities(self) -> SandboxCapabilities:
        """Get the capabilities of the sandbox backend."""
        raise NotImplementedError()


class SandboxMetadata(TypedDict):
    """Metadata for a sandbox instance."""

    id: str
    """Unique identifier for the sandbox."""


class SandboxProvider(abc.ABC):
    """Abstract class for sandbox providers."""

    @abc.abstractmethod
    def get_or_create(self, id: str | None) -> Sandbox:
        """Get or create a sandbox instance by ID."""

    @abc.abstractmethod
    def delete(self, id: str) -> None:
        """Delete a sandbox instance by ID.

        Do not raise an error if the sandbox does not exist.
        """
        pass

    @abc.abstractmethod
    def list(self, *, cursor: PaginationCursor | None, **kwargs) -> PageResults[SandboxMetadata]:
        """List all sandbox IDs."""

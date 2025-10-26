"""Generalized factory for creating file system or shell tools based on configuration.

This module is meant to support interacting with ephemeral containers/sandboxes.
"""

from __future__ import annotations

from typing import Any, Literal, NotRequired

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import StateT
from langchain.tools import BaseTool, ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.runtime import Runtime
from langgraph.types import Command
from langgraph.typing import ContextT

from deepagents.backends.sandbox import Sandbox
from deepagents.backends.fs import FileSystem



def _get_tools(backend: Sandbox | FileSystem) -> list[BaseTool]:
    """Generate tools for interacting with the backend.

    Args:
        backend: The sandbox or filesystem to use.

    Returns:
        List of tools for backend interaction.
    """


    @tool
    def read(
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """Read a file from the sandbox filesystem."""
        return backend.read(file_path, offset, limit)

    @tool
    def edit(
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> str:
        """Edit a file in the sandbox filesystem."""
        return backend.edit(file_path, old_string, new_string, replace_all)


    @tool
    def ls(
        prefix: str | None = None,
    ) -> str:
        """List files in the sandbox filesystem."""
        return str(backend.ls(prefix))


    @tool
    def write(
        file_path: str,
    ) -> str:
        """Delete a file from the sandbox filesystem."""
        backend.write(file_path)
        return f"Successfully deleted {file_path}"

    @tool
    def grep(
        pattern: str,
        path: str = "/",
        include: str | None = None,
        output_mode: str = "files_with_matches",
    ) -> str:
        """Search for a pattern in files in the sandbox filesystem."""
        return backend.grep(pattern, path, include, output_mode)

    @tool
    def glob(
        pattern: str,
        path: str = "/",
    ) -> str:
        """Find files matching a glob pattern in the sandbox filesystem."""
        return str(backend.glob(pattern, path))

    @tool
    def bash(command: str) -> str:
        """Execute a bash command in the isolated sandbox environment.

        Use this tool to run shell commands, execute scripts, install dependencies,
        or perform any other bash operations needed to verify the documentation.

        Args:
            command: The bash command to execute (e.g., "python script.py", "pip install requests")

        Returns:
            The output from the command execution, including stdout and stderr.
        """
        execute_response = backend.execute(command)
        return f"Output:\n{execute_response['result']}\nExit Code: {execute_response['exit_code']}"


    return [ls, read, ]


class GeneralizedFilesystemMiddleware(AgentMiddleware):
    state_schema = SandboxState

    def __init__(
        self, backend: Sandbox | FileSystem
    ) -> None:
        """Initialize the Daytona sandbox middleware."""
        self.backend = backend

        self.tools = _get_tools(backend)

    @property
    def middleware(self):
        if hasattr(self.backend, "middleware"):
            return self.backend.middleware


class GeneralizedShellMiddleware(AgentMiddleware):
    state_schema = SandboxState

    def __init__(
        self,
        sandbox_provider: SandboxProvider,
        *,
        env: dict[str, str] | None = None,
        terminate_on_complete: bool = True,
    ) -> None:
        """Initialize the Daytona shell middleware."""
        self.sandbox_provider = sandbox_provider
        self.terminate_on_complete = terminate_on_complete
        self.env = env or {}

        @tool
        def bash(command: str, runtime: ToolRuntime) -> Command:
            """Execute a bash command in the isolated sandbox environment.

            Use this tool to run shell commands, execute scripts, install dependencies,
            or perform any other bash operations needed to verify the documentation.

            Args:
                command: The bash command to execute (e.g., "python script.py", "pip install requests")

            Returns:
                The output from the command execution, including stdout and stderr.
            """
            sandbox_id = runtime.state.get("sandbox_id")
            # If sandbox_id was not found, create a new sandbox
            created = sandbox_id is None
            sandbox = self.sandbox_provider.get_or_create(sandbox_id)
            # Execute command directly on sandbox
            # TODO(Eugene): Add env support
            execute_response = sandbox.process.execute(command)
            content = f"Output:\n{execute_response.result}\nExit Code: {execute_response.exit_code}"

            update = {
                "messages": [
                    ToolMessage(
                        content=content,
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }

            if created:
                update["sandbox_id"] = sandbox.id

            return Command(update=update)

        self.tools = [bash]

    def after_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        """Clean up the sandbox after the agent completes."""
        sandbox_id = state.get("sandbox_id")

        if sandbox_id and self.terminate_on_complete:
            self.sandbox_provider.delete(sandbox_id)

        return None


#!/usr/bin/env python3
"""Documentation Verifier Agent

This script verifies technical documentation by extracting Python code snippets
and creating executable scripts to test that the documentation is correct.
"""

import argparse
import os
from pathlib import Path
from typing import Optional, Any

from daytona import Daytona, DaytonaConfig, Sandbox
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.tools import tool, ToolRuntime
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
from typing_extensions import NotRequired

from deepagents import create_deep_agent


class DaytonaSandbox:
    """Simplified wrapper for Daytona sandbox operations."""

    def __init__(self, sandbox: Sandbox):
        """Initialize with a Daytona sandbox instance."""
        self.sandbox = sandbox
        self.exec_session_id = "main-exec-session"

    @property
    def id(self) -> str:
        """Get the sandbox ID."""
        return self.sandbox.id

    def exec(self, command: str, cwd: Optional[str] = None, *, timeout: int = 30 * 60):
        """Execute a command in the sandbox.

        Args:
            command: Command to execute as list of arguments.
            cwd: Working directory to execute the command in.
            timeout: Maximum execution time in seconds.

        Returns:
            Dictionary with 'result' and 'exit_code' keys.
        """
        execute_response = self.sandbox.process.exec(command, cwd=cwd, timeout=timeout)
        return {
            "result": execute_response.result,
            "exit_code": execute_response.exit_code,
        }


class DaytonaSandboxManager:
    """Simplified sandbox manager for creating and managing Daytona sandboxes."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        """Initialize the SandboxManager with a Daytona client.

        Args:
            api_key: Daytona API key. If not provided, uses DAYTONA_API_KEY env var.
        """
        api_key = api_key or os.environ.get("DAYTONA_API_KEY")
        if api_key is None:
            raise ValueError("Either api_key parameter or DAYTONA_API_KEY environment variable must be provided.")

        config = DaytonaConfig(api_key=api_key)
        self.daytona_client = Daytona(config)

    def create(self, **kwargs) -> DaytonaSandbox:
        """Create and return a new sandbox.

        Returns:
            A SimpleDaytonaSandbox instance.
        """
        sandbox = self.daytona_client.create(**kwargs)
        # Create a session for execution
        session_id = "main-exec-session"
        sandbox.process.create_session(session_id)
        return DaytonaSandbox(sandbox)

    def get_or_create(self, id: str | None = None, **kwargs) -> DaytonaSandbox:
        """Retrieve an existing sandbox by ID or create a new one if ID is None.

        Args:
            id: The sandbox ID to retrieve. If None, a new sandbox is created.

        Returns:
            A SimpleDaytonaSandbox instance.
        """
        if id is not None:
            return self.get(id)
        else:
            return self.create(**kwargs)

    def delete(self, id: str) -> None:
        """Delete a sandbox by its ID.

        Args:
            id: The sandbox ID to delete.
        """
        sandbox = self.daytona_client.get(id)
        self.daytona_client.delete(sandbox)

    def get(self, id: str) -> DaytonaSandbox:
        """Retrieve a sandbox by its ID.

        Args:
            id: The sandbox ID to retrieve.

        Returns:
            A SimpleDaytonaSandbox instance.
        """
        sandbox = self.daytona_client.get(id)
        return DaytonaSandbox(sandbox)


# Custom state schema with sandbox_id
class DocVerifierState(AgentState):
    """State schema for the documentation verifier agent."""

    sandbox_id: NotRequired[str]


# Sandbox middleware
class SandboxMiddleware(AgentMiddleware[DocVerifierState]):
    """Middleware that manages a Daytona sandbox for code execution."""

    state_schema = DocVerifierState

    def __init__(self, api_key: str | None = None):
        """Initialize the sandbox middleware.

        Args:
            api_key: Daytona API key. If not provided, uses DAYTONA_API_KEY env var.
        """
        super().__init__()
        self.sandbox_manager = DaytonaSandboxManager(api_key=api_key)

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
            sandbox = self.sandbox_manager.get_or_create(runtime.state.get("sandbox_id"))
            result = sandbox.exec(command)
            content = f"Output:\n{result['result']}\nExit Code: {result['exit_code']}"
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=content,
                            tool_call_id=runtime.tool_call_id,
                        )
                    ],
                }
            )

        self.tools = [bash]

    def after_agent(self, state: DocVerifierState, runtime: Runtime) -> dict[str, Any] | None:
        """Clean up the sandbox after the agent completes."""
        sandbox_id = state.get("sandbox_id")

        if sandbox_id:
            try:
                self.sandbox_manager.delete(sandbox_id)
                print("Sandbox deleted successfully")
            except Exception as e:
                print(f"Warning: Failed to delete sandbox: {e}")

        return None


# System prompt for the documentation verifier agent
DOC_VERIFIER_PROMPT = """You are an expert documentation verification agent. Your job is to verify that technical documentation works correctly by extracting and testing code snippets.

Your workflow should be:

1. Read the markdown documentation file that was provided
2. Extract all Python code snippets from the documentation
3. Analyze the code snippets to understand their dependencies and execution order
4. Create a verification script that:
   - Sets up any necessary imports or dependencies
   - Runs each code snippet in the correct order
   - Validates that the code executes without errors
   - Tests any assertions or expected outputs mentioned in the documentation
5. Execute the verification script to test the documentation
6. Report any issues found, including:
   - Syntax errors
   - Runtime errors
   - Missing imports or dependencies
   - Incorrect examples or outputs
7. If all tests pass, provide a summary confirming the documentation is valid

Important guidelines:
- Be thorough in extracting ALL code snippets, including inline code
- Pay attention to code block language hints (```python vs ```bash, etc.)
- Consider the context around code snippets for understanding intended behavior
- Create isolated test environments when needed to avoid conflicts
- Provide clear, actionable feedback on any issues found
- The verification script should be named 'verify_{original_doc_name}.py'

You have access to a bash tool that executes commands in an isolated Daytona sandbox environment.
Use this tool to:
- Create and run verification scripts
- Install dependencies (e.g., "pip install requests")
- Execute Python code
- Run any bash commands needed for testing

The sandbox is automatically created at the start and cleaned up at the end.
"""


def main():
    """Main entry point for the documentation verifier."""
    parser = argparse.ArgumentParser(description="Verify technical documentation by extracting and testing code snippets")
    parser.add_argument("markdown_file", type=str, help="Path to the markdown documentation file to verify")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./verification_output",
        help="Directory to store verification scripts and results (default: ./verification_output)",
    )

    args = parser.parse_args()

    # Validate input file
    markdown_path = Path(args.markdown_file)
    if not markdown_path.exists():
        print(f"Error: File not found: {args.markdown_file}")
        return 1

    if not markdown_path.suffix.lower() in [".md", ".markdown"]:
        print(f"Warning: File does not have .md or .markdown extension: {args.markdown_file}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read the markdown content
    with open(markdown_path, "r", encoding="utf-8") as f:
        markdown_content = f.read()

    print(f"Verifying documentation: {markdown_path}")
    print(f"Output directory: {output_dir}")
    print("-" * 80)

    # Create the sandbox middleware
    sandbox_middleware = SandboxMiddleware()

    # Create the agent with sandbox middleware
    agent = create_deep_agent(
        system_prompt=DOC_VERIFIER_PROMPT,
        middleware=[sandbox_middleware],
    )

    # Prepare the initial message with the documentation
    initial_message = f"""Please verify the following technical documentation.

Documentation file: {markdown_path.name}
Output directory for verification scripts: {output_dir.absolute()}

Documentation content:

```markdown
{markdown_content}
```

Please extract all Python code snippets, create a verification script, and test that the documentation is correct.
"""

    # Run the agent with streaming
    print("Starting documentation verification...")
    print()

    # Stream the agent's execution to see intermediate outputs
    for chunk in agent.stream({"messages": [{"role": "user", "content": initial_message}]}):
        # Print each chunk as it arrives for real-time feedback
        print(chunk)
        print()  # Add spacing between chunks

    # Print completion message
    print()
    print("=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())

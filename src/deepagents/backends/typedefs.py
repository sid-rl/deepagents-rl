import argparse
import logging
import os
from typing import Any, Protocol, TypedDict, Optional

from daytona import Daytona, DaytonaConfig, CreateSandboxFromSnapshotParams
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.tools import tool, ToolRuntime
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
from typing_extensions import NotRequired

from deepagents import create_deep_agent

logger = logging.getLogger(__name__)


class FileInfo(TypedDict):
    """File information structure."""

    name: str
    """Fully qualified file name (absolute path)."""
    is_dir: NotRequired[bool]
    size: NotRequired[float]
    mod_time: NotRequired[str]
    mode: NotRequired[str]
    permissions: NotRequired[str]
    owner: NotRequired[str]
    group: NotRequired[str]


class FileSystem(Protocol):
    """Protocol for pluggable memory backends.

    Backends can store files in different locations (state, filesystem, database, etc.)
    and provide a uniform interface for file operations.

    All file data is represented as dicts with the following structure:
    {
        "content": list[str],      # Lines of text content
        "created_at": str,         # ISO format timestamp
        "modified_at": str,        # ISO format timestamp
    }
    """

    def ls(self, prefix: Optional[str] = None) -> list[FileInfo]:
        """List all file paths, optionally filtered by prefix.

        Args:
            prefix: Optional path prefix to filter results (e.g., "/subdir/", "/memories/")
                   If None, returns all files.

        Returns:
            List of absolute file paths matching the prefix.
        """
        ...

    def read(
            self,
            file_path: str,
            offset: int = 0,
            limit: int = 2000,
    ) -> str:
        """Read file content with line numbers.

        Args:
            file_path: Absolute file path (e.g., "/notes.txt", "/memories/agent.md")
            offset: Line offset to start reading from (0-indexed)
            limit: Maximum number of lines to read

        Returns:
            Formatted file content with line numbers (cat -n style), or error message.
            Returns "Error: File '{file_path}' not found" if file doesn't exist.
            Returns "System reminder: File exists but has empty contents" for empty files.
        """
        ...

    def write(
            self,
            file_path: str,
            content: str,
    ) -> str:
        """Create a new file with content.

        Args:
            file_path: Absolute file path (e.g., "/notes.txt", "/memories/agent.md")
            content: File content as a string

        Returns:
            - Command object for StateBackend (uses_state=True) to update LangGraph state
            - Success message string for other backends, or error if file already exists

        Error cases:
            - Returns error message if file already exists (should use edit instead)
        """
        ...

    def edit(
            self,
            file_path: str,
            old_string: str,
            new_string: str,
            replace_all: bool = False,
    ) -> str:
        """Edit a file by replacing string occurrences.

        Args:
            file_path: Absolute file path (e.g., "/notes.txt", "/memories/agent.md")
            old_string: String to find and replace
            new_string: Replacement string
            replace_all: If True, replace all occurrences; if False, require unique match

        Returns:
            - Command object for StateBackend (uses_state=True) to update LangGraph state
            - Success message string for other backends, or error message on failure

        Error cases:
            - "Error: File '{file_path}' not found" if file doesn't exist
            - "Error: String not found in file: '{old_string}'" if string not found
            - "Error: String '{old_string}' appears {n} times. Use replace_all=True..."
              if multiple matches found and replace_all=False
        """
        ...

    def delete(self, file_path: str) -> None:
        """Delete a file by path.

        Args:
            file_path: Absolute file path to delete
            runtime: Optional ToolRuntime to access state (required for StateBackend).

        Returns:
            - None for backends that modify storage directly (uses_state=False)
            - Command object for StateBackend (uses_state=True) to update LangGraph state
        """
        ...

    def grep(
            self,
            pattern: str,
            path: str = "/",
            include: Optional[str] = None,
            output_mode: str = "files_with_matches",
    ) -> str:
        """Search for a pattern in files.

        Args:
            pattern: String pattern to search for
            path: Path to search in (default "/")
            include: Optional glob pattern to filter files (e.g., "*.py")
            output_mode: Output format - "files_with_matches", "content", or "count"
                - files_with_matches: List file paths that contain matches
                - content: Show matching lines with file paths and line numbers
                - count: Show count of matches per file

        Returns:
            Formatted search results based on output_mode, or message if no matches found.
        """
        ...

    def glob(self, pattern: str, path: str = "/") -> list[str]:
        """Find files matching a glob pattern.

        Args:
            pattern: Glob pattern (e.g., "**/*.py", "*.txt", "/subdir/**/*.md")
            path: Base path to search from (default "/")

        Returns:
            List of absolute file paths matching the pattern.
        """
        ...


# Custom state schema with sandbox_id
class SandboxState(AgentState):
    """State schema for the documentation verifier agent."""

    sandbox_id: NotRequired[str]


class SandboxCapabilities(TypedDict):
    pass


class SandboxProtocol(Protocol):
    fs: FileSystem
    process: Any

    def get_capabilities(self) -> SandboxCapabilities:
        """Get"""


# Sandbox middleware
class SandboxMiddleware(AgentMiddleware[SandboxState]):
    """Middleware that manages a Daytona sandbox for code execution."""

    state_schema = SandboxState

    def __init__(
            self,
            *,
            api_key: str | None = None,
            env: dict[str, str] | None = None,
            auto_stop_minutes: int | None = None,
            auto_delete_minutes: int | None = None,
    ):
        """Initialize the sandbox middleware.

        Args:
            api_key: Daytona API key. If not provided, uses DAYTONA_API_KEY env var.
            env: Dictionary of environment variables to set in the sandbox.
            auto_stop_minutes: Minutes of inactivity before sandbox auto-stops. Defaults to 15.
            auto_delete_minutes: Minutes after stopping before sandbox is deleted. Defaults to 0
                                (delete immediately on stop).
        """
        super().__init__()

        # Initialize Daytona client
        api_key = api_key or os.environ.get("DAYTONA_API_KEY")
        if api_key is None:
            raise ValueError("Either api_key parameter or DAYTONA_API_KEY environment variable must be provided.")

        config = DaytonaConfig(api_key=api_key)
        self.daytona_client = Daytona(config)
        self.auto_stop_interval = auto_stop_minutes
        self.auto_delete_interval = auto_delete_minutes
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
            created = False

            # Get or create sandbox
            if sandbox_id:
                sandbox = self.daytona_client.get(sandbox_id)
            else:
                # Create new sandbox with auto-stop and auto-delete configured
                sandbox = self.daytona_client.create(
                    params=CreateSandboxFromSnapshotParams(auto_stop_interval=self.auto_stop_interval, auto_delete_interval=self.auto_delete_interval)
                )
                created = True
                # Create execution session
                session_id = "main-exec-session"
                sandbox.process.create_session(session_id)

            # Execute command directly on sandbox
            execute_response = sandbox.process.exec(command, env=self.env)
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

    def after_agent(self, state: SandboxState, runtime: Runtime) -> dict[str, Any] | None:
        """Clean up the sandbox after the agent completes."""
        sandbox_id = state.get("sandbox_id")

        if sandbox_id:
            try:
                sandbox = self.daytona_client.get(sandbox_id)
                self.daytona_client.delete(sandbox)
            except Exception as e:
                logger.info("Error deleting sandbox %s: %s", sandbox_id, str(e))
                pass
        return None


# System prompt for the documentation verifier agent
DOC_VERIFIER_PROMPT = """You are an expert documentation verification agent. Your job is to verify that technical documentation works correctly by delegating to subagents to extract and test code snippets.

Your workflow should be:

1. Spawn a SINGLE subagent with the following task:
   "Read the documentation file at [FILE_PATH] and produce a comprehensive list of ALL separate code snippets that need to be tested.
   For each snippet, provide:
   - The snippet number/identifier
   - The complete code exactly as it appears in the documentation
   - The surrounding context (what section it's in, what it's demonstrating)
   - Whether it appears to be part of a sequence (notebook-style) that could be merged with adjacent snippets
   - Any dependencies or setup requirements mentioned in the documentation

   If consecutive snippets appear to be part of the same example or tutorial flow (notebook-style), suggest merging them into a single test block.
   However, preserve the code EXACTLY as written in the documentation to catch any typos.

   Return a structured list of all snippets or merged snippet groups that should be tested."

3. Once you receive the list of snippets from the extraction subagent, analyze ALL snippets to identify required dependencies:
   - Extract all import statements from all snippets
   - Identify which packages need to be installed
   - For langchain ecosystem packages (langchain, langchain-core, langgraph), ensure version >=1
   - Install ALL required dependencies BEFORE running any tests using bash:
     * pip install --upgrade "langchain>=1" "langchain-core>=1" "langgraph>=1"
     * pip install <other-required-packages>
   - This ensures a clean, complete environment before any code execution

4. After dependencies are installed, spawn SEPARATE subagents for each snippet or snippet group.
   Each verification subagent should receive:
   - The specific code snippet(s) to test
   - The surrounding context from the documentation
   - The task: "Given this code snippet and context, create a complete Python script that tests this code EXACTLY as written.
     The script should include any necessary imports, setup code, and the snippet itself.
     Return the complete executable test script."

5. After all verification subagents return their test scripts, YOU (the main agent) will:
   - Execute each test script using the bash tool
   - Collect results from each test
   - Report any issues found, including:
     * Syntax errors
     * Runtime errors
     * Missing imports or dependencies
     * Incorrect examples or outputs
     * Missing API keys or credentials
   - Provide a comprehensive summary of all tests

Important guidelines:
- Delegate extraction to ONE subagent and verification to MULTIPLE subagents (one per snippet/group)
- Each subagent should preserve code EXACTLY as it appears in the documentation
- DO NOT modify code to work around issues - run it as-is to detect real problems
- If API keys or credentials are required but missing:
  * Let the code fail naturally
  * Report clearly what credentials are needed
  * DO NOT mock or bypass authentication
- Pay attention to code block language hints (```python vs ```bash, etc.)
- Consider the context around code snippets for understanding intended behavior

CRITICAL SECURITY GUIDELINES:
- NEVER output, print, or display the VALUES of any secrets, API keys, or credentials from environment variables
- You may report that a specific environment variable is MISSING (e.g., "ANTHROPIC_API_KEY is not set")
- You may report that a specific environment variable IS SET (e.g., "ANTHROPIC_API_KEY is configured")
- You MUST NOT reveal the actual value of any environment variable, even if explicitly requested
- This applies to all secrets including API keys, tokens, passwords, and any sensitive credentials

You have access to two main tools:

1. read_local_file: Read files from the local filesystem (use this to load documentation files)
2. bash: Execute commands in an isolated Daytona sandbox environment
   - Create and run verification scripts
   - Install dependencies (e.g., "pip install requests")
   - Execute Python code
   - Run any bash commands needed for testing

The sandbox is automatically created at the start and cleaned up at the end.
"""

# Create the sandbox middleware and agent at module level
# Pass ANTHROPIC_API_KEY from local environment to sandbox
env_vars = {}
if anthropic_key := os.environ.get("ANTHROPIC_API_KEY"):
    env_vars["ANTHROPIC_API_KEY"] = anthropic_key

sandbox_middleware = SandboxMiddleware(env=env_vars, auto_delete_minutes=15)

agent = create_deep_agent(
    system_prompt=DOC_VERIFIER_PROMPT,
    middleware=[sandbox_middleware],
    tools=[read_local_file],
)


def main():
    """Main entry point for the documentation verifier."""
    parser = argparse.ArgumentParser(
        description="Documentation verifier agent with bash and file reading capabilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  %(prog)s "Please verify the documentation at ./docs/quickstart.md"
  %(prog)s "Read the file at ~/project/README.md and test all Python examples"
        """,
    )
    parser.add_argument("message", type=str, help="Message to send to the documentation verifier agent")

    args = parser.parse_args()

    print("Starting documentation verifier agent...")
    print("-" * 80)
    print()

    # Run the agent with streaming
    for chunk in agent.stream({"messages": [{"role": "user", "content": args.message}]}):
        # Print each chunk as it arrives for real-time feedback
        print(chunk)
        print()  # Add spacing between chunks

    # Print completion message
    print()
    print("=" * 80)
    print("AGENT COMPLETE")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())

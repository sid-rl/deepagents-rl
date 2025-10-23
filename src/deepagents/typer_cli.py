#!/usr/bin/env python3
"""Interactive CLI for DeepAgents using Typer for a great developer experience.

This module provides a rich, interactive command-line interface with:
- Streaming todo list updates
- Interactive question/answer prompts
- Real-time agent output
- Beautiful formatting with Rich
"""

import asyncio
import os
from pathlib import Path
from typing import Annotated, Optional

import dotenv
import typer
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.progress import Progress, SpinnerColumn, TextColumn
from tavily import TavilyClient

from deepagents import create_deep_agent
from deepagents.memory.backends.filesystem import FilesystemBackend
from deepagents.middleware.agent_memory import AgentMemoryMiddleware
from langchain.agents.middleware import ShellToolMiddleware, HostExecutionPolicy
from langgraph.checkpoint.memory import InMemorySaver

dotenv.load_dotenv()

# Initialize Typer app
app = typer.Typer(
    name="deepagents",
    help="Interactive AI coding assistant with streaming todos and rich output",
    add_completion=False,
    pretty_exceptions_enable=True,
)

console = Console()


# Tool definitions (reused from original cli.py)
tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY")) if os.environ.get("TAVILY_API_KEY") else None


def http_request(
    url: str,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    data: str | dict | None = None,
    params: dict[str, str] | None = None,
    timeout: int = 30,
) -> dict:
    """Make HTTP requests to APIs and web services."""
    import requests

    try:
        kwargs = {"url": url, "method": method.upper(), "timeout": timeout}

        if headers:
            kwargs["headers"] = headers
        if params:
            kwargs["params"] = params
        if data:
            if isinstance(data, dict):
                kwargs["json"] = data
            else:
                kwargs["data"] = data

        response = requests.request(**kwargs)

        try:
            content = response.json()
        except Exception:
            content = response.text

        return {
            "success": response.status_code < 400,
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "content": content,
            "url": response.url,
        }

    except requests.exceptions.Timeout:
        return {
            "success": False,
            "status_code": 0,
            "headers": {},
            "content": f"Request timed out after {timeout} seconds",
            "url": url,
        }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "status_code": 0,
            "headers": {},
            "content": f"Request error: {e!s}",
            "url": url,
        }
    except Exception as e:
        return {
            "success": False,
            "status_code": 0,
            "headers": {},
            "content": f"Error making request: {e!s}",
            "url": url,
        }


def web_search(
    query: str,
    max_results: int = 5,
    topic: str = "general",
    include_raw_content: bool = False,
) -> dict:
    """Search the web using Tavily for programming-related information."""
    if tavily_client is None:
        return {"error": "Tavily API key not configured. Please set TAVILY_API_KEY environment variable.", "query": query}

    try:
        search_docs = tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic,
        )
        return search_docs
    except Exception as e:
        return {"error": f"Web search error: {e!s}", "query": query}


def get_default_coding_instructions() -> str:
    """Get the default coding agent instructions."""
    default_prompt_path = Path(__file__).parent / "default_agent_prompt.md"
    return default_prompt_path.read_text()


# Constants for display
MAX_ARG_LENGTH = 200
MAX_RESULT_LENGTH = 200

# Tool icons mapping
TOOL_ICONS = {
    "read_file": "📖",
    "write_file": "✏️",
    "edit_file": "✂️",
    "ls": "📁",
    "glob": "🔍",
    "grep": "🔎",
    "shell": "⚡",
    "web_search": "🌐",
    "http_request": "🌍",
    "task": "🤖",
    "write_todos": "📋",
}


def truncate_value(value: str, max_length: int = MAX_ARG_LENGTH) -> str:
    """Truncate a string value if it exceeds max_length."""
    if len(value) > max_length:
        return value[:max_length] + "..."
    return value


def format_tool_args(tool_input: dict) -> str:
    """Format tool arguments for display, truncating long values."""
    if not tool_input:
        return ""

    args_parts = []
    for key, value in tool_input.items():
        value_str = str(value)
        value_str = truncate_value(value_str)
        args_parts.append(f"{key}={value_str}")

    return ", ".join(args_parts)


class TodoTracker:
    """Tracks and displays todos in real-time with streaming updates."""

    def __init__(self):
        self.todos: list[dict] = []
        self.live: Live | None = None

    def update_todos(self, new_todos: list[dict]) -> None:
        """Update the todo list and refresh the display."""
        self.todos = new_todos
        if self.live:
            self.live.update(self.render())

    def render(self) -> Table:
        """Render the current todo list as a Rich table."""
        table = Table(title="📋 Task Progress", box=box.ROUNDED, show_header=True, header_style="bold magenta")
        table.add_column("Status", style="dim", width=10)
        table.add_column("Task", style="white")

        for todo in self.todos:
            status = todo.get("status", "pending")
            content = todo.get("activeForm" if status == "in_progress" else "content", "Unknown task")

            if status == "completed":
                status_icon = "[green]✓[/green]"
                style = "dim green"
            elif status == "in_progress":
                status_icon = "[yellow]⋯[/yellow]"
                style = "bold yellow"
            else:
                status_icon = "[dim]○[/dim]"
                style = "dim"

            table.add_row(status_icon, content, style=style)

        return table

    def start_live_display(self) -> None:
        """Start the live display for streaming updates."""
        if self.live is None:
            self.live = Live(self.render(), console=console, refresh_per_second=4)
            self.live.start()

    def stop_live_display(self) -> None:
        """Stop the live display."""
        if self.live:
            self.live.stop()
            self.live = None


def display_tool_call(tool_name: str, tool_input: dict) -> None:
    """Display a tool call with icon and arguments."""
    icon = TOOL_ICONS.get(tool_name, "🔧")
    args_str = format_tool_args(tool_input)

    if args_str:
        console.print(f"[dim]{icon} {tool_name}({args_str})[/dim]")
    else:
        console.print(f"[dim]{icon} {tool_name}()[/dim]")


def display_text_content(text: str) -> None:
    """Display text content with markdown rendering."""
    if text.strip():
        if "```" in text or "#" in text or "**" in text:
            md = Markdown(text)
            console.print(md)
        else:
            console.print(text, style="white")


def extract_and_display_content(message_content) -> None:
    """Extract and display content from agent messages."""
    if isinstance(message_content, str):
        display_text_content(message_content)
        return

    if isinstance(message_content, list):
        for block in message_content:
            if isinstance(block, dict):
                if block.get("type") == "text" and "text" in block:
                    display_text_content(block["text"])
                elif block.get("type") == "tool_use":
                    tool_name = block.get("name", "unknown_tool")
                    tool_input = block.get("input", {})
                    display_tool_call(tool_name, tool_input)


def execute_task_with_streaming(user_input: str, agent, assistant_id: str | None, show_todos: bool = True) -> None:
    """Execute a task with streaming output and todo tracking."""
    console.print()

    todo_tracker = TodoTracker()

    config = {"configurable": {"thread_id": "main"}, "metadata": {"assistant_id": assistant_id} if assistant_id else {}}

    # Start streaming
    for _, chunk in agent.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        stream_mode="updates",
        subgraphs=True,
        config=config,
        durability="exit",
    ):
        chunk = list(chunk.values())[0]
        if chunk is not None:
            # Check for interrupts
            if "__interrupt__" in chunk:
                break

            # Check for todo updates
            if show_todos and "todos" in chunk:
                todos = chunk["todos"]
                if isinstance(todos, list):
                    if not todo_tracker.live:
                        todo_tracker.start_live_display()
                    todo_tracker.update_todos(todos)

            # Process messages
            if "messages" in chunk and chunk["messages"]:
                last_message = chunk["messages"][-1]

                message_content = None
                message_role = getattr(last_message, "type", None)
                if isinstance(message_role, dict):
                    message_role = last_message.get("role", "unknown")

                if hasattr(last_message, "content"):
                    message_content = last_message.content
                elif isinstance(last_message, dict) and "content" in last_message:
                    message_content = last_message["content"]

                if message_content:
                    # Show tool calls
                    if message_role != "tool":
                        if isinstance(message_content, list):
                            for block in message_content:
                                if isinstance(block, dict) and block.get("type") == "tool_use":
                                    tool_name = block.get("name", "unknown_tool")
                                    tool_input = block.get("input", {})
                                    display_tool_call(tool_name, tool_input)

                    # Show tool results
                    if message_role == "tool":
                        result_str = str(message_content)
                        result_str = truncate_value(result_str, MAX_RESULT_LENGTH)
                        console.print(f"[dim]  → {result_str}[/dim]")

                    # Show text content
                    if message_role != "tool":
                        has_text_content = False
                        if isinstance(message_content, str):
                            has_text_content = True
                        elif isinstance(message_content, list):
                            for block in message_content:
                                if isinstance(block, dict):
                                    if block.get("type") == "text" and block.get("text", "").strip():
                                        has_text_content = True
                                        break

                        if has_text_content:
                            extract_and_display_content(message_content)
                            console.print()

    # Stop live display when done
    todo_tracker.stop_live_display()

    # Show final todo state if any
    if show_todos and todo_tracker.todos:
        console.print()
        console.print(todo_tracker.render())

    console.print()


async def interactive_session(agent, assistant_id: str | None, show_todos: bool = True) -> None:
    """Main interactive CLI loop with rich prompts."""
    console.print()
    console.print(
        Panel.fit("[bold cyan]DeepAgents[/bold cyan] [dim]|[/dim] Interactive AI Coding Assistant", border_style="cyan", box=box.DOUBLE)
    )
    console.print("[dim]Type 'quit' to exit, 'help' for commands[/dim]")

    if tavily_client is None:
        console.print("[yellow]⚠ Web search disabled:[/yellow] TAVILY_API_KEY not found.")
        console.print("[dim]  Set TAVILY_API_KEY to enable web search (get key at https://tavily.com)[/dim]")

    console.print()

    while True:
        try:
            console.print("[bold green]❯[/bold green] ", end="")
            user_input = input().strip()
        except EOFError:
            break
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted[/yellow]")
            continue

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit", "q"]:
            console.print("\n[bold cyan]👋 Goodbye![/bold cyan]\n")
            break

        if user_input.lower() == "help":
            show_interactive_help()
            continue

        if user_input.lower() == "clear":
            console.clear()
            continue

        # Execute the task with streaming
        console.print("[dim]✓ Processing...[/dim]")
        execute_task_with_streaming(user_input, agent, assistant_id, show_todos)


def show_interactive_help() -> None:
    """Show help for interactive session commands."""
    help_text = """
[bold cyan]Interactive Commands:[/bold cyan]

  [bold]help[/bold]     Show this help message
  [bold]clear[/bold]    Clear the console
  [bold]quit[/bold]     Exit the session (also: exit, q)

[bold cyan]Features:[/bold cyan]

  • Real-time streaming output
  • Live todo list updates
  • Tool call visualization with icons
  • Markdown rendering
  • Persistent memory across sessions
    """
    console.print(Panel(help_text, border_style="cyan", box=box.ROUNDED))


@app.command()
def chat(
    agent: Annotated[Optional[str], typer.Option("--agent", "-a", help="Agent name for separate memory stores")] = "agent",
    show_todos: Annotated[bool, typer.Option("--show-todos/--no-todos", help="Show streaming todo list updates")] = True,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose output")] = False,
) -> None:
    """Start an interactive chat session with the coding agent.

    This is the main command for interacting with DeepAgents. It provides:
    - Real-time streaming output
    - Live todo list tracking
    - Interactive prompts
    - Persistent memory

    Examples:
        deepagents chat                          # Start with default agent
        deepagents chat --agent mybot            # Use specific agent
        deepagents chat --no-todos              # Disable todo tracking
        deepagents chat --verbose               # Show more details
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[bold red]Error:[/bold red] ANTHROPIC_API_KEY environment variable is not set.")
        console.print("Please set your Anthropic API key:")
        console.print("  export ANTHROPIC_API_KEY=your_api_key_here")
        console.print("\nOr add it to your .env file.")
        raise typer.Exit(1)

    # Build agent
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task(description="Initializing agent...", total=None)

        tools = [http_request]
        if tavily_client is not None:
            tools.append(web_search)

        shell_middleware = ShellToolMiddleware(workspace_root=os.getcwd(), execution_policy=HostExecutionPolicy())

        backend = FilesystemBackend()

        # Long-term memory setup
        agent_dir = Path.home() / ".deepagents" / agent
        agent_dir.mkdir(parents=True, exist_ok=True)
        agent_md = agent_dir / "agent.md"
        if not agent_md.exists():
            source_content = get_default_coding_instructions()
            agent_md.write_text(source_content)

        long_term_backend = FilesystemBackend(root_dir=agent_dir, virtual_mode=True)

        agent_middleware = [AgentMemoryMiddleware(backend=long_term_backend), shell_middleware]
        system_prompt = f"""### Current Working Directory

The filesystem backend is currently operating in: `{Path.cwd()}`"""

        deep_agent = create_deep_agent(
            system_prompt=system_prompt,
            tools=tools,
            memory_backend=backend,
            use_longterm_memory=long_term_backend,
            middleware=agent_middleware,
        ).with_config({"recursion_limit": 1000})

        deep_agent.checkpointer = InMemorySaver()

    # Start interactive session
    try:
        asyncio.run(interactive_session(deep_agent, agent, show_todos))
    except KeyboardInterrupt:
        console.print("\n\n[bold cyan]👋 Goodbye![/bold cyan]\n")
    except Exception as e:
        console.print(f"\n[bold red]❌ Error:[/bold red] {e}\n")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def run(
    prompt: Annotated[str, typer.Argument(help="The task or question for the agent")],
    agent: Annotated[Optional[str], typer.Option("--agent", "-a", help="Agent name for separate memory stores")] = "agent",
    show_todos: Annotated[bool, typer.Option("--show-todos/--no-todos", help="Show streaming todo list updates")] = True,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose output")] = False,
) -> None:
    """Run a single task without entering interactive mode.

    This command executes a single task and exits, perfect for:
    - Scripting and automation
    - CI/CD pipelines
    - Quick one-off tasks

    Examples:
        deepagents run "Fix the type errors in main.py"
        deepagents run "Run tests and report results" --agent testing
        deepagents run "Analyze code quality" --no-todos
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[bold red]Error:[/bold red] ANTHROPIC_API_KEY environment variable is not set.")
        raise typer.Exit(1)

    # Build agent (same as chat command)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task(description="Initializing agent...", total=None)

        tools = [http_request]
        if tavily_client is not None:
            tools.append(web_search)

        shell_middleware = ShellToolMiddleware(workspace_root=os.getcwd(), execution_policy=HostExecutionPolicy())
        backend = FilesystemBackend()

        agent_dir = Path.home() / ".deepagents" / agent
        agent_dir.mkdir(parents=True, exist_ok=True)
        agent_md = agent_dir / "agent.md"
        if not agent_md.exists():
            source_content = get_default_coding_instructions()
            agent_md.write_text(source_content)

        long_term_backend = FilesystemBackend(root_dir=agent_dir, virtual_mode=True)
        agent_middleware = [AgentMemoryMiddleware(backend=long_term_backend), shell_middleware]
        system_prompt = f"""### Current Working Directory

The filesystem backend is currently operating in: `{Path.cwd()}`"""

        deep_agent = create_deep_agent(
            system_prompt=system_prompt,
            tools=tools,
            memory_backend=backend,
            use_longterm_memory=long_term_backend,
            middleware=agent_middleware,
        ).with_config({"recursion_limit": 1000})

        deep_agent.checkpointer = InMemorySaver()

    # Execute single task
    try:
        execute_task_with_streaming(prompt, deep_agent, agent, show_todos)
    except Exception as e:
        console.print(f"\n[bold red]❌ Error:[/bold red] {e}\n")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command(name="list")
def list_agents() -> None:
    """List all available agents with their details.

    Shows all agents stored in ~/.deepagents/ with their configuration.
    """
    agents_dir = Path.home() / ".deepagents"

    if not agents_dir.exists() or not any(agents_dir.iterdir()):
        console.print("[yellow]No agents found.[/yellow]")
        console.print("[dim]Agents will be created in ~/.deepagents/ when you first use them.[/dim]")
        return

    console.print("\n[bold cyan]Available Agents:[/bold cyan]\n")

    for agent_path in sorted(agents_dir.iterdir()):
        if agent_path.is_dir():
            agent_name = agent_path.name
            agent_md = agent_path / "agent.md"

            if agent_md.exists():
                console.print(f"  [green]•[/green] [bold]{agent_name}[/bold]")
                console.print(f"    [dim]{agent_path}[/dim]")
            else:
                console.print(f"  [yellow]•[/yellow] [bold]{agent_name}[/bold] [dim](incomplete)[/dim]")
                console.print(f"    [dim]{agent_path}[/dim]")

    console.print()


@app.command()
def reset(
    agent: Annotated[str, typer.Option("--agent", "-a", help="Name of agent to reset", prompt=True)],
    source: Annotated[Optional[str], typer.Option("--source", "-s", help="Copy from another agent")] = None,
    force: Annotated[bool, typer.Option("--force", "-f", help="Skip confirmation prompt")] = False,
) -> None:
    """Reset an agent to default or copy from another agent.

    This will remove the agent's memory and reset it to defaults.

    Examples:
        deepagents reset --agent mybot                # Reset to defaults
        deepagents reset --agent mybot --source other # Copy from 'other' agent
        deepagents reset --agent mybot --force        # Skip confirmation
    """
    import shutil

    agents_dir = Path.home() / ".deepagents"
    agent_dir = agents_dir / agent

    if agent_dir.exists() and not force:
        confirm = typer.confirm(f"Are you sure you want to reset agent '{agent}'? This will delete all memory.")
        if not confirm:
            console.print("[yellow]Cancelled.[/yellow]")
            raise typer.Exit(0)

    if source:
        source_dir = agents_dir / source
        source_md = source_dir / "agent.md"

        if not source_md.exists():
            console.print(f"[bold red]Error:[/bold red] Source agent '{source}' not found or has no agent.md")
            raise typer.Exit(1)

        source_content = source_md.read_text()
        action_desc = f"contents of agent '{source}'"
    else:
        source_content = get_default_coding_instructions()
        action_desc = "default"

    if agent_dir.exists():
        shutil.rmtree(agent_dir)
        console.print(f"[yellow]Removed existing agent directory:[/yellow] {agent_dir}")

    agent_dir.mkdir(parents=True, exist_ok=True)
    agent_md = agent_dir / "agent.md"
    agent_md.write_text(source_content)

    console.print(f"[bold green]✓[/bold green] Agent '{agent}' reset to {action_desc}")
    console.print(f"[dim]Location: {agent_dir}[/dim]\n")


@app.command()
def info(
    agent: Annotated[Optional[str], typer.Option("--agent", "-a", help="Agent name to show info for")] = "agent",
) -> None:
    """Show detailed information about an agent.

    Displays the agent's memory location, configuration, and statistics.
    """
    agents_dir = Path.home() / ".deepagents"
    agent_dir = agents_dir / agent

    if not agent_dir.exists():
        console.print(f"[yellow]Agent '{agent}' does not exist yet.[/yellow]")
        console.print(f"[dim]It will be created at: {agent_dir}[/dim]")
        return

    agent_md = agent_dir / "agent.md"
    memories_dir = agent_dir / "memories"

    table = Table(title=f"Agent Info: {agent}", box=box.ROUNDED, show_header=False)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Name", agent)
    table.add_row("Location", str(agent_dir))
    table.add_row("Config File", str(agent_md) if agent_md.exists() else "[red]Not found[/red]")

    if memories_dir.exists():
        memory_count = len(list(memories_dir.rglob("*")))
        table.add_row("Memory Files", str(memory_count))
    else:
        table.add_row("Memory Files", "0")

    console.print()
    console.print(table)
    console.print()

    if agent_md.exists():
        content_preview = agent_md.read_text()[:200]
        if len(agent_md.read_text()) > 200:
            content_preview += "..."

        console.print(Panel(content_preview, title="[cyan]agent.md Preview[/cyan]", border_style="dim", box=box.ROUNDED))
        console.print()


def cli_main() -> None:
    """Main entry point for the Typer CLI."""
    app()


if __name__ == "__main__":
    cli_main()

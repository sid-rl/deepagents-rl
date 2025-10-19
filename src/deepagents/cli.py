#!/usr/bin/env python3
import argparse
import asyncio
import os
import subprocess
import platform
import requests
from typing import Dict, Any, Union, Literal
from pathlib import Path

from tavily import TavilyClient
from deepagents import create_deep_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, Interrupt

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from rich.live import Live
from rich.spinner import Spinner
from rich.prompt import Prompt
from rich import box

import dotenv

dotenv.load_dotenv()

console = Console()

tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY")) if os.environ.get("TAVILY_API_KEY") else None


def http_request(
    url: str,
    method: str = "GET",
    headers: Dict[str, str] = None,
    data: Union[str, Dict] = None,
    params: Dict[str, str] = None,
    timeout: int = 30,
) -> Dict[str, Any]:
    """
    Make HTTP requests to APIs and web services.

    Args:
        url: Target URL
        method: HTTP method (GET, POST, PUT, DELETE, etc.)
        headers: HTTP headers to include
        data: Request body data (string or dict)
        params: URL query parameters
        timeout: Request timeout in seconds

    Returns:
        Dictionary with response data including status, headers, and content
    """
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
        except:
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
            "content": f"Request error: {str(e)}",
            "url": url,
        }
    except Exception as e:
        return {
            "success": False,
            "status_code": 0,
            "headers": {},
            "content": f"Error making request: {str(e)}",
            "url": url,
        }


def web_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Search the web using Tavily for programming-related information."""
    if tavily_client is None:
        return {
            "error": "Tavily API key not configured. Please set TAVILY_API_KEY environment variable.",
            "query": query
        }
    
    try:
        search_docs = tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic,
        )
        return search_docs
    except Exception as e:
        return {
            "error": f"Web search error: {str(e)}",
            "query": query
        }


def get_default_coding_instructions() -> str:
    """Get the default coding agent instructions."""
    return """You are a coding assistant that helps users with software engineering tasks.

# Tone and Style
Be concise and direct. Answer in fewer than 4 lines unless the user asks for detail.
After working on a file, just stop - don't explain what you did unless asked.
Avoid unnecessary introductions or conclusions.

When you run non-trivial bash commands, briefly explain what they do.

## Proactiveness
Take action when asked, but don't surprise users with unrequested actions.
If asked how to approach something, answer first before taking action.

## Following Conventions
- Check existing code for libraries and frameworks before assuming availability
- Mimic existing code style, naming conventions, and patterns
- Never add comments unless asked

## Task Management
Use write_todos for complex multi-step tasks (3+ steps). Mark tasks in_progress before starting, completed immediately after finishing.
For simple 1-2 step tasks, just do them without todos.

## Tools

### execute_bash
Execute shell commands. Always quote paths with spaces.
Examples: `pytest /foo/bar/tests` (good), `cd /foo/bar && pytest tests` (bad)

### File Tools
- read_file: Read file contents (use absolute paths)
- edit_file: Replace exact strings in files (must read first, provide unique old_string)
- write_file: Create or overwrite files
- ls: List directory contents
- glob: Find files by pattern (e.g., "**/*.py")
- grep: Search file contents

Always use absolute paths starting with /.

### web_search
Search for documentation, error solutions, and code examples.

### http_request
Make HTTP requests to APIs (GET, POST, etc.).

## Code References
When referencing code, use format: `file_path:line_number`

## Sub Agents
Use specialized sub-agents for complex one-off tasks:

- **code-reviewer**: Review code quality, security, best practices
- **debugger**: Investigate errors and bugs
- **test-generator**: Create comprehensive test suites

Example: `task(description="Debug the login function throwing TypeError", subagent_type="debugger")`
"""


def get_coding_instructions(agent_name: str) -> str:
    """Get the coding agent instructions from file or create default."""
    agent_dir = Path.home() / ".deepagents" / agent_name
    agent_prompt_file = agent_dir / "agent.md"
    
    if agent_prompt_file.exists():
        return agent_prompt_file.read_text()
    else:
        agent_dir.mkdir(parents=True, exist_ok=True)
        default_prompt = get_default_coding_instructions()
        agent_prompt_file.write_text(default_prompt)
        return default_prompt


config = {"recursion_limit": 1000}

# Constants for display truncation
MAX_ARG_LENGTH = 200
MAX_RESULT_LENGTH = 200


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


def display_tool_call(tool_name: str, tool_input: dict):
    """Display a tool call with arguments, truncating long values."""
    
    tool_icons = {
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
    
    icon = tool_icons.get(tool_name, "🔧")
    args_str = format_tool_args(tool_input)
    
    # Display: icon tool_name(args)
    if args_str:
        console.print(f"[dim]{icon} {tool_name}({args_str})[/dim]")
    else:
        console.print(f"[dim]{icon} {tool_name}()[/dim]")


def display_text_content(text: str):
    """Display text content with markdown rendering."""
    if text.strip():
        if "```" in text or "#" in text or "**" in text:
            md = Markdown(text)
            console.print(md)
        else:
            console.print(text, style="white")


def extract_and_display_content(message_content):
    """Extract content from agent messages and display with rich formatting."""
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
                # Skip tool_result blocks - they're just noise
                elif block.get("type") == "tool_result":
                    pass  # Don't display tool results


def execute_task(user_input: str, agent, agent_name: str):
    """Execute any task by passing it directly to the AI agent."""
    console.print()
    
    config = {"configurable": {"thread_id": "main", "agent_name": agent_name}}
    
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
                result = chunk
                break
            
            # Normal message processing
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
                    # Show tool calls with truncated args
                    if message_role != "tool":
                        if isinstance(message_content, list):
                            for block in message_content:
                                if isinstance(block, dict) and block.get("type") == "tool_use":
                                    tool_name = block.get("name", "unknown_tool")
                                    tool_input = block.get("input", {})
                                    
                                    tool_icons = {
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
                                    
                                    icon = tool_icons.get(tool_name, "🔧")
                                    args_str = format_tool_args(tool_input)
                                    
                                    if args_str:
                                        console.print(f"[dim]{icon} {tool_name}({args_str})[/dim]")
                                    else:
                                        console.print(f"[dim]{icon} {tool_name}()[/dim]")
                    
                    # Show tool results with truncation
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
    
    console.print()


async def simple_cli(agent, agent_name: str):
    """Main CLI loop."""
    console.print()
    console.print(Panel.fit(
        "[bold cyan]DeepAgents[/bold cyan] [dim]|[/dim] AI Coding Assistant",
        border_style="cyan",
        box=box.DOUBLE
    ))
    console.print("[dim]Type 'quit' to exit, 'help' for examples[/dim]")
    console.print()

    while True:
        try:
            console.print("[bold green]❯[/bold green] ", end="")
            user_input = input().strip()
        except EOFError:
            break

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit", "q"]:
            console.print("\n[bold cyan]👋 Goodbye![/bold cyan]\n")
            break

        elif user_input.lower() == "help":
            help_text = """
[bold cyan]Examples:[/bold cyan]

  [yellow]•[/yellow] Create a function to calculate fibonacci numbers
  [yellow]•[/yellow] Debug this sorting code: [paste code]
  [yellow]•[/yellow] Review my Flask app for security issues
  [yellow]•[/yellow] Generate tests for this calculator class
  [yellow]•[/yellow] Search for Python async best practices
            """
            console.print(Panel(help_text.strip(), border_style="cyan", box=box.ROUNDED))
            console.print()
            continue

        else:
            execute_task(user_input, agent, agent_name)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="DeepAgents - AI Coding Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Custom type to validate agent name
    def validate_agent_name(value):
        if value == "agent":
            raise argparse.ArgumentTypeError(
                "Cannot explicitly pass 'agent' as the agent name. Use a different name or omit --agent to use the default."
            )
        return value
    
    parser.add_argument(
        "--agent",
        type=validate_agent_name,
        default=None,
        help="Agent identifier for separate memory stores (default: agent). Cannot explicitly pass 'agent'.",
    )
    
    return parser.parse_args()


async def main(agent_name: str):
    """Main entry point."""
    # Create agent
    agent = create_deep_agent(
        tools=[http_request, web_search],
        system_prompt=get_coding_instructions(agent_name),
        use_local_filesystem=True,
    ).with_config(config)
    
    agent.checkpointer = InMemorySaver()
    
    try:
        await simple_cli(agent, agent_name)
    except KeyboardInterrupt:
        console.print("\n\n[bold cyan]👋 Goodbye![/bold cyan]\n")
    except Exception as e:
        console.print(f"\n[bold red]❌ Error:[/bold red] {e}\n")


def cli_main():
    """Entry point for console script."""
    args = parse_args()
    asyncio.run(main(args.agent or "agent"))


if __name__ == "__main__":
    cli_main()

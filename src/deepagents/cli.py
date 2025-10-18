#!/usr/bin/env python3
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

import dotenv

dotenv.load_dotenv()

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


def get_coding_instructions() -> str:
    """Get the coding agent instructions."""
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


config = {"recursion_limit": 1000}

agent = create_deep_agent(
    tools=[http_request, web_search],
    system_prompt=get_coding_instructions(),
    use_local_filesystem=True,
).with_config(config)

agent.checkpointer = InMemorySaver()


def extract_content_with_tools(message_content) -> str:
    """Extract content from agent messages, including tool calls for transparency."""
    if isinstance(message_content, str):
        return message_content

    if isinstance(message_content, list):
        parts = []
        for block in message_content:
            if isinstance(block, dict):
                if block.get("type") == "text" and "text" in block:
                    parts.append(block["text"])
                elif block.get("type") == "tool_use":
                    tool_name = block.get("name", "unknown_tool")
                    parts.append(f"\n🔧 Using tool: {tool_name}")

                    if "input" in block:
                        tool_input = block["input"]
                        if isinstance(tool_input, dict):
                            for key, value in tool_input.items():
                                if key in ["file_path", "content", "old_string", "new_string"]:
                                    if len(str(value)) > 100:
                                        parts.append(f"  • {key}: {str(value)[:50]}...")
                                    else:
                                        parts.append(f"  • {key}: {value}")
                    parts.append("")

        return "\n".join(parts).strip() if parts else ""

    if hasattr(message_content, "__dict__"):
        return ""
    return str(message_content)


def execute_task(user_input: str):
    """Execute any task by passing it directly to the AI agent."""
    print(f"\n🤖 Working on: {user_input[:60]}{'...' if len(user_input) > 60 else ''}\n")

    for _, chunk in agent.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        stream_mode="updates",
        subgraphs=True,
        config={"thread_id": "main"},
        durability="exit",
    ):
        chunk = list(chunk.values())[0]
        if chunk is not None and "messages" in chunk and chunk["messages"]:
            last_message = chunk["messages"][-1]

            message_content = None
            message_role = getattr(last_message, "role", None)
            if isinstance(message_role, dict):
                message_role = last_message.get("role", "unknown")

            if hasattr(last_message, "content"):
                message_content = last_message.content
            elif isinstance(last_message, dict) and "content" in last_message:
                message_content = last_message["content"]

            if message_content:
                content = extract_content_with_tools(message_content)

                if content.strip():
                    if message_role == "tool":
                        print(f"🔧 {content}\n")
                    else:
                        print(f"{content}\n")


async def simple_cli():
    """Main CLI loop."""
    print("🤖 Software Engineering CLI")
    print("Type 'quit' to exit, 'help' for examples\n")

    while True:
        user_input = input(">>> ").strip()

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit", "q"]:
            print("👋 Goodbye!")
            break

        elif user_input.lower() == "help":
            print("""
Examples:
• "Create a function to calculate fibonacci numbers"
• "Debug this sorting code: [paste code]"
• "Review my Flask app for security issues"
• "Generate tests for this calculator class"
""")
            continue

        else:
            execute_task(user_input)


async def main():
    """Main entry point."""
    try:
        await simple_cli()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")


def cli_main():
    """Entry point for console script."""
    asyncio.run(main())


if __name__ == "__main__":
    cli_main()

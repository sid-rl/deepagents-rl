"""CLI agent that handles natural language commands for docs reviewing."""

import os
from pathlib import Path
from typing import Any, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

from deepagents import create_deep_agent
from docs_reviewer.markdown_parser import extract_code_snippets, filter_executable_snippets, categorize_snippets
from docs_reviewer.markdown_writer import write_corrected_markdown, write_review_report
from docs_reviewer.agent import DocsReviewerAgent


class DocsReviewerCLIAgent:
    """Simple agent for natural language docs reviewing."""

    def __init__(self):
        """Initialize the CLI agent."""
        self.conversation_history: list = []
        self.current_working_directory = Path.cwd()

        # Get API key
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")

        # Initialize LLM - simple, no config needed
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-5-20250929",
            temperature=0.1,
            api_key=api_key,
        )

        # Create tools and agent
        self.tools = self._create_cli_tools()
        self.agent = create_deep_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=self._get_system_prompt(),
        )

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the CLI agent."""
        return f"""You are a documentation reviewer that validates code snippets in markdown files.

Working directory: {self.current_working_directory}

When reviewing files:
1. Use list_snippets to show what code was found
2. Use review_markdown_file to validate and generate corrections
3. Report results concisely

Be direct and concise. Use relative paths from the working directory.
"""

    def _create_cli_tools(self) -> list:
        """Create tools for the CLI agent."""
        tools = []

        @tool
        def list_snippets(markdown_file: str) -> dict[str, Any]:
            """
            List all code snippets found in a markdown file without executing them.

            Args:
                markdown_file: Path to the markdown file (relative or absolute)

            Returns:
                Dictionary with snippet information
            """
            try:
                file_path = Path(markdown_file)
                if not file_path.is_absolute():
                    file_path = self.current_working_directory / file_path

                if not file_path.exists():
                    return {"error": f"File not found: {file_path}"}

                snippets = extract_code_snippets(file_path)
                executable = filter_executable_snippets(snippets)
                categories = categorize_snippets(snippets)

                return {
                    "file": str(file_path),
                    "total_snippets": len(snippets),
                    "executable_snippets": len(executable),
                    "categories": {lang: len(snips) for lang, snips in categories.items()},
                    "snippets": [
                        {
                            "language": s["language"],
                            "lines": f"{s['start_line']}-{s['end_line']}",
                            "length": len(s["code"]),
                            "preview": s["code"][:100] + "..." if len(s["code"]) > 100 else s["code"],
                        }
                        for s in snippets
                    ],
                }
            except Exception as e:
                return {"error": str(e)}

        @tool
        def review_markdown_file(
            markdown_file: str, output_file: Optional[str] = None
        ) -> dict[str, Any]:
            """
            Review a markdown file and validate all code snippets, generating corrected output.

            Args:
                markdown_file: Path to the markdown file to review
                output_file: Optional path for the corrected markdown (default: <input>_corrected.md)

            Returns:
                Dictionary with review results and output file path
            """
            try:
                file_path = Path(markdown_file)
                if not file_path.is_absolute():
                    file_path = self.current_working_directory / file_path

                if not file_path.exists():
                    return {"error": f"File not found: {file_path}"}

                # Set output path
                if output_file:
                    output_path = Path(output_file)
                    if not output_path.is_absolute():
                        output_path = self.current_working_directory / output_path
                else:
                    output_path = file_path.parent / f"{file_path.stem}_corrected.md"

                # Extract snippets
                snippets = extract_code_snippets(file_path)
                if not snippets:
                    return {"error": "No code snippets found in file"}

                # Review with the docs reviewer agent
                reviewer = DocsReviewerAgent()
                results = reviewer.review_snippets(file_path, snippets)

                # Write corrected markdown
                write_corrected_markdown(file_path, output_path, snippets, results)

                # Generate report
                report_path = file_path.parent / f"{file_path.stem}_review_report.md"
                write_review_report(report_path, file_path, snippets, results)

                # Calculate stats
                total = len(results)
                successful = sum(1 for r in results if r["success"])
                failed = total - successful

                return {
                    "success": True,
                    "original_file": str(file_path),
                    "corrected_file": str(output_path),
                    "report_file": str(report_path),
                    "total_snippets": total,
                    "successful": successful,
                    "failed": failed,
                    "success_rate": f"{(successful / total * 100):.1f}%" if total > 0 else "0%",
                }
            except Exception as e:
                return {"error": str(e)}

        @tool
        def change_directory(directory: str) -> dict[str, Any]:
            """
            Change the current working directory.

            Args:
                directory: Path to the new working directory

            Returns:
                Dictionary with the new working directory
            """
            try:
                new_dir = Path(directory)
                if not new_dir.is_absolute():
                    new_dir = self.current_working_directory / new_dir

                if not new_dir.exists():
                    return {"error": f"Directory not found: {new_dir}"}

                if not new_dir.is_dir():
                    return {"error": f"Not a directory: {new_dir}"}

                self.current_working_directory = new_dir.resolve()
                return {
                    "success": True,
                    "working_directory": str(self.current_working_directory),
                }
            except Exception as e:
                return {"error": str(e)}

        @tool
        def get_working_directory() -> str:
            """
            Get the current working directory.

            Returns:
                The current working directory path
            """
            return str(self.current_working_directory)

        @tool
        def find_markdown_files(directory: Optional[str] = None) -> dict[str, Any]:
            """
            Find all markdown files in a directory.

            Args:
                directory: Directory to search (default: current working directory)

            Returns:
                Dictionary with list of markdown files found
            """
            try:
                search_dir = Path(directory) if directory else self.current_working_directory
                if not search_dir.is_absolute():
                    search_dir = self.current_working_directory / search_dir

                if not search_dir.exists():
                    return {"error": f"Directory not found: {search_dir}"}

                md_files = list(search_dir.rglob("*.md"))
                return {
                    "directory": str(search_dir),
                    "count": len(md_files),
                    "files": [str(f.relative_to(search_dir)) for f in md_files],
                }
            except Exception as e:
                return {"error": str(e)}

        tools.append(list_snippets)
        tools.append(review_markdown_file)
        tools.append(change_directory)
        tools.append(get_working_directory)
        tools.append(find_markdown_files)

        return tools

    def process_message(self, message: str, console) -> str:
        """
        Process a user message with streaming output.

        Args:
            message: User's message
            console: Rich console for output

        Returns:
            Agent's response as a string
        """
        import json
        from rich.syntax import Syntax

        # Add to conversation history
        self.conversation_history.append(HumanMessage(content=message))

        # Stream the agent's response
        try:
            final_response = ""
            seen_tool_calls = set()
            shown_todos = set()
            last_ai_message = None

            # Use only "updates" stream mode for smoother, less jumpy output
            for stream_item in self.agent.stream(
                {"messages": self.conversation_history},
                stream_mode="updates"
            ):
                # stream_item is a dict mapping node_name -> node_output
                if not isinstance(stream_item, dict):
                    continue

                # Process each node's update
                for node_name, node_data in stream_item.items():
                    if not isinstance(node_data, dict):
                        continue

                    # Extract messages from the update
                    if "messages" in node_data:
                        messages = node_data["messages"]
                        if not isinstance(messages, list):
                            messages = [messages]

                        for msg in messages:
                            # Handle AI messages
                            if hasattr(msg, 'type') and msg.type == 'ai':
                                last_ai_message = msg

                                # Check if this AI message has text content (not just tool calls)
                                has_content = False
                                if hasattr(msg, 'content'):
                                    content = msg.content
                                    if isinstance(content, str) and content.strip():
                                        has_content = True
                                        # Print the AI's text content
                                        console.print(content, end="")
                                        final_response += content
                                    elif isinstance(content, list):
                                        # Handle Anthropic content blocks format
                                        for block in content:
                                            if isinstance(block, dict) and block.get('type') == 'text':
                                                text = block.get('text', '')
                                                if text:
                                                    has_content = True
                                                    console.print(text, end="")
                                                    final_response += text

                                # Check for tool calls
                                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                    for tool_call in msg.tool_calls:
                                        tool_id = tool_call.get('id', '') or str(tool_call)

                                        if tool_id not in seen_tool_calls:
                                            seen_tool_calls.add(tool_id)

                                            # Add newline if we just printed content
                                            if has_content and final_response and not final_response.endswith('\n'):
                                                console.print()

                                            tool_name = tool_call.get('name', 'unknown')
                                            tool_args = tool_call.get('args', {})

                                            # Format tool call nicely - but only show args if they're concise
                                            args_str = json.dumps(tool_args)
                                            if len(args_str) > 150:
                                                # Too verbose, just show the tool name
                                                console.print(f"\n[yellow]→ {tool_name}[/yellow]")
                                            else:
                                                # Show args nicely if they're reasonable
                                                console.print(f"\n[yellow]→ {tool_name}[/yellow]")
                                                if tool_args:
                                                    for key, value in tool_args.items():
                                                        if isinstance(value, str) and len(value) > 60:
                                                            console.print(f"  [dim]{key}:[/dim] {value[:60]}...")
                                                        else:
                                                            console.print(f"  [dim]{key}:[/dim] {value}")

                            # Handle tool response messages
                            elif hasattr(msg, 'type') and msg.type == 'tool':
                                tool_name = getattr(msg, 'name', 'unknown')
                                tool_content = getattr(msg, 'content', '')

                                # Parse and display tool results
                                try:
                                    result = json.loads(tool_content) if isinstance(tool_content, str) else tool_content

                                    # Check for errors first
                                    if isinstance(result, dict) and "error" in result:
                                        console.print(f"[red]  ✗ Error: {result['error']}[/red]\n")
                                        continue

                                    # Special formatting for list_snippets
                                    if tool_name == "list_snippets" and isinstance(result, dict):
                                        total = result.get("total_snippets", 0)
                                        console.print(f"[green]  ✓ Found {total} snippet{'s' if total != 1 else ''}[/green]")

                                        snippets = result.get("snippets", [])
                                        if snippets:
                                            for i, snip in enumerate(snippets, 1):
                                                lang = snip.get("language", "unknown")
                                                lines = snip.get("lines", "?")
                                                console.print(f"    {i}. [cyan]{lang}[/cyan] (lines {lines})")
                                        console.print()

                                    # Special formatting for review_markdown_file
                                    elif tool_name == "review_markdown_file" and isinstance(result, dict):
                                        if result.get("success"):
                                            total = result.get("total_snippets", 0)
                                            successful = result.get("successful", 0)
                                            failed = result.get("failed", 0)

                                            console.print(f"[green]  ✓ Review complete[/green]")
                                            console.print(f"    {successful}/{total} passed")

                                            if failed > 0:
                                                console.print(f"    [yellow]{failed} failed[/yellow]")

                                            corrected = result.get("corrected_file", "")
                                            if corrected:
                                                console.print(f"    [dim]→ {corrected}[/dim]")
                                            console.print()

                                    # Special formatting for find_markdown_files
                                    elif tool_name == "find_markdown_files" and isinstance(result, dict):
                                        count = result.get("count", 0)
                                        console.print(f"[green]  ✓ Found {count} markdown file{'s' if count != 1 else ''}[/green]")
                                        files = result.get("files", [])
                                        if files:
                                            for f in files[:10]:  # Show first 10
                                                console.print(f"    • {f}")
                                            if len(files) > 10:
                                                console.print(f"    [dim]... and {len(files) - 10} more[/dim]")
                                        console.print()

                                    # Generic success for other tools
                                    elif isinstance(result, dict) and result.get("success"):
                                        console.print(f"[green]  ✓ {tool_name}[/green]\n")

                                    else:
                                        # Default: just show completion
                                        console.print(f"[dim]  ✓[/dim]\n")

                                except (json.JSONDecodeError, TypeError):
                                    # Non-JSON response, just show completion
                                    console.print(f"[dim]  ✓[/dim]\n")

                    # Handle todos from state
                    if "todos" in node_data:
                        todos = node_data["todos"]
                        if todos:
                            todos_hash = str([(t.get("content"), t.get("status")) for t in todos])
                            if todos_hash not in shown_todos:
                                shown_todos.add(todos_hash)

                                if final_response and not final_response.endswith('\n'):
                                    console.print()

                                console.print("\n[dim]Tasks:[/dim]")
                                for todo in todos:
                                    status = todo.get("status", "pending")
                                    status_emoji = {"pending": "○", "in_progress": "◐", "completed": "●"}.get(status, "•")
                                    todo_content = todo.get("content", "Unknown task")

                                    if status == "completed":
                                        console.print(f"  [dim]{status_emoji} {todo_content}[/dim]")
                                    elif status == "in_progress":
                                        active_form = todo.get("activeForm", todo_content)
                                        console.print(f"  {status_emoji} {active_form}")
                                    else:
                                        console.print(f"  [dim]{status_emoji} {todo_content}[/dim]")
                                console.print()

            # Ensure final newline
            if final_response and not final_response.endswith('\n'):
                console.print()

            # Add to conversation history
            if final_response:
                self.conversation_history.append(AIMessage(content=final_response))

            return final_response

        except Exception as e:
            error_message = f"Error: {str(e)}"
            console.print(f"\n[red]{error_message}[/red]")
            self.conversation_history.append(AIMessage(content=error_message))
            return error_message

    def reset_conversation(self) -> None:
        """Reset the conversation history."""
        self.conversation_history = []

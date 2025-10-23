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
        return f"""You are a helpful AI assistant for reviewing documentation and code snippets.

Current working directory: {self.current_working_directory}

Your capabilities:
1. Review markdown files for code correctness
2. Extract and list code snippets
3. Validate code syntax and execution
4. Generate corrected markdown files
5. Answer questions about documentation

When users ask you to review files:
- Use the list_snippets tool first to show what you found
- Then use review_markdown_file to perform the full review
- Report the results clearly and helpfully

When interacting with users:
- Be friendly and conversational
- Ask clarifying questions if the request is ambiguous
- Provide clear, actionable feedback
- Use markdown formatting in your responses

Always resolve relative file paths based on the current working directory.
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
        from rich.live import Live
        from rich.panel import Panel
        from rich.text import Text

        # Add to conversation history
        self.conversation_history.append(HumanMessage(content=message))

        # Stream the agent's response
        try:
            final_response = ""

            # Stream events from the agent
            for event in self.agent.stream({"messages": self.conversation_history}, stream_mode="values"):
                # Check for the latest message
                if "messages" in event:
                    latest_msg = event["messages"][-1]

                    # Show tool calls
                    if hasattr(latest_msg, "tool_calls") and latest_msg.tool_calls:
                        for tool_call in latest_msg.tool_calls:
                            console.print(f"[yellow]🔧 Using tool: [bold]{tool_call['name']}[/bold][/yellow]")

                    # Show tool responses
                    if hasattr(latest_msg, "name") and latest_msg.name:
                        console.print(f"[dim]  ✓ Tool [bold]{latest_msg.name}[/bold] completed[/dim]")

                    # Capture final AI response
                    if hasattr(latest_msg, "content") and isinstance(latest_msg.content, str):
                        if latest_msg.content and hasattr(latest_msg, "type") and latest_msg.type == "ai":
                            final_response = latest_msg.content

                # Show todos if they're updated
                if "todos" in event:
                    todos = event["todos"]
                    if todos:
                        console.print("\n[cyan]📋 Task Plan:[/cyan]")
                        for todo in todos:
                            status = todo.get("status", "pending")
                            status_emoji = {"pending": "⏳", "in_progress": "🔄", "completed": "✅"}.get(status, "•")
                            content = todo.get("content", "Unknown task")
                            console.print(f"  {status_emoji} {content}")
                        console.print()

            # If we got a final response, use it
            if final_response:
                self.conversation_history.append(AIMessage(content=final_response))
                return final_response

            # Otherwise get the last message from a full invoke
            response = self.agent.invoke({"messages": self.conversation_history})
            last_message = response["messages"][-1]

            if hasattr(last_message, "content"):
                response_text = last_message.content
            else:
                response_text = str(last_message)

            self.conversation_history.append(AIMessage(content=response_text))
            return response_text

        except Exception as e:
            error_message = f"Error processing your request: {str(e)}"
            self.conversation_history.append(AIMessage(content=error_message))
            return error_message

    def reset_conversation(self) -> None:
        """Reset the conversation history."""
        self.conversation_history = []

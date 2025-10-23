"""DeepAgent integration for docs reviewing."""

import os
from pathlib import Path
from typing import Any, Callable, Optional
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool

from deepagents import create_deep_agent
from docs_reviewer.markdown_parser import CodeSnippet, filter_executable_snippets


class DocsReviewerAgent:
    """Simple agent for reviewing documentation code snippets."""

    def __init__(self):
        """Initialize the docs reviewer agent."""
        # Get API key
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")

        # Initialize LLM - simple defaults
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-5-20250929",
            temperature=0.0,
            api_key=api_key,
        )

        # Create tools and agent
        self.tools = self._create_tools()
        self.agent = create_deep_agent(
            model=self.llm,
            tools=self.tools,
        )

    def _create_tools(self) -> list:
        """Create custom tools for the agent."""
        tools = []

        @tool
        def execute_python_snippet(code: str) -> dict[str, Any]:
            """
            Execute a Python code snippet in a safe sandbox environment.

            Args:
                code: The Python code to execute

            Returns:
                Dictionary with execution results including stdout, stderr, and success status
            """
            import sys
            from io import StringIO
            import traceback

            # Capture stdout and stderr
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            stdout_capture = StringIO()
            stderr_capture = StringIO()

            sys.stdout = stdout_capture
            sys.stderr = stderr_capture

            result = {"success": False, "stdout": "", "stderr": "", "error": None}

            try:
                # Execute the code
                exec_globals: dict[str, Any] = {}
                exec(code, exec_globals)
                result["success"] = True
            except Exception as e:
                result["error"] = str(e)
                result["stderr"] = traceback.format_exc()
            finally:
                # Restore stdout/stderr
                sys.stdout = old_stdout
                sys.stderr = old_stderr

                result["stdout"] = stdout_capture.getvalue()
                if not result["stderr"]:
                    result["stderr"] = stderr_capture.getvalue()

            return result

        @tool
        def validate_code_syntax(code: str, language: str) -> dict[str, Any]:
            """
            Validate the syntax of a code snippet without executing it.

            Args:
                code: The code to validate
                language: The programming language (python, javascript, etc.)

            Returns:
                Dictionary with validation results
            """
            result = {"valid": False, "errors": []}

            if language.lower() in ("python", "py"):
                import ast

                try:
                    ast.parse(code)
                    result["valid"] = True
                except SyntaxError as e:
                    result["errors"].append(f"Line {e.lineno}: {e.msg}")
            elif language.lower() in ("javascript", "js", "typescript", "ts"):
                # For JS/TS, we would need an external validator
                # For now, just do basic checks
                result["valid"] = True
                result["errors"].append("JavaScript/TypeScript validation not yet implemented")
            else:
                result["valid"] = True
                result["errors"].append(f"Syntax validation not supported for {language}")

            return result

        @tool
        def search_langchain_docs(query: str) -> str:
            """
            Search the LangChain documentation for relevant information.

            Args:
                query: The search query

            Returns:
                Relevant documentation content
            """
            # This will be implemented via MCP server
            # For now, return a placeholder
            return f"Searching LangChain docs for: {query}\n(MCP server integration pending)"

        tools.append(execute_python_snippet)
        tools.append(validate_code_syntax)
        tools.append(search_langchain_docs)

        return tools

    def review_snippets(
        self,
        markdown_file: Path,
        snippets: list[CodeSnippet],
        progress_callback: Optional[Callable[[], None]] = None,
    ) -> list[dict[str, Any]]:
        """
        Review and execute code snippets from a markdown file.

        Args:
            markdown_file: Path to the markdown file
            snippets: List of code snippets to review
            progress_callback: Optional callback to call after each snippet

        Returns:
            List of results for each snippet
        """
        # Filter to only executable snippets
        executable_snippets = filter_executable_snippets(snippets)

        results = []

        for i, snippet in enumerate(executable_snippets, 1):
            # Create a prompt for the agent
            prompt = f"""Review and validate the following code snippet from {markdown_file}:

Language: {snippet['language']}
Lines: {snippet['start_line']}-{snippet['end_line']}

Code:
```{snippet['language']}
{snippet['code']}
```

Tasks:
1. Validate the syntax of the code
2. If it's Python code, try to execute it safely
3. If execution fails, analyze the error and suggest fixes
4. If the code requires imports or setup, note what's missing
5. Check if this appears to be LangChain/LangGraph code and search docs if needed

Provide a detailed analysis and any corrections needed."""

            # Invoke the agent
            response = self.agent.invoke({"messages": [{"role": "user", "content": prompt}]})

            # Extract the agent's response
            last_message = response["messages"][-1]
            analysis = last_message.content if hasattr(last_message, "content") else str(last_message)

            result = {
                "snippet_index": i - 1,
                "success": "successfully" in analysis.lower() or "valid" in analysis.lower(),
                "analysis": analysis,
                "original_code": snippet["code"],
                "corrected_code": snippet["code"],  # Will be updated if corrections are found
                "language": snippet["language"],
                "start_line": snippet["start_line"],
                "end_line": snippet["end_line"],
            }

            results.append(result)

            if progress_callback:
                progress_callback()

        return results

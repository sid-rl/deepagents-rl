#!/usr/bin/env python3
"""Documentation Verifier Agent

This script verifies technical documentation by extracting Python code snippets
and creating executable scripts to test that the documentation is correct.
"""

import argparse
from pathlib import Path

from deepagents import create_deep_agent

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

You have access to standard file operations (read, write, edit) and bash commands to execute scripts and install dependencies if needed.
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

    # Create the agent with planning middleware
    agent = create_deep_agent(
        system_prompt=DOC_VERIFIER_PROMPT,
        interrupt_on={"shell": True},
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

"""Conversational CLI entrypoint powered by DeepAgent."""

import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from dotenv import load_dotenv

from docs_reviewer.cli_agent import DocsReviewerCLIAgent
from docs_reviewer.config import init_config, load_config

# Load environment variables from .env file if it exists
load_dotenv()

app = typer.Typer(
    name="docs-reviewer",
    help="AI-powered docs reviewer - just tell me what you want to do!",
    add_completion=False,
)
console = Console()


def check_api_key() -> bool:
    """Check if Anthropic API key is configured."""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def show_welcome():
    """Display minimal welcome message."""
    console.print("[dim]Docs Reviewer ready. Type your request or 'exit' to quit.[/dim]")


def show_setup_instructions():
    """Show setup instructions if not configured."""
    console.print("\n[yellow]⚠️  Configuration needed![/yellow]\n")
    console.print("To get started:")
    console.print("1. Set your Anthropic API key:")
    console.print("   [cyan]export ANTHROPIC_API_KEY='your-api-key'[/cyan]")
    console.print("\n2. (Optional) Run: [cyan]docs-reviewer init[/cyan] to create a config file")
    console.print("\n3. Start the CLI: [cyan]docs-reviewer[/cyan]\n")


@app.command()
def init() -> None:
    """
    Initialize configuration file with default settings (OPTIONAL).

    Note: Configuration is optional! The CLI works fine with just environment variables.
    """
    config_path = Path.cwd() / "docs_reviewer_config.yaml"

    if config_path.exists():
        overwrite = typer.confirm(f"Config file {config_path} already exists. Overwrite?")
        if not overwrite:
            console.print("[yellow]Initialization cancelled.[/yellow]")
            raise typer.Exit()

    init_config(config_path)
    console.print(f"[green]✓ Created configuration file at {config_path}[/green]")
    console.print("\n[bold]Next steps:[/bold]")
    console.print("1. Edit the config file to add your API key (or use ANTHROPIC_API_KEY env var)")
    console.print("2. Run: [cyan]docs-reviewer[/cyan] to start the interactive CLI")
    console.print("\n[dim]Note: The config file is optional - the CLI works fine with just env vars![/dim]")


@app.command()
def chat(
    message: Optional[str] = typer.Option(
        None,
        "--message",
        "-m",
        help="Single message to process (non-interactive mode)",
    ),
) -> None:
    """Start interactive chat session with the docs reviewer agent."""
    # Check API key
    if not check_api_key():
        show_setup_instructions()
        raise typer.Exit(1)

    # Show minimal welcome message (only in interactive mode)
    if not message:
        show_welcome()
        console.print()

    # Initialize the CLI agent (simple - just needs API key in env)
    try:
        agent = DocsReviewerCLIAgent()
    except Exception as e:
        console.print(f"[red]Error initializing agent: {e}[/red]")
        raise typer.Exit(1)

    # Single message mode
    if message:
        console.print("\n[bold green]Docs Reviewer[/bold green]")
        response = agent.process_message(message, console)
        console.print()
        return

    # Interactive chat loop
    conversation_active = True
    while conversation_active:
        try:
            # Get user input
            user_input = Prompt.ask("\n[bold blue]You[/bold blue]")

            if user_input.lower() in ["exit", "quit", "bye", "goodbye"]:
                console.print()
                conversation_active = False
                continue

            if not user_input.strip():
                continue

            # Process with agent - show streaming progress
            console.print("\n[bold green]Docs Reviewer[/bold green]")
            response = agent.process_message(user_input, console)
            console.print()

        except KeyboardInterrupt:
            console.print("\n")
            conversation_active = False
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-v", help="Show version and exit"),
    message: Optional[str] = typer.Option(None, "--message", "-m", help="Single message mode"),
) -> None:
    """
    Docs Reviewer - AI-powered documentation code reviewer.

    Just run 'docs-reviewer' to start chatting!
    """
    if version:
        console.print("docs-reviewer version 0.1.0")
        raise typer.Exit()

    # If no subcommand was invoked, start chat by default
    if ctx.invoked_subcommand is None:
        ctx.invoke(chat, message=message)


if __name__ == "__main__":
    app()

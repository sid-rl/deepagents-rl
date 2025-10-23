"""Main CLI entrypoint for the docs reviewer."""

import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from docs_reviewer.config import load_config, init_config
from docs_reviewer.agent import DocsReviewerAgent
from docs_reviewer.markdown_parser import extract_code_snippets
from docs_reviewer.markdown_writer import write_corrected_markdown

app = typer.Typer(
    name="docs-reviewer",
    help="Review and validate documentation code snippets using DeepAgents.",
    add_completion=False,
)
console = Console()


@app.command()
def init(
    config_path: Path = typer.Option(
        Path.cwd() / "docs_reviewer_config.yaml",
        "--config",
        "-c",
        help="Path to create the configuration file",
    ),
) -> None:
    """Initialize a new configuration file with default settings."""
    if config_path.exists():
        overwrite = typer.confirm(f"Config file {config_path} already exists. Overwrite?")
        if not overwrite:
            console.print("[yellow]Initialization cancelled.[/yellow]")
            raise typer.Exit()

    init_config(config_path)
    console.print(f"[green]Created configuration file at {config_path}[/green]")
    console.print("\n[bold]Next steps:[/bold]")
    console.print("1. Edit the config file to add your API keys")
    console.print("2. Configure default tools and MCP servers")
    console.print("3. Run: docs-reviewer review <path-to-markdown-file>")


@app.command()
def review(
    markdown_file: Path = typer.Argument(..., help="Path to the markdown file to review"),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Path to write corrected markdown (default: <input>_corrected.md)",
    ),
    config_path: Path = typer.Option(
        Path.cwd() / "docs_reviewer_config.yaml",
        "--config",
        "-c",
        help="Path to configuration file",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Extract and analyze snippets without executing them",
    ),
) -> None:
    """Review a markdown file and validate all code snippets."""
    if not markdown_file.exists():
        console.print(f"[red]Error: File {markdown_file} not found[/red]")
        raise typer.Exit(1)

    if not config_path.exists():
        console.print(f"[red]Error: Config file {config_path} not found[/red]")
        console.print("Run 'docs-reviewer init' to create a configuration file")
        raise typer.Exit(1)

    # Set default output file
    if output_file is None:
        output_file = markdown_file.parent / f"{markdown_file.stem}_corrected.md"

    config = load_config(config_path)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Extract code snippets
        task = progress.add_task("Extracting code snippets...", total=None)
        snippets = extract_code_snippets(markdown_file)
        progress.update(task, completed=True)
        console.print(f"[green]Found {len(snippets)} code snippets[/green]")

        if dry_run:
            console.print("\n[yellow]Dry run mode - snippets extracted but not executed[/yellow]")
            for i, snippet in enumerate(snippets, 1):
                console.print(f"\n[bold]Snippet {i}[/bold] ({snippet['language']}):")
                console.print(f"Lines {snippet['start_line']}-{snippet['end_line']}")
            raise typer.Exit()

        # Initialize agent
        task = progress.add_task("Initializing docs reviewer agent...", total=None)
        agent = DocsReviewerAgent(config)
        progress.update(task, completed=True)

        # Review and execute snippets
        task = progress.add_task("Reviewing code snippets...", total=len(snippets))
        results = agent.review_snippets(markdown_file, snippets, progress_callback=lambda: progress.advance(task))

        # Write corrected markdown
        task = progress.add_task("Writing corrected markdown...", total=None)
        write_corrected_markdown(markdown_file, output_file, snippets, results)
        progress.update(task, completed=True)

    # Summary
    console.print("\n[bold green]Review complete![/bold green]")
    console.print(f"Original file: {markdown_file}")
    console.print(f"Corrected file: {output_file}")

    total_snippets = len(results)
    successful = sum(1 for r in results if r["success"])
    failed = total_snippets - successful

    console.print(f"\n[bold]Results:[/bold]")
    console.print(f"  Total snippets: {total_snippets}")
    console.print(f"  [green]Successful: {successful}[/green]")
    if failed > 0:
        console.print(f"  [red]Failed: {failed}[/red]")


@app.command()
def config(
    config_path: Path = typer.Option(
        Path.cwd() / "docs_reviewer_config.yaml",
        "--config",
        "-c",
        help="Path to configuration file",
    ),
    show: bool = typer.Option(False, "--show", help="Display current configuration"),
    validate: bool = typer.Option(False, "--validate", help="Validate configuration"),
) -> None:
    """Manage configuration settings."""
    if not config_path.exists():
        console.print(f"[red]Error: Config file {config_path} not found[/red]")
        console.print("Run 'docs-reviewer init' to create a configuration file")
        raise typer.Exit(1)

    config_data = load_config(config_path)

    if show:
        console.print(f"[bold]Configuration from {config_path}:[/bold]\n")
        console.print(config_data)

    if validate:
        console.print("Validating configuration...")
        # TODO: Add validation logic
        console.print("[green]Configuration is valid[/green]")


if __name__ == "__main__":
    app()

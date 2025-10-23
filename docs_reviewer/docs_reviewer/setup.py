"""Setup script for docs-reviewer package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="docs-reviewer",
    version="0.1.0",
    description="AI-powered documentation code snippet reviewer built on DeepAgents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Sydney Runkle",
    author_email="noreply@example.com",
    url="https://github.com/langchain-ai/deepagents",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "deepagents>=0.1.3",
        "typer>=0.9.0",
        "rich>=13.0.0",
        "pyyaml>=6.0",
        "pydantic>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "ruff",
        ],
    },
    entry_points={
        "console_scripts": [
            "docs-reviewer=docs_reviewer.cli:app",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)

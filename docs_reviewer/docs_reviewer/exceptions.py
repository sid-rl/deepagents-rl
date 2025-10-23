"""Custom exceptions for the docs reviewer."""


class DocsReviewerError(Exception):
    """Base exception for docs reviewer errors."""

    pass


class ConfigurationError(DocsReviewerError):
    """Raised when there's a configuration error."""

    pass


class MarkdownParseError(DocsReviewerError):
    """Raised when markdown parsing fails."""

    pass


class CodeExecutionError(DocsReviewerError):
    """Raised when code execution fails."""

    pass


class MCPServerError(DocsReviewerError):
    """Raised when MCP server operations fail."""

    pass

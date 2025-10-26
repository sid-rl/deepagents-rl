"""CompositeBackend: Route operations to different backends based on path prefix."""

from typing import Any, Literal, Optional, TYPE_CHECKING

from langchain.tools import ToolRuntime

from deepagents.backends.protocol import BackendProtocol, BackendProvider, StateBackendProvider, StateBackendProtocol
from deepagents.backends.state import StateBackend, StateBackendProvider
from langgraph.types import Command


class _CompositeBackend:
    
    def __init__(
        self,
        default: BackendProtocol | StateBackend,
        routes: dict[str, BackendProtocol],
    ) -> None:
        # Default backend
        self.default = default

        # Virtual routes
        self.routes = routes
        
        # Sort routes by length (longest first) for correct prefix matching
        self.sorted_routes = sorted(routes.items(), key=lambda x: len(x[0]), reverse=True)

    def _get_backend_and_key(self, key: str) -> tuple[BackendProtocol, str]:
        """Determine which backend handles this key and strip prefix.
        
        Args:
            key: Original file path
        
        Returns:
            Tuple of (backend, stripped_key) where stripped_key has the route
            prefix removed (but keeps leading slash).
        """
        # Check routes in order of length (longest first)
        for prefix, backend in self.sorted_routes:
            if key.startswith(prefix):
                # Strip prefix but keep leading slash
                # e.g., "/memories/notes.txt" → "/notes.txt"
                stripped_key = key[len(prefix) - 1:] if key[len(prefix) - 1:] else "/"
                return backend, stripped_key
        
        return self.default, key
    
    def ls(self, path: str) -> list[str]:
        """List files from backends, with appropriate prefixes.
        
        Args:
            path: Absolute path to directory.
        
        Returns:
            List of file paths with route prefixes added.
        """
        # Check if path matches a specific route
        for route_prefix, backend in self.sorted_routes:
            if path.startswith(route_prefix.rstrip("/")):
                # Query only the matching routed backend
                search_path = path[len(route_prefix) - 1:]
                keys = backend.ls(search_path if search_path else "/")
                return [f"{route_prefix[:-1]}{key}" for key in keys]
        
        # Path doesn't match a route: query only default backend
        return self.default.ls(path)
    
    def read(
        self, 
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """Read file content, routing to appropriate backend.
        
        Args:
            file_path: Absolute file path
            offset: Line offset to start reading from (0-indexed)
            limit: Maximum number of lines to readReturns:
            Formatted file content with line numbers, or error message.
        """
        backend, stripped_key = self._get_backend_and_key(file_path)
        return backend.read(stripped_key, offset=offset, limit=limit)
    
    def grep(
        self,
        pattern: str,
        path: Optional[str] = None,
        glob: Optional[str] = None,
        output_mode: str = "files_with_matches",
    ) -> str:
        """Search for a pattern in files, routing to appropriate backend(s).
        
        Args:
            pattern: String pattern to search for
            path: Path to search in (default "/")
            glob: Optional glob pattern to filter files (e.g., "*.py")
            output_mode: Output format - "files_with_matches", "content", or "count"Returns:
            Formatted search results based on output_mode.
        """
        # If path targets a specific route, search only that backend
        for route_prefix, backend in self.sorted_routes:
            if path is not None and path.startswith(route_prefix.rstrip("/")):
                search_path = path[len(route_prefix) - 1:]
                raw = backend.grep_raw(pattern, search_path if search_path else "/", glob)
                if isinstance(raw, str):
                    return raw
                if not raw:
                    return f"No matches found for pattern: '{pattern}'"
                prefixed = [{**m, "path": f"{route_prefix[:-1]}{m['path']}"} for m in raw]
                from .utils import format_grep_matches, truncate_if_too_long  # lazy import
                formatted = format_grep_matches(prefixed, output_mode)
                return truncate_if_too_long(formatted)  # type: ignore[arg-type]

        # Otherwise, search default and all routed backends and merge
        all_matches: list[dict] = []
        raw_default = self.default.grep_raw(pattern, path, glob)  # type: ignore[attr-defined]
        if isinstance(raw_default, str):
            return raw_default
        all_matches.extend(raw_default)

        for route_prefix, backend in self.routes.items():
            raw = backend.grep_raw(pattern, "/", glob)
            if isinstance(raw, str):
                return raw
            all_matches.extend({**m, "path": f"{route_prefix[:-1]}{m['path']}"} for m in raw)

        from .utils import format_grep_matches, truncate_if_too_long
        formatted = format_grep_matches(all_matches, output_mode)
        return truncate_if_too_long(formatted)  # type: ignore[arg-type]
    
    def glob(self, pattern: str, path: str = "/") -> list[str]:
        """Find files matching a glob pattern across all backends.

        Args:
            pattern: Glob pattern (e.g., "**/*.py", "*.txt", "/subdir/**/*.md")
            path: Base path to search from (default "/")Returns:
            List of absolute file paths matching the pattern.
        """
        results = []

        # Route based on path, not pattern
        for route_prefix, backend in self.sorted_routes:
            if path.startswith(route_prefix.rstrip("/")):
                # Path matches a specific route - search only that backend
                search_path = path[len(route_prefix) - 1:]
                matches = backend.glob(pattern, search_path if search_path else "/")
                results.extend(f"{route_prefix[:-1]}{match}" for match in matches)
                return sorted(results)

        # Path doesn't match any specific route - search default backend AND all routed backends
        default_matches = self.default.glob(pattern, path)
        results.extend(default_matches)

        # Also search in all routed backends and prefix results
        for route_prefix, backend in self.routes.items():
            matches = backend.glob(pattern, "/")
            results.extend(f"{route_prefix[:-1]}{match}" for match in matches)

        return sorted(results)


class CompositeStateBacked(_CompositeBackend):

    def __init__(
            self,
            default: StateBackend,
            routes: dict[str, BackendProtocol],
    ) -> None:
        self.default = default
        self.routes = routes

        # Sort routes by length (longest first) for correct prefix matching
        self.sorted_routes = sorted(routes.items(), key=lambda x: len(x[0]), reverse=True)

    def write(
            self,
            file_path: str,
            content: str,
    ) -> Command | str:
        """Create a new file, routing to appropriate backend.

        Args:
            file_path: Absolute file path
            content: File content as a stringReturns:
            Success message or Command object, or error if file already exists.
        """
        backend, stripped_key = self._get_backend_and_key(file_path)
        return backend.write(stripped_key, content)

    def edit(
            self,
            file_path: str,
            old_string: str,
            new_string: str,
            replace_all: bool = False,
    ) -> Command | str:
        """Edit a file, routing to appropriate backend.

        Args:
            file_path: Absolute file path
            old_string: String to find and replace
            new_string: Replacement string
            replace_all: If True, replace all occurrencesReturns:
            Success message or Command object, or error message on failure.
        """
        backend, stripped_key = self._get_backend_and_key(file_path)
        return backend.edit(stripped_key, old_string, new_string, replace_all=replace_all)


class CompositeBackend(_CompositeBackend):
    def write(
            self,
            file_path: str,
            content: str,
    ) -> str:
        """Create a new file, routing to appropriate backend.

        Args:
            file_path: Absolute file path
            content: File content as a stringReturns:
            Success message or Command object, or error if file already exists.
        """
        backend, stripped_key = self._get_backend_and_key(file_path)
        return backend.write(stripped_key, content)

    def edit(
            self,
            file_path: str,
            old_string: str,
            new_string: str,
            replace_all: bool = False,
    ) -> str:
        """Edit a file, routing to appropriate backend.

        Args:
            file_path: Absolute file path
            old_string: String to find and replace
            new_string: Replacement string
            replace_all: If True, replace all occurrencesReturns:
            Success message or Command object, or error message on failure.
        """
        backend, stripped_key = self._get_backend_and_key(file_path)
        return backend.edit(stripped_key, old_string, new_string, replace_all=replace_all)

class CompositeStateBackendProvider(StateBackendProvider):

    def __init__(self, routes: dict[str, BackendProtocol | BackendProvider | "BackendFactory"]):
        self.routes = routes

    def get_backend(self, runtime: ToolRuntime) -> StateBackendProtocol:
        from deepagents.backends.protocol import BackendFactory  # avoid circular import at module load
        # Build routed backends, allowing instances, providers, or factories
        built_routes: dict[str, BackendProtocol] = {}
        for k, v in self.routes.items():
            if isinstance(v, BackendProtocol):
                built_routes[k] = v
            elif callable(v):  # BackendFactory
                built_routes[k] = v(runtime)  # type: ignore[misc]
            else:
                built_routes[k] = v.get_backend(runtime)  # type: ignore[union-attr]

        # Default state-backed storage for writes/edits
        default_state = StateBackend(runtime)
        return CompositeStateBacked(default=default_state, routes=built_routes)

"""Version information for langgraph-checkpoint-redis."""

import importlib.metadata
import os
import sys

from redisvl.version import __version__ as __redisvl_version__


def _get_version() -> str:
    """Get version from package metadata or pyproject.toml."""
    try:
        # Try to get version from installed package metadata
        return importlib.metadata.version("langgraph-checkpoint-redis")
    except importlib.metadata.PackageNotFoundError:
        # Fallback for development/editable installs
        if sys.version_info >= (3, 11):
            import tomllib
        else:
            import tomli as tomllib

        from pathlib import Path

        # Get search depth from environment variable (default to 5)
        search_depth = int(
            os.environ.get("LANGGRAPH_REDIS_PYPROJECT_SEARCH_DEPTH", "5")
        )

        # Look for pyproject.toml in parent directories
        current = Path(__file__).resolve()
        for _ in range(search_depth):
            pyproject_path = current.parent / "pyproject.toml"
            if pyproject_path.exists():
                with open(pyproject_path, "rb") as f:
                    data = tomllib.load(f)
                    return data["tool"]["poetry"]["version"]
            current = current.parent

        raise RuntimeError(
            f"Unable to determine package version. "
            f"Package is not installed and pyproject.toml not found within {search_depth} levels. "
            f"Set LANGGRAPH_REDIS_PYPROJECT_SEARCH_DEPTH environment variable to adjust search depth."
        )


__version__ = _get_version()
__lib_name__ = f"langgraph-checkpoint-redis_v{__version__}"
__full_lib_name__ = f"redis-py(redisvl_v{__redisvl_version__};{__lib_name__})"

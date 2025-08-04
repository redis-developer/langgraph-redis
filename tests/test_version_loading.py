"""Test that version is properly loaded from pyproject.toml."""

from pathlib import Path
from unittest.mock import patch

import pytest


def test_version_matches_pyproject():
    """Test that the version in version.py matches pyproject.toml."""
    # Read version from pyproject.toml
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore

    # Find pyproject.toml
    current = Path(__file__).resolve()
    pyproject_path = None
    for _ in range(5):
        potential_path = current.parent / "pyproject.toml"
        if potential_path.exists():
            pyproject_path = potential_path
            break
        current = current.parent

    assert pyproject_path is not None, "Could not find pyproject.toml"

    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
        expected_version = data["tool"]["poetry"]["version"]

    # Import and check version
    from langgraph.checkpoint.redis.version import __version__

    assert __version__ == expected_version, (
        f"Version mismatch: version.py has '{__version__}' "
        f"but pyproject.toml has '{expected_version}'"
    )


def test_version_from_installed_package():
    """Test version loading from installed package metadata."""
    import importlib.metadata

    # Mock the package as installed with a specific version
    mock_version = "1.2.3"

    with patch.object(importlib.metadata, "version", return_value=mock_version):
        # Reload the version module to pick up the mocked version
        import importlib

        import langgraph.checkpoint.redis.version

        importlib.reload(langgraph.checkpoint.redis.version)

        assert langgraph.checkpoint.redis.version.__version__ == mock_version


def test_version_fallback_when_not_installed():
    """Test version loading fallback when package is not installed."""
    import importlib.metadata

    # Mock PackageNotFoundError
    def mock_version(name):
        raise importlib.metadata.PackageNotFoundError(name)

    with patch.object(importlib.metadata, "version", side_effect=mock_version):
        # Reload the version module
        import importlib

        import langgraph.checkpoint.redis.version

        importlib.reload(langgraph.checkpoint.redis.version)

        # Should fall back to reading from pyproject.toml
        # Let's verify it's a valid version string
        version = langgraph.checkpoint.redis.version.__version__
        assert isinstance(version, str)
        assert len(version.split(".")) >= 2  # At least major.minor


def test_version_fails_when_not_found():
    """Test that version loading fails with clear error when version cannot be determined."""
    import importlib.metadata

    # Mock PackageNotFoundError
    def mock_version(name):
        raise importlib.metadata.PackageNotFoundError(name)

    # Mock Path.exists to return False (no pyproject.toml found)
    with (
        patch.object(importlib.metadata, "version", side_effect=mock_version),
        patch.object(Path, "exists", return_value=False),
    ):
        # Attempting to reload should raise RuntimeError
        with pytest.raises(RuntimeError) as exc_info:
            import importlib

            import langgraph.checkpoint.redis.version

            importlib.reload(langgraph.checkpoint.redis.version)

        # Check error message
        assert "Unable to determine package version" in str(exc_info.value)
        assert "pyproject.toml not found" in str(exc_info.value)


def test_configurable_search_depth():
    """Test that pyproject.toml search depth can be configured via environment variable."""
    import importlib.metadata
    import os

    # Mock PackageNotFoundError to force fallback to pyproject.toml search
    def mock_version(name):
        raise importlib.metadata.PackageNotFoundError(name)

    # Test with custom search depth = 2 (should fail if pyproject.toml is deeper)
    with (
        patch.object(importlib.metadata, "version", side_effect=mock_version),
        patch.dict(os.environ, {"LANGGRAPH_REDIS_PYPROJECT_SEARCH_DEPTH": "2"}),
        patch.object(
            Path, "exists", return_value=False
        ),  # Mock no pyproject.toml found
    ):
        with pytest.raises(RuntimeError) as exc_info:
            import importlib

            import langgraph.checkpoint.redis.version

            importlib.reload(langgraph.checkpoint.redis.version)

        # Check that error message mentions the configured depth
        assert "2 levels" in str(exc_info.value)
        assert "LANGGRAPH_REDIS_PYPROJECT_SEARCH_DEPTH" in str(exc_info.value)

    # Test with larger search depth that should succeed
    with (
        patch.object(importlib.metadata, "version", side_effect=mock_version),
        patch.dict(os.environ, {"LANGGRAPH_REDIS_PYPROJECT_SEARCH_DEPTH": "10"}),
    ):
        import importlib

        import langgraph.checkpoint.redis.version

        importlib.reload(langgraph.checkpoint.redis.version)

        # Should succeed in finding pyproject.toml
        version = langgraph.checkpoint.redis.version.__version__
        assert isinstance(version, str)
        assert len(version.split(".")) >= 2


def test_lib_name_format():
    """Test that library name is formatted correctly."""
    from langgraph.checkpoint.redis.version import (
        __full_lib_name__,
        __lib_name__,
        __redisvl_version__,
        __version__,
    )

    # Check lib_name format
    assert __lib_name__ == f"langgraph-checkpoint-redis_v{__version__}"

    # Check full_lib_name format
    expected_full = f"redis-py(redisvl_v{__redisvl_version__};{__lib_name__})"
    assert __full_lib_name__ == expected_full

    # Ensure redisvl version is included
    assert __redisvl_version__ in __full_lib_name__


def test_version_is_valid_semver():
    """Test that the version string follows semantic versioning."""
    from langgraph.checkpoint.redis.version import __version__

    # Basic semver check
    parts = __version__.split(".")
    assert len(parts) >= 2, f"Version should have at least major.minor: {__version__}"

    # Each part should be a valid integer (for basic versions)
    # Note: This is simplified and doesn't handle pre-release versions like "1.0.0-beta"
    for i, part in enumerate(parts[:3]):  # Check major, minor, patch if present
        try:
            int(part)
        except ValueError:
            # Could be a pre-release version
            if i == 2 and "-" in part:
                # Patch version might have pre-release suffix
                patch_part = part.split("-")[0]
                int(patch_part)
            else:
                pytest.fail(f"Invalid version part '{part}' in version {__version__}")


if __name__ == "__main__":
    # Run the tests
    test_version_matches_pyproject()
    test_version_from_installed_package()
    test_version_fallback_when_not_installed()
    test_version_fails_when_not_found()
    test_lib_name_format()
    test_version_is_valid_semver()
    print("All version tests passed!")

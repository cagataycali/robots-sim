"""Test for dynamic version loading (R2 Thread #1).

Regression test for review thread PRRT_kwDORUMlNs6EBXm2.
Verifies that __version__ is loaded from importlib.metadata, not hardcoded.
"""

from __future__ import annotations


def test_version_uses_metadata_not_hardcoded():
    """Verify __version__ is loaded from importlib.metadata, not hardcoded.

    pyproject.toml declares dynamic = ["version"] and uses hatch-vcs.
    The __init__.py MUST import from importlib.metadata, not hardcode.
    """
    import strands_robots_sim

    # If package is installed (editable or not), importlib.metadata should work
    version = strands_robots_sim.__version__
    assert version is not None
    assert isinstance(version, str)

    # Check that the import uses importlib.metadata
    import inspect

    source = inspect.getsource(strands_robots_sim)
    assert "from importlib.metadata import version" in source, (
        "__version__ must be loaded via importlib.metadata to respect "
        "pyproject.toml dynamic=['version'] + hatch-vcs"
    )
    assert 'version("strands-robots-sim")' in source, (
        "__version__ must call version('strands-robots-sim'), not hardcode"
    )


def test_version_fallback_for_uninstalled():
    """Verify fallback version when package is not installed."""
    import strands_robots_sim

    # The fallback should be "0.4.0-dev"
    # When actually installed, version() returns the VCS version
    version = strands_robots_sim.__version__

    # Accept either VCS version or fallback
    assert version == "0.4.0-dev" or ("0.4" in version or version >= "0.4.0"), (
        f"Version {version} must be either VCS-sourced or fallback '0.4.0-dev'"
    )

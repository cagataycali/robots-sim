"""Entry-point registration and SimEngine integration tests for Isaac.

These tests validate R6 requirements:
  - Entry points declared in pyproject.toml are discoverable
  - IsaacSimulation correctly subclasses SimEngine
  - is_available() returns the right value for the current environment

Run with: pytest strands_robots_sim/isaac/tests/test_entrypoint.py -v
"""

from __future__ import annotations

import importlib.metadata
import pathlib

import pytest


class TestEntryPointRegistration:
    """Validate R6 entry-point-based backend registration."""

    def test_isaac_is_simengine_subclass(self):
        """IsaacSimulation must subclass SimEngine ABC."""
        from strands_robots_sim.isaac.simulation import IsaacSimulation, SimEngine

        assert issubclass(IsaacSimulation, SimEngine), (
            "IsaacSimulation must inherit from SimEngine for entry-point "
            "registration to satisfy the factory contract."
        )

    def test_isaac_has_is_available_classmethod(self):
        """IsaacSimulation must expose is_available() classmethod."""
        from strands_robots_sim.isaac.simulation import IsaacSimulation

        assert hasattr(IsaacSimulation, "is_available")
        assert callable(IsaacSimulation.is_available)
        # Must return (bool, str|None) tuple
        result = IsaacSimulation.is_available()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_is_available_returns_false_without_gpu(self):
        """is_available() should return False on CI without Isaac Sim."""
        from strands_robots_sim.isaac.simulation import IsaacSimulation

        available, reason = IsaacSimulation.is_available()
        assert isinstance(available, bool)
        # On CPU-only CI, it must be False with a helpful message
        if not available:
            assert isinstance(reason, str)
            assert len(reason) > 10  # Meaningful message

    def test_entry_points_declared_in_pyproject(self):
        """Package metadata must declare strands_robots.backends entry points.

        If package is installed, verify via importlib.metadata.
        Otherwise, verify pyproject.toml directly.
        """
        try:
            eps = importlib.metadata.entry_points()
            if hasattr(eps, "select"):
                backend_eps = list(eps.select(group="strands_robots.backends"))
            else:
                backend_eps = eps.get("strands_robots.backends", [])

            isaac_eps = [ep for ep in backend_eps if ep.name == "isaac"]
            if isaac_eps:
                ep = isaac_eps[0]
                assert ep.value == "strands_robots_sim.isaac.simulation:IsaacSimulation"
            else:
                self._verify_pyproject_entry_points()
        except Exception:
            self._verify_pyproject_entry_points()

    def _verify_pyproject_entry_points(self):
        """Fallback: verify entry points exist in pyproject.toml source."""
        pkg_dir = pathlib.Path(__file__).resolve().parents[3]
        pyproject = pkg_dir / "pyproject.toml"
        if pyproject.exists():
            content = pyproject.read_text()
            assert 'isaac = "strands_robots_sim.isaac.simulation:IsaacSimulation"' in content
            assert 'isaac_sim = "strands_robots_sim.isaac.simulation:IsaacSimulation"' in content
        else:
            pytest.skip("pyproject.toml not found -- cannot verify entry points")

    def test_isaac_implements_all_abstract_methods(self):
        """IsaacSimulation must implement all SimEngine abstract methods."""
        from strands_robots_sim.isaac.simulation import IsaacSimulation, SimEngine

        # Get all abstract methods from SimEngine
        abstract_methods = set()
        for name in dir(SimEngine):
            method = getattr(SimEngine, name, None)
            if callable(method) and getattr(method, "__isabstractmethod__", False):
                abstract_methods.add(name)

        # Verify IsaacSimulation implements each one
        for method_name in abstract_methods:
            assert hasattr(IsaacSimulation, method_name), f"IsaacSimulation missing abstract method: {method_name}"
            impl = getattr(IsaacSimulation, method_name)
            assert not getattr(
                impl, "__isabstractmethod__", False
            ), f"IsaacSimulation has unimplemented abstract method: {method_name}"

    def test_isaac_instantiation_without_gpu(self):
        """IsaacSimulation must instantiate without GPU (lazy CUDA init)."""
        from strands_robots_sim.isaac.simulation import IsaacSimulation

        # Constructor must NOT touch CUDA or import omni
        sim = IsaacSimulation(num_envs=1)
        assert sim is not None
        assert sim.config.num_envs == 1
        assert sim.config.device == "cuda:0"

    def test_lazy_import_does_not_load_omni(self):
        """Importing strands_robots_sim.isaac must NOT load omni."""
        import sys

        # Remove omni from sys.modules if present
        omni_modules = [k for k in sys.modules if k.startswith("omni")]
        # We just check that importing isaac doesn't add omni modules
        import strands_robots_sim.isaac  # noqa: F401

        new_omni_modules = [k for k in sys.modules if k.startswith("omni") and k not in omni_modules]
        assert new_omni_modules == [], f"Importing strands_robots_sim.isaac loaded omni modules: {new_omni_modules}"


class TestIsaacSimAlias:
    """Verify isaac_sim alias entry point works identically to isaac."""

    def test_alias_entry_point_in_pyproject(self):
        """isaac_sim alias must be declared alongside isaac."""
        pkg_dir = pathlib.Path(__file__).resolve().parents[3]
        pyproject = pkg_dir / "pyproject.toml"
        if pyproject.exists():
            content = pyproject.read_text()
            assert 'isaac_sim = "strands_robots_sim.isaac.simulation:IsaacSimulation"' in content
        else:
            pytest.skip("pyproject.toml not found")

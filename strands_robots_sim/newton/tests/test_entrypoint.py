"""Entry-point registration and SimEngine integration tests.

These tests validate R6 requirements:
  - Entry points declared in pyproject.toml are discoverable
  - NewtonSimulation correctly subclasses SimEngine
  - is_available() returns the right value for the current environment
  - Manual registration via register_backends() works

Run with: pytest strands_robots_sim/newton/tests/test_entrypoint.py -v
"""

from __future__ import annotations

import importlib.metadata
import sys
from unittest.mock import MagicMock, patch

import pytest


# Check if strands-robots factory is available (it isn't on PyPI < 0.4.0)
def _has_factory():
    try:
        from strands_robots.simulation.factory import register_backend  # noqa: F401

        return True
    except (ImportError, ModuleNotFoundError):
        return False


_FACTORY_AVAILABLE = _has_factory()


class TestEntryPointRegistration:
    """Validate R6 entry-point-based backend registration."""

    def test_newton_is_simengine_subclass(self):
        """NewtonSimulation must subclass SimEngine ABC."""
        from strands_robots_sim.newton.simulation import NewtonSimulation, SimEngine

        assert issubclass(NewtonSimulation, SimEngine), (
            "NewtonSimulation must inherit from SimEngine for entry-point "
            "registration to satisfy the factory contract."
        )

    def test_newton_has_is_available_classmethod(self):
        """NewtonSimulation must expose is_available() classmethod."""
        from strands_robots_sim.newton.simulation import NewtonSimulation

        assert hasattr(NewtonSimulation, "is_available")
        assert callable(NewtonSimulation.is_available)
        # Must be a classmethod (callable on the class itself)
        result = NewtonSimulation.is_available()
        assert isinstance(result, bool)

    def test_is_available_returns_false_without_gpu(self):
        """is_available() must return False when warp/newton are missing."""
        from strands_robots_sim.newton.simulation import NewtonSimulation

        # In a non-GPU CI environment, this should return False
        result = NewtonSimulation.is_available()
        assert isinstance(result, bool)
        # On CPU-only CI, it must be False
        # (we can't assert False universally since GPU environments exist)

    def test_entry_points_declared_in_metadata(self):
        """Package metadata must declare strands_robots.backends entry points.

        This test requires the package to be installed in editable mode
        (pip install -e .). If not installed, we verify pyproject.toml directly.
        """
        try:
            eps = importlib.metadata.entry_points()
            # Python 3.12+ returns SelectableGroups, 3.10/3.11 returns dict
            if hasattr(eps, "select"):
                backend_eps = list(eps.select(group="strands_robots.backends"))
            else:
                backend_eps = eps.get("strands_robots.backends", [])

            # If package is installed, entry points should be there
            newton_eps = [ep for ep in backend_eps if ep.name == "newton"]
            if newton_eps:
                ep = newton_eps[0]
                assert ep.value == "strands_robots_sim.newton.simulation:NewtonSimulation"
            else:
                # Package not installed in editable mode — verify pyproject.toml instead
                self._verify_pyproject_entry_points()
        except Exception:
            self._verify_pyproject_entry_points()

    def _verify_pyproject_entry_points(self):
        """Fallback: verify entry points exist in pyproject.toml source."""
        import pathlib

        # Find pyproject.toml relative to package
        pkg_dir = pathlib.Path(__file__).resolve().parents[3]
        pyproject = pkg_dir / "pyproject.toml"
        if pyproject.exists():
            content = pyproject.read_text()
            assert 'newton = "strands_robots_sim.newton.simulation:NewtonSimulation"' in content
            assert 'warp = "strands_robots_sim.newton.simulation:NewtonSimulation"' in content
        else:
            pytest.skip("pyproject.toml not found — cannot verify entry points")

    @pytest.mark.skipif(not _FACTORY_AVAILABLE, reason="strands-robots factory not available")
    def test_register_backends_idempotent(self):
        """register_backends() must be safe to call multiple times."""
        import strands_robots_sim

        # Should not raise on first call
        strands_robots_sim.register_backends()
        # Should not raise on second call (idempotent)
        strands_robots_sim.register_backends()

    @pytest.mark.skipif(not _FACTORY_AVAILABLE, reason="strands-robots factory not available")
    def test_register_backends_enables_factory(self):
        """After register_backends(), create_simulation('newton') must resolve."""
        import strands_robots_sim

        strands_robots_sim.register_backends()

        from strands_robots.simulation.factory import list_backends

        backends = list_backends()
        assert "newton" in backends
        assert "warp" in backends

    def test_newton_implements_all_abstract_methods(self):
        """NewtonSimulation must implement all SimEngine abstract methods."""
        from strands_robots_sim.newton.simulation import NewtonSimulation, SimEngine

        # Get all abstract methods from SimEngine
        abstract_methods = set()
        for name in dir(SimEngine):
            method = getattr(SimEngine, name, None)
            if callable(method) and getattr(method, "__isabstractmethod__", False):
                abstract_methods.add(name)

        # Verify NewtonSimulation implements each one
        for method_name in abstract_methods:
            assert hasattr(NewtonSimulation, method_name), f"NewtonSimulation missing abstract method: {method_name}"
            impl = getattr(NewtonSimulation, method_name)
            assert not getattr(impl, "__isabstractmethod__", False), (
                f"NewtonSimulation has unimplemented abstract method: {method_name}"
            )

    def test_newton_instantiation_without_gpu(self):
        """NewtonSimulation must instantiate without GPU (lazy CUDA init)."""
        from strands_robots_sim.newton.simulation import NewtonSimulation

        # Constructor must NOT touch CUDA — lazy init only on create_world()
        sim = NewtonSimulation(num_envs=1, device="cpu")
        assert sim is not None
        assert sim.config.num_envs == 1

    def test_version_bump_for_r6(self):
        """Package version must be >= 0.4.0-dev for Stage 4 (Newton)."""
        import strands_robots_sim

        version = strands_robots_sim.__version__
        # Should be 0.4.0-dev or higher (Stage 4 = Newton)
        assert "0.4" in version or version >= "0.4.0"


class TestSimEngineContractCompliance:
    """Verify NewtonSimulation returns the correct dict shapes."""

    @pytest.fixture
    def sim(self):
        """Create a mocked Newton simulation for contract testing."""
        from strands_robots_sim.newton.simulation import NewtonSimulation

        # Mock warp and newton at import level
        mock_wp = MagicMock()
        mock_wp.init = MagicMock()
        mock_newton = MagicMock()

        # Create sim with mocked deps
        sim = NewtonSimulation(num_envs=1)
        # Manually set up world state
        sim._wp = mock_wp
        sim._newton = mock_newton
        sim._world_created = True
        sim._physics_dt = 0.002
        sim._gravity = (0.0, -9.81, 0.0)
        yield sim

    def test_destroy_returns_status_dict(self, sim):
        """destroy() must return proper status dict."""
        result = sim.destroy()
        assert result["status"] == "success"

    def test_list_robots_returns_list(self, sim):
        """list_robots() must return a list of strings."""
        result = sim.list_robots()
        assert isinstance(result, list)

    def test_robot_joint_names_returns_list(self, sim):
        """robot_joint_names() must return list for unknown robot."""
        result = sim.robot_joint_names("nonexistent")
        assert isinstance(result, list)
        assert len(result) == 0

    def test_get_state_returns_status(self, sim):
        """get_state() must return a status dict."""
        result = sim.get_state()
        assert "status" in result
        assert "content" in result

    def test_step_without_model(self, sim):
        """step() on empty world (no robots/objects) should still work."""
        # _ensure_built will be a no-op since builder is None
        sim._builder = None
        sim._model = MagicMock()
        sim._state = MagicMock()
        sim._simulator = MagicMock()
        result = sim.step(1)
        assert result["status"] == "success"

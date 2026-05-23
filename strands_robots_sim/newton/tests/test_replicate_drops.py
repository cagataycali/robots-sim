"""Test for replicate() dropping URDF robots and add_object objects (R2 Thread #3).

Regression test for review thread PRRT_kwDORUMlNs6EBXm9.
Verifies that replicate() raises NotImplementedError when URDF robots or
add_object() objects exist, instead of silently dropping them.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestReplicateDropsBehavior:
    """Test that replicate() does NOT silently drop URDF robots or add_object() objects."""

    @pytest.fixture
    def mock_env(self):
        """Setup mocked Newton/Warp environment."""
        newton = MagicMock()
        builder = MagicMock()
        builder.add_ground_plane = MagicMock()
        builder.add_articulation = MagicMock()
        builder.add_body = MagicMock(side_effect=lambda *a, **kw: 0)
        builder.add_shape_box = MagicMock()
        newton.ModelBuilder.return_value = builder

        model = MagicMock()
        model.gravity = (0.0, -9.81, 0.0)
        state = MagicMock()
        model.state.return_value = state
        builder.finalize.return_value = model

        newton.SolverType = MagicMock()
        newton.SolverType.MUJOCO = "MUJOCO"
        simulator = MagicMock()
        newton.Simulator.return_value = simulator

        wp = MagicMock()
        wp.init = MagicMock()

        return newton, wp

    @pytest.fixture
    def sim(self, mock_env):
        """Create a NewtonSimulation with mocked deps."""
        newton, wp = mock_env
        with (
            patch("strands_robots_sim.newton.simulation._lazy_import_warp", return_value=wp),
            patch("strands_robots_sim.newton.simulation._lazy_import_newton", return_value=newton),
        ):
            from strands_robots_sim.newton.config import NewtonConfig
            from strands_robots_sim.newton.simulation import NewtonSimulation

            sim = NewtonSimulation(NewtonConfig())
            sim.create_world()
            return sim

    def test_replicate_with_urdf_robot_raises_not_implemented(self, sim):
        """replicate() MUST raise NotImplementedError when URDF robot exists.

        Previously, replicate() silently dropped URDF robots (procedural=None).
        This is a CRITICAL bug for examples/libero_newton_fleet.py which loads
        LIBERO URDF robots.
        """
        from strands_robots_sim.newton.simulation import _RobotState

        # Simulate a URDF-loaded robot (procedural=None)
        sim._robots["panda_urdf"] = _RobotState(
            name="panda_urdf",
            procedural=None,  # URDF loaded, not procedural
            joint_start=0,
            joint_count=7,
            body_start=0,
            body_count=8,
            joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"],
        )

        # replicate() MUST raise NotImplementedError
        with pytest.raises(NotImplementedError, match="URDF-loaded robots"):
            sim.replicate(2)

    def test_replicate_with_add_object_raises_not_implemented(self, sim):
        """replicate() MUST raise NotImplementedError when add_object() objects exist.

        Previously, replicate() silently dropped objects added via add_object().
        """
        # Add an object via add_object()
        sim.add_object("cube", shape="box", position=[0.3, 0, 0.05])

        # replicate() MUST raise NotImplementedError
        with pytest.raises(NotImplementedError, match="add_object"):
            sim.replicate(2)

    def test_replicate_with_urdf_and_object_raises_not_implemented(self, sim):
        """replicate() MUST raise when BOTH URDF robot AND add_object() object exist.

        This is the exact scenario in examples/libero_newton_fleet.py.
        """
        from strands_robots_sim.newton.simulation import _RobotState

        # URDF robot
        sim._robots["panda_urdf"] = _RobotState(
            name="panda_urdf",
            procedural=None,
            joint_start=0,
            joint_count=7,
            body_start=0,
            body_count=8,
            joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"],
        )

        # add_object() object
        sim.add_object("cube", shape="box")

        # replicate() MUST raise (URDF check comes first)
        with pytest.raises(NotImplementedError):
            sim.replicate(2)

    def test_replicate_with_procedural_robot_works(self, sim):
        """replicate() SHOULD work with procedural robots (existing behavior)."""
        # Add a procedural robot
        sim.add_robot("so100")

        # Should NOT raise
        try:
            sim.replicate(2)
        except NotImplementedError:
            pytest.fail("replicate() should NOT raise for procedural robots")

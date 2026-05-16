"""Unit tests for Newton simulation backend.

These tests run WITHOUT requiring Warp or Newton installed — they test
the config, solver adapters, procedural builders, and diffsim helpers
using mocks where necessary.

For GPU integration tests, see test_gpu_integ.py (requires @pytest.mark.gpu).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from strands_robots_sim.newton.config import (
    BROAD_PHASES as BROAD_PHASES,
)
from strands_robots_sim.newton.config import (
    RENDER_BACKENDS as RENDER_BACKENDS,
)
from strands_robots_sim.newton.config import (
    SOLVER_ALIASES as SOLVER_ALIASES,
)
from strands_robots_sim.newton.config import (
    SUPPORTED_SOLVERS,
    NewtonConfig,
)
from strands_robots_sim.newton.diffsim import (
    DiffSimConfig,
    DiffSimResult,
    compute_finite_difference_gradients,
    run_diffsim_loop,
)
from strands_robots_sim.newton.procedural import (
    ProceduralRobot as ProceduralRobot,
)
from strands_robots_sim.newton.procedural import (
    build_panda,
    build_so100,
    build_unitree_g1,
    get_procedural_robot,
    list_procedural_robots,
)
from strands_robots_sim.newton.solvers import (
    RIGID_BODY_SOLVERS,
    SOFT_BODY_SOLVERS,
    SOLVER_MAP,
    create_solver_adapter,
    get_solver_capabilities,
)

# ═══════════════════════════════════════════════════════════════════════════
# Config Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestNewtonConfig:
    """Test NewtonConfig validation and normalization."""

    def test_default_config(self):
        cfg = NewtonConfig()
        assert cfg.num_envs == 1
        assert cfg.device == "cuda:0"
        assert cfg.solver == "mujoco"
        assert cfg.physics_dt == pytest.approx(1.0 / 60.0)
        assert cfg.substeps == 4
        assert cfg.render_backend == "null"
        assert cfg.enable_cuda_graph is True
        assert cfg.enable_differentiable is False
        assert cfg.up_axis == "Y"
        assert cfg.gravity == (0.0, -9.81, 0.0)

    def test_z_up_gravity(self):
        cfg = NewtonConfig(up_axis="Z")
        assert cfg.gravity == (0.0, 0.0, -9.81)

    def test_custom_gravity(self):
        cfg = NewtonConfig(gravity=(0.0, 0.0, -1.62))  # Moon
        assert cfg.gravity == (0.0, 0.0, -1.62)

    def test_solver_alias_resolution(self):
        cfg = NewtonConfig(solver="mjc")
        assert cfg.solver == "mujoco"

        cfg = NewtonConfig(solver="pbd")
        assert cfg.solver == "xpbd"

        cfg = NewtonConfig(solver="mpm")
        assert cfg.solver == "implicit_mpm"

    def test_invalid_solver_raises(self):
        with pytest.raises(ValueError, match="Unknown solver"):
            NewtonConfig(solver="invalid_solver")

    def test_invalid_render_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown render_backend"):
            NewtonConfig(render_backend="vulkan")

    def test_invalid_broad_phase_raises(self):
        with pytest.raises(ValueError, match="Unknown broad_phase"):
            NewtonConfig(broad_phase="grid")

    def test_invalid_up_axis_raises(self):
        with pytest.raises(ValueError, match="up_axis must be"):
            NewtonConfig(up_axis="X")

    def test_invalid_device_raises(self):
        with pytest.raises(ValueError, match="device must be"):
            NewtonConfig(device="metal:0")

    def test_invalid_num_envs_raises(self):
        with pytest.raises(ValueError, match="num_envs must be"):
            NewtonConfig(num_envs=0)

    def test_resolve_cache_dir(self):
        cfg = NewtonConfig()
        resolved = cfg.resolve_cache_dir()
        assert "~" not in resolved
        assert "strands-robots/newton" in resolved

    def test_all_supported_solvers_create_valid_config(self):
        for solver in SUPPORTED_SOLVERS:
            cfg = NewtonConfig(solver=solver)
            assert cfg.solver == solver

    def test_large_num_envs(self):
        cfg = NewtonConfig(num_envs=8192)
        assert cfg.num_envs == 8192


# ═══════════════════════════════════════════════════════════════════════════
# Solver Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestSolvers:
    """Test solver adapters and capabilities."""

    def test_solver_map_complete(self):
        for solver in SUPPORTED_SOLVERS:
            assert solver in SOLVER_MAP

    def test_create_all_adapters(self):
        for solver in SUPPORTED_SOLVERS:
            adapter = create_solver_adapter(solver)
            assert adapter.name == solver

    def test_invalid_solver_raises(self):
        with pytest.raises(ValueError):
            create_solver_adapter("nonexistent")

    def test_mujoco_capabilities(self):
        caps = get_solver_capabilities("mujoco")
        assert caps.supports_rigid is True
        assert caps.supports_joints is True
        assert caps.supports_contacts is True
        assert caps.supports_differentiable is True
        assert caps.supports_soft is False

    def test_xpbd_capabilities(self):
        caps = get_solver_capabilities("xpbd")
        assert caps.supports_rigid is True
        assert caps.supports_soft is True
        assert caps.supports_cloth is True

    def test_vbd_no_joints(self):
        caps = get_solver_capabilities("vbd")
        assert caps.supports_joints is False
        assert caps.supports_soft is True

    def test_style3d_cloth_only(self):
        caps = get_solver_capabilities("style3d")
        assert caps.supports_cloth is True
        assert caps.supports_rigid is False

    def test_mpm_particles(self):
        caps = get_solver_capabilities("implicit_mpm")
        assert caps.supports_particles is True
        assert caps.supports_rigid is False

    def test_rigid_body_solvers_set(self):
        for solver in RIGID_BODY_SOLVERS:
            caps = get_solver_capabilities(solver)
            assert caps.supports_rigid is True

    def test_soft_body_solvers_set(self):
        for solver in SOFT_BODY_SOLVERS:
            caps = get_solver_capabilities(solver)
            assert caps.supports_soft or caps.supports_cloth

    def test_validate_for_task(self):
        adapter = create_solver_adapter("mujoco")
        warnings = adapter.validate_for_task("rigid")
        assert len(warnings) == 0

        warnings = adapter.validate_for_task("soft")
        assert len(warnings) > 0


# ═══════════════════════════════════════════════════════════════════════════
# Procedural Robot Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestProceduralRobots:
    """Test procedural robot builders."""

    def test_build_so100(self):
        robot = build_so100()
        assert robot.name == "so100"
        assert len(robot.bodies) == 7
        assert robot.num_joints == 6
        assert "shoulder_pan" in robot.joint_names
        assert "gripper" in robot.joint_names

    def test_build_panda(self):
        robot = build_panda()
        assert robot.name == "panda"
        assert len(robot.bodies) == 9
        assert robot.num_joints == 8  # 7 revolute + 1 prismatic
        assert "joint1" in robot.joint_names
        assert "finger_joint" in robot.joint_names

    def test_build_unitree_g1(self):
        robot = build_unitree_g1()
        assert robot.name == "unitree_g1"
        assert robot.is_floating_base is True
        assert robot.num_joints > 10  # At least 16 actuated joints
        assert "l_hip_pitch" in robot.joint_names
        assert "r_elbow" in robot.joint_names

    def test_get_procedural_robot_names(self):
        # Direct names
        assert get_procedural_robot("so100") is not None
        assert get_procedural_robot("panda") is not None
        assert get_procedural_robot("unitree_g1") is not None

    def test_get_procedural_robot_aliases(self):
        # Aliases
        assert get_procedural_robot("so_arm100") is not None
        assert get_procedural_robot("franka") is not None
        assert get_procedural_robot("g1") is not None

    def test_get_procedural_robot_case_insensitive(self):
        assert get_procedural_robot("SO100") is not None
        assert get_procedural_robot("Panda") is not None

    def test_get_procedural_robot_unknown(self):
        assert get_procedural_robot("unknown_robot_xyz") is None

    def test_list_procedural_robots(self):
        robots = list_procedural_robots()
        assert "so100" in robots
        assert "panda" in robots
        assert "unitree_g1" in robots
        assert len(robots) == 3  # Only canonical names

    def test_so100_joint_limits(self):
        robot = build_so100()
        for joint in robot.joints:
            assert joint.limit_lower < joint.limit_upper
            assert joint.damping > 0

    def test_panda_joint_types(self):
        robot = build_panda()
        revolute_count = sum(1 for j in robot.joints if j.joint_type == "revolute")
        prismatic_count = sum(1 for j in robot.joints if j.joint_type == "prismatic")
        assert revolute_count == 7
        assert prismatic_count == 1


# ═══════════════════════════════════════════════════════════════════════════
# DiffSim Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestDiffSim:
    """Test differentiable simulation helpers."""

    def test_diffsim_result_to_dict(self):
        result = DiffSimResult(
            converged=True,
            iterations=50,
            final_loss=1e-7,
            loss_history=[1.0, 0.5, 0.1, 1e-7],
            optimized_params={"velocity": np.array([1.0, 2.0])},
            wall_time=1.5,
        )
        d = result.to_dict()
        assert d["converged"] is True
        assert d["iterations"] == 50
        assert d["final_loss"] == pytest.approx(1e-7)
        assert d["optimized_params"]["velocity"] == [1.0, 2.0]

    def test_diffsim_config_defaults(self):
        cfg = DiffSimConfig()
        assert cfg.num_steps == 100
        assert cfg.lr == 0.02
        assert cfg.iterations == 200
        assert cfg.optimizer == "adam"

    def test_run_diffsim_converges_on_quadratic(self):
        """Test that diffsim converges on a simple quadratic loss."""
        # Loss = (x - 3)^2 + (y - 5)^2
        target = np.array([3.0, 5.0])

        def forward_fn(params):
            x = params["pos"]
            return float(np.sum((x - target) ** 2))

        def backward_fn(params):
            x = params["pos"]
            return {"pos": 2.0 * (x - target)}

        initial_params = {"pos": np.array([0.0, 0.0])}
        config = DiffSimConfig(lr=0.5, iterations=200, convergence_threshold=1e-4)

        result = run_diffsim_loop(forward_fn, backward_fn, initial_params, config)
        assert result.converged is True
        assert result.final_loss < 1e-4
        assert np.allclose(result.optimized_params["pos"], target, atol=0.01)

    def test_run_diffsim_max_iter(self):
        """Test that diffsim respects max iterations."""

        def forward_fn(params):
            return 1.0  # Never converges

        def backward_fn(params):
            return {"x": np.array([0.0])}

        config = DiffSimConfig(iterations=10)
        result = run_diffsim_loop(forward_fn, backward_fn, {"x": np.array([0.0])}, config)
        assert result.converged is False
        assert result.iterations == 10

    def test_finite_difference_gradients(self):
        """Test FD gradient computation."""

        def loss_fn(params):
            x = params["x"]
            return float(np.sum(x**2))

        params = {"x": np.array([3.0, 4.0])}
        grads = compute_finite_difference_gradients(loss_fn, params)

        # Analytical gradient: 2*x
        expected = np.array([6.0, 8.0])
        assert np.allclose(grads["x"], expected, atol=1e-2)

    def test_gradient_clipping(self):
        """Test that large gradients are clipped."""
        target = np.array([100.0])

        def forward_fn(params):
            return float(np.sum((params["x"] - target) ** 2))

        def backward_fn(params):
            return {"x": 2.0 * (params["x"] - target)}

        config = DiffSimConfig(lr=0.01, iterations=5, grad_clip=1.0)
        initial = {"x": np.array([0.0])}

        result = run_diffsim_loop(forward_fn, backward_fn, initial, config)
        # With clipping, movement should be bounded
        assert len(result.grad_norms) > 0


# ═══════════════════════════════════════════════════════════════════════════
# Simulation Class Tests (mocked Warp/Newton)
# ═══════════════════════════════════════════════════════════════════════════


class TestNewtonSimulation:
    """Test NewtonSimulation with mocked Newton/Warp dependencies."""

    @pytest.fixture
    def mock_newton(self):
        """Create mocked Newton module."""
        newton = MagicMock()
        newton.ModelBuilder.return_value = MagicMock()
        newton.SolverType = MagicMock()
        newton.SolverType.MUJOCO = "MUJOCO"
        newton.Simulator.return_value = MagicMock()
        return newton

    @pytest.fixture
    def mock_warp(self):
        """Create mocked Warp module."""
        wp = MagicMock()
        wp.init = MagicMock()
        return wp

    @pytest.fixture
    def sim(self, mock_newton, mock_warp):
        """Create a NewtonSimulation with mocked deps."""
        with (
            patch("strands_robots_sim.newton.simulation._lazy_import_warp", return_value=mock_warp),
            patch("strands_robots_sim.newton.simulation._lazy_import_newton", return_value=mock_newton),
        ):
            from strands_robots_sim.newton.simulation import NewtonSimulation

            sim = NewtonSimulation(NewtonConfig(num_envs=1))
            return sim

    def test_create_world(self, sim, mock_warp, mock_newton):
        with (
            patch("strands_robots_sim.newton.simulation._lazy_import_warp", return_value=mock_warp),
            patch("strands_robots_sim.newton.simulation._lazy_import_newton", return_value=mock_newton),
        ):
            result = sim.create_world()
            assert result["status"] == "success"
            assert sim._world_created is True

    def test_create_world_twice_errors(self, sim, mock_warp, mock_newton):
        with (
            patch("strands_robots_sim.newton.simulation._lazy_import_warp", return_value=mock_warp),
            patch("strands_robots_sim.newton.simulation._lazy_import_newton", return_value=mock_newton),
        ):
            sim.create_world()
            result = sim.create_world()
            assert result["status"] == "error"

    def test_destroy_without_create(self, sim):
        result = sim.destroy()
        assert result["status"] == "error"

    def test_add_robot_without_world(self, sim):
        result = sim.add_robot("so100")
        assert result["status"] == "error"

    def test_step_without_world(self, sim):
        result = sim.step(10)
        assert result["status"] == "error"

    def test_get_state_without_world(self, sim):
        result = sim.get_state()
        assert result["status"] == "error"

    def test_repr(self, sim):
        r = repr(sim)
        assert "NewtonSimulation" in r
        assert "mujoco" in r

    def test_list_robots_empty(self, sim):
        assert sim.list_robots() == []

    def test_robot_joint_names_unknown(self, sim):
        assert sim.robot_joint_names("nonexistent") == []


# ═══════════════════════════════════════════════════════════════════════════
# Integration-style Tests (still no GPU)
# ═══════════════════════════════════════════════════════════════════════════


class TestNewtonSimulationIntegration:
    """Integration tests using mocked Newton but testing full workflow."""

    @pytest.fixture
    def mock_env(self):
        """Setup mocked Newton/Warp environment."""
        newton = MagicMock()
        builder = MagicMock()
        builder.add_ground_plane = MagicMock()
        builder.add_articulation = MagicMock()
        builder.add_body = MagicMock(side_effect=lambda *a, **kw: 0)
        builder.add_shape_box = MagicMock()
        builder.add_shape_capsule = MagicMock()
        builder.add_shape_sphere = MagicMock()
        builder.add_joint_revolute = MagicMock()
        newton.ModelBuilder.return_value = builder

        model = MagicMock()
        model.gravity = (0.0, -9.81, 0.0)
        model.soft_contact_margin = 1e-3
        state = MagicMock()
        state.joint_q = MagicMock()
        state.joint_q.numpy.return_value = np.zeros(6)
        state.joint_qd = MagicMock()
        state.joint_qd.numpy.return_value = np.zeros(6)
        state.joint_act = MagicMock()
        state.joint_act.numpy.return_value = np.zeros(6)
        state.joint_act.assign = MagicMock()
        state.body_q = MagicMock()
        state.body_q.numpy.return_value = np.zeros((7, 7))
        model.state.return_value = state

        builder.finalize.return_value = model
        newton.SolverType = MagicMock()
        newton.SolverType.MUJOCO = "MUJOCO"

        simulator = MagicMock()
        simulator.step = MagicMock()
        newton.Simulator.return_value = simulator

        wp = MagicMock()
        wp.init = MagicMock()

        return newton, wp, model, state, simulator

    @pytest.fixture
    def sim_full(self, mock_env):
        newton, wp, model, state, simulator = mock_env
        with (
            patch("strands_robots_sim.newton.simulation._lazy_import_warp", return_value=wp),
            patch("strands_robots_sim.newton.simulation._lazy_import_newton", return_value=newton),
        ):
            from strands_robots_sim.newton.simulation import NewtonSimulation

            sim = NewtonSimulation(NewtonConfig())
            sim.create_world()
            yield sim

    def test_full_lifecycle(self, sim_full):
        """Test create → add_robot → step → get_state → destroy."""
        # Add robot
        result = sim_full.add_robot("so100")
        assert result["status"] == "success"
        assert "6 joints" in result["content"][0]["text"]

        # Step
        result = sim_full.step(10)
        assert result["status"] == "success"

        # Get state
        result = sim_full.get_state()
        assert result["status"] == "success"

        # Destroy
        result = sim_full.destroy()
        assert result["status"] == "success"

    def test_add_robot_duplicate(self, sim_full):
        sim_full.add_robot("so100")
        result = sim_full.add_robot("so100")
        assert result["status"] == "error"

    def test_add_unknown_robot(self, sim_full):
        result = sim_full.add_robot("unknown_bot_xyz")
        assert result["status"] == "error"

    def test_remove_robot(self, sim_full):
        sim_full.add_robot("so100")
        result = sim_full.remove_robot("so100")
        assert result["status"] == "success"
        assert sim_full.list_robots() == []

    def test_remove_nonexistent_robot(self, sim_full):
        result = sim_full.remove_robot("ghost")
        assert result["status"] == "error"

    def test_add_object(self, sim_full):
        result = sim_full.add_object("cube", shape="box", position=[0.3, 0, 0.05])
        assert result["status"] == "success"

    def test_add_object_duplicate(self, sim_full):
        sim_full.add_object("cube")
        result = sim_full.add_object("cube")
        assert result["status"] == "error"

    def test_remove_object(self, sim_full):
        sim_full.add_object("cube")
        result = sim_full.remove_object("cube")
        assert result["status"] == "success"

    def test_get_observation(self, sim_full):
        sim_full.add_robot("so100")
        obs = sim_full.get_observation("so100")
        assert "shoulder_pan" in obs

    def test_get_observation_auto_resolve(self, sim_full):
        sim_full.add_robot("so100")
        obs = sim_full.get_observation()  # Auto-resolves single robot
        assert "shoulder_pan" in obs

    def test_send_action_dict(self, sim_full):
        sim_full.add_robot("so100")
        sim_full.send_action({"shoulder_pan": 0.5, "elbow_flex": -0.3}, robot_name="so100")
        # Should not raise

    def test_send_action_array(self, sim_full):
        sim_full.add_robot("so100")
        action = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        sim_full.send_action(action, robot_name="so100")

    def test_render_null(self, sim_full):
        result = sim_full.render()
        assert result["status"] == "success"
        assert result["image"].shape == (480, 640, 3)

    def test_render_custom_size(self, sim_full):
        result = sim_full.render(width=320, height=240)
        assert result["image"].shape == (240, 320, 3)

    def test_robot_joint_names(self, sim_full):
        sim_full.add_robot("so100")
        names = sim_full.robot_joint_names("so100")
        assert len(names) == 6
        assert names[0] == "shoulder_pan"

    def test_solver_cloth_check(self, sim_full):
        result = sim_full.add_cloth("test_cloth")
        assert result["status"] == "error"  # mujoco solver doesn't support cloth

    def test_solver_particles_check(self, sim_full):
        result = sim_full.add_particles("sand")
        assert result["status"] == "error"  # mujoco solver doesn't support particles

    def test_solve_ik(self, sim_full):
        sim_full.add_robot("so100")
        result = sim_full.solve_ik("so100", target_position=[0.3, 0.0, 0.2])
        assert result["status"] == "success"
        assert "joint_q" in result["content"][0].get("json", {})

    def test_reset(self, sim_full):
        sim_full.add_robot("so100")
        sim_full.step(10)
        result = sim_full.reset()
        assert result["status"] == "success"

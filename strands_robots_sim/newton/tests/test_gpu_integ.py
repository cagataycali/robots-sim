"""GPU integration tests for Newton simulation backend.

These tests require:
    - NVIDIA GPU with CUDA 12+
    - warp-lang >= 1.11
    - newton-physics >= 1.0

Run with: pytest strands_robots_sim/newton/tests/test_gpu_integ.py -v -m gpu

Skip these in CI unless the GPU runner is available.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

# Mark all tests in this module as GPU-only
pytestmark = pytest.mark.gpu


def _check_gpu_available():
    """Check if GPU and required packages are available."""
    try:
        import warp as wp

        wp.init()
        if not wp.is_cuda_available():
            return False
        import newton  # noqa: F401

        return True
    except (ImportError, RuntimeError):
        return False


# Skip entire module if no GPU
if not _check_gpu_available():
    pytest.skip("No GPU or Newton/Warp not installed", allow_module_level=True)


class TestNewtonGPU:
    """GPU integration tests for Newton simulation."""

    def test_create_world_on_gpu(self):
        """Test that world creation works on real GPU."""
        from strands_robots_sim.newton.config import NewtonConfig
        from strands_robots_sim.newton.simulation import NewtonSimulation

        sim = NewtonSimulation(NewtonConfig(device="cuda:0"))
        result = sim.create_world()
        assert result["status"] == "success"
        sim.destroy()

    def test_so100_add_and_step(self):
        """Test adding SO-100 and stepping on GPU."""
        from strands_robots_sim.newton.config import NewtonConfig
        from strands_robots_sim.newton.simulation import NewtonSimulation

        sim = NewtonSimulation(NewtonConfig(device="cuda:0", solver="mujoco"))
        sim.create_world()
        result = sim.add_robot("so100")
        assert result["status"] == "success"

        result = sim.step(100)
        assert result["status"] == "success"

        obs = sim.get_observation("so100")
        assert "shoulder_pan" in obs

        sim.destroy()

    def test_4096_env_replication(self):
        """Test fleet-scale replication to 4096 envs."""
        from strands_robots_sim.newton.config import NewtonConfig
        from strands_robots_sim.newton.simulation import NewtonSimulation

        sim = NewtonSimulation(NewtonConfig(device="cuda:0", solver="mujoco", num_envs=4096))
        sim.create_world()
        sim.add_robot("so100")

        result = sim.replicate(4096)
        assert result["status"] == "success"

        # Benchmark throughput
        t0 = time.time()
        sim.step(100)
        elapsed = time.time() - t0

        # Target: < 2.0s on A100-class GPU
        print(f"4096 envs × 100 steps = {elapsed:.2f}s")
        assert elapsed < 10.0  # Generous limit for CI

        sim.destroy()

    def test_multiple_solvers(self):
        """Test that multiple solvers initialize correctly."""
        from strands_robots_sim.newton.config import NewtonConfig
        from strands_robots_sim.newton.simulation import NewtonSimulation

        for solver in ["mujoco", "semi_implicit", "xpbd"]:
            sim = NewtonSimulation(NewtonConfig(device="cuda:0", solver=solver))
            sim.create_world()
            sim.add_robot("so100")
            result = sim.step(10)
            assert result["status"] == "success", f"Solver {solver} failed"
            sim.destroy()

    def test_send_action_and_observe(self):
        """Test action → step → observation loop."""
        from strands_robots_sim.newton.config import NewtonConfig
        from strands_robots_sim.newton.simulation import NewtonSimulation

        sim = NewtonSimulation(NewtonConfig(device="cuda:0"))
        sim.create_world()
        sim.add_robot("so100")

        # Apply action
        action = {"shoulder_pan": 0.5, "elbow_flex": -0.3}
        sim.send_action(action, robot_name="so100", n_substeps=10)

        obs = sim.get_observation("so100")
        assert "shoulder_pan" in obs
        # Joint should have moved (not necessarily to exact target due to dynamics)

        sim.destroy()

    def test_render_opengl(self):
        """Test OpenGL rendering (if available)."""
        from strands_robots_sim.newton.config import NewtonConfig
        from strands_robots_sim.newton.simulation import NewtonSimulation

        sim = NewtonSimulation(NewtonConfig(device="cuda:0", render_backend="opengl"))
        sim.create_world()
        sim.add_robot("so100")
        sim.step(10)

        result = sim.render(width=640, height=480)
        assert result["status"] == "success"
        assert result["image"].shape == (480, 640, 3)

        sim.destroy()

    def test_diffsim_toy_optimization(self):
        """Test differentiable simulation on a toy task."""
        from strands_robots_sim.newton.config import NewtonConfig
        from strands_robots_sim.newton.simulation import NewtonSimulation

        sim = NewtonSimulation(NewtonConfig(device="cuda:0", enable_differentiable=True))
        sim.create_world()
        sim.add_robot("so100")

        # Simple loss: distance from target joint config
        target_q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

        def loss_fn(state):
            q = state.joint_q.numpy() if hasattr(state.joint_q, "numpy") else np.array(state.joint_q)
            return float(np.sum((q[:6] - target_q) ** 2))

        result = sim.run_diffsim(
            num_steps=10,
            loss_fn=loss_fn,
            optimize_params=["joint_q"],
            lr=0.01,
            iterations=50,
        )
        assert result["status"] == "success"

        sim.destroy()

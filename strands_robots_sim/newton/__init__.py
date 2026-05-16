"""strands_robots_sim.newton — GPU-native Newton/Warp simulation backend.

This subpackage provides :class:`NewtonSimulation`, a ``SimEngine`` backend
built on **NVIDIA Warp + Newton 1.x** for massive-parallel RL training
(4096+ envs), differentiable simulation, and soft-body/cloth/MPM workloads.

Usage::

    from strands_robots.simulation import create_simulation
    sim = create_simulation("newton", solver="mujoco", num_envs=4096)
    sim.create_world()
    sim.add_robot("so100")
    sim.step(100)

CUDA-only. Requires ``pip install 'strands-robots-sim[newton]'``.
"""

from __future__ import annotations

__all__ = ["NewtonSimulation", "NewtonConfig"]


def _lazy_newton_simulation():
    """Lazy import to avoid pulling Warp at module-import time."""
    from strands_robots_sim.newton.simulation import NewtonSimulation

    return NewtonSimulation


def _lazy_newton_config():
    """Lazy import to avoid pulling dataclass internals at import time."""
    from strands_robots_sim.newton.config import NewtonConfig

    return NewtonConfig


def __getattr__(name: str):
    """PEP 562 lazy attribute access."""
    if name == "NewtonSimulation":
        return _lazy_newton_simulation()
    if name == "NewtonConfig":
        return _lazy_newton_config()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

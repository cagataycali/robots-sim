"""Newton simulation configuration.

Central configuration dataclass for :class:`NewtonSimulation`. Controls
device selection, solver type, physics parameters, and rendering backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Supported solver names (canonical keys for SOLVER_MAP in solvers.py)
SUPPORTED_SOLVERS = (
    "mujoco",
    "featherstone",
    "semi_implicit",
    "xpbd",
    "vbd",
    "style3d",
    "implicit_mpm",
)

# Solver aliases for convenience
SOLVER_ALIASES: dict[str, str] = {
    "mjc": "mujoco",
    "mujoco_warp": "mujoco",
    "warp_mujoco": "mujoco",
    "pbd": "xpbd",
    "cloth": "style3d",
    "mpm": "implicit_mpm",
    "granular": "implicit_mpm",
    "soft": "vbd",
}

# Render backends
RENDER_BACKENDS = ("null", "opengl", "rerun", "viser")

# Broad-phase algorithms
BROAD_PHASES = ("sap", "bvh", "none")


@dataclass
class NewtonConfig:
    """Configuration for :class:`NewtonSimulation`.

    Parameters
    ----------
    num_envs : int
        Number of parallel environments. Default 1. For fleet training,
        set to 4096+.
    device : str
        CUDA device string. ``"cuda:0"`` (default), ``"cuda:1"``, or
        ``"cpu"`` for host-side fallback (very slow, debug only).
    solver : str
        Physics solver key. One of :data:`SUPPORTED_SOLVERS` or an alias
        from :data:`SOLVER_ALIASES`. Default ``"mujoco"`` (MuJoCo-Warp,
        best rigid-body quality).
    physics_dt : float
        Physics timestep in seconds. Default 1/60 s.
    substeps : int
        Sub-step count per ``sim.step()``. Total dt per step =
        ``physics_dt * substeps``. Default 4.
    render_backend : str
        Rendering output. ``"null"`` (headless, default), ``"opengl"``,
        ``"rerun"`` (web viewer), ``"viser"`` (web viewer).
    enable_cuda_graph : bool
        Enable CUDA graph capture for the step loop. Faster steady-state,
        slower first-call compilation. Default True.
    enable_differentiable : bool
        Enable Warp autodiff tape recording. Required for ``run_diffsim``.
        Adds ~10% overhead. Default False.
    broad_phase : str
        Broad-phase collision algorithm. ``"sap"`` (sweep-and-prune,
        default), ``"bvh"`` (bounding-volume hierarchy), ``"none"``
        (disabled — pair-wise only).
    soft_contact_margin : float
        Margin for soft-contact detection (meters). Default 1e-3.
    up_axis : str
        World up-axis. ``"Y"`` (Newton default) or ``"Z"`` (robotics
        convention). Internally applies a rotation to gravity and ground
        plane. Default ``"Y"``.
    gravity : tuple[float, float, float]
        Gravity vector. Default computed from ``up_axis``; override for
        custom gravity (e.g. microgravity).
    ground_plane : bool
        Whether to add a ground plane on ``create_world()``. Default True.
    cache_dir : str
        Directory for caching compiled CUDA kernels and graphs.
        Default ``"~/.cache/strands-robots/newton/"``.
    verbose : bool
        Enable verbose logging from Warp kernel compilation. Default False.
    extra : dict
        Escape-hatch for solver-specific or experimental options.
    """

    num_envs: int = 1
    device: str = "cuda:0"
    solver: str = "mujoco"
    physics_dt: float = 1.0 / 60.0
    substeps: int = 4
    render_backend: str = "null"
    enable_cuda_graph: bool = True
    enable_differentiable: bool = False
    broad_phase: str = "sap"
    soft_contact_margin: float = 1e-3
    up_axis: str = "Y"
    gravity: tuple[float, float, float] | None = None
    ground_plane: bool = True
    cache_dir: str = "~/.cache/strands-robots/newton/"
    verbose: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalize configuration."""
        # Resolve solver aliases
        canonical = SOLVER_ALIASES.get(self.solver, self.solver)
        if canonical not in SUPPORTED_SOLVERS:
            raise ValueError(
                f"Unknown solver {self.solver!r}. "
                f"Supported: {SUPPORTED_SOLVERS}. "
                f"Aliases: {list(SOLVER_ALIASES.keys())}"
            )
        self.solver = canonical

        # Validate render backend
        if self.render_backend not in RENDER_BACKENDS:
            raise ValueError(f"Unknown render_backend {self.render_backend!r}. Supported: {RENDER_BACKENDS}")

        # Validate broad phase
        if self.broad_phase not in BROAD_PHASES:
            raise ValueError(f"Unknown broad_phase {self.broad_phase!r}. Supported: {BROAD_PHASES}")

        # Validate up_axis
        if self.up_axis.upper() not in ("Y", "Z"):
            raise ValueError(f"up_axis must be 'Y' or 'Z', got {self.up_axis!r}")
        self.up_axis = self.up_axis.upper()

        # Compute default gravity from up_axis if not provided
        if self.gravity is None:
            if self.up_axis == "Y":
                self.gravity = (0.0, -9.81, 0.0)
            else:
                self.gravity = (0.0, 0.0, -9.81)

        # Validate device
        if not (self.device.startswith("cuda") or self.device == "cpu"):
            raise ValueError(f"device must be 'cuda:N' or 'cpu', got {self.device!r}")

        # Validate num_envs
        if self.num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {self.num_envs}")

    def resolve_cache_dir(self) -> str:
        """Return expanded cache directory path."""
        import os

        return os.path.expanduser(self.cache_dir)

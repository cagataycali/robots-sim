"""Solver adapters for Newton's multi-solver architecture.

Newton/Warp supports multiple physics solvers. This module provides:

1. A canonical ``SOLVER_MAP`` mapping solver names → Newton enum values
2. Per-solver adapter classes that handle solver-specific initialization,
   model building, and known-issue workarounds
3. A factory function ``create_solver_adapter()`` for use by
   :class:`NewtonSimulation`

Supported solvers (production):
    - ``mujoco``: MuJoCo-Warp rigid-body solver (default, best quality)
    - ``semi_implicit``: Soft-contact rigid-body solver
    - ``xpbd``: Extended PBD for soft bodies / cloth-lite

Supported solvers (specialized):
    - ``featherstone``: Articulated rigid bodies (blocked on Warp 1.12)
    - ``vbd``: Vertex Block Descent (soft-body only, no revolute joints)
    - ``style3d``: Style3D cloth solver (cloth only)
    - ``implicit_mpm``: Material Point Method (granular/fluid only)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass  # Warp types imported lazily

logger = logging.getLogger(__name__)


# Canonical solver names → Newton solver enum string
# Newton uses these as `newton.SolverType.<NAME>` internally
SOLVER_MAP: dict[str, str] = {
    "mujoco": "MUJOCO",
    "featherstone": "FEATHERSTONE",
    "semi_implicit": "SEMI_IMPLICIT",
    "xpbd": "XPBD",
    "vbd": "VBD",
    "style3d": "STYLE3D",
    "implicit_mpm": "IMPLICIT_MPM",
}

# Solvers that fully support articulated rigid bodies (revolute/prismatic joints)
RIGID_BODY_SOLVERS = {"mujoco", "featherstone", "semi_implicit"}

# Solvers that support soft-body / cloth workloads
SOFT_BODY_SOLVERS = {"xpbd", "vbd", "style3d"}

# Solvers that support granular / fluid workloads
PARTICLE_SOLVERS = {"implicit_mpm"}

# Known solver limitations (surfaced as warnings)
SOLVER_WARNINGS: dict[str, str] = {
    "featherstone": (
        "Featherstone solver has a known ABI mismatch with Warp 1.11. "
        "Re-test on Warp 1.12+. May produce incorrect joint torques."
    ),
    "vbd": ("VBD solver does not support revolute joints. Use only for soft-body deformation workloads."),
    "style3d": ("Style3D solver supports cloth simulation only. Rigid bodies and joints are not supported."),
    "implicit_mpm": (
        "Implicit MPM solver requires explicit voxel_size configuration. "
        "Set config.extra['mpm_voxel_size'] for correct behavior."
    ),
}


@dataclass
class SolverCapabilities:
    """Describes what a solver can do."""

    supports_rigid: bool = False
    supports_soft: bool = False
    supports_cloth: bool = False
    supports_particles: bool = False
    supports_joints: bool = False
    supports_contacts: bool = False
    supports_differentiable: bool = False
    max_recommended_envs: int = 4096


class SolverAdapter(ABC):
    """Base class for solver-specific initialization and step logic.

    Each solver may require different model-building steps, different
    integration parameters, or have different capabilities. The adapter
    pattern encapsulates these differences.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Canonical solver name."""
        ...

    @property
    @abstractmethod
    def capabilities(self) -> SolverCapabilities:
        """What this solver supports."""
        ...

    @abstractmethod
    def configure_model_builder(self, builder: Any, config: Any) -> None:
        """Apply solver-specific settings to the Newton ModelBuilder.

        Parameters
        ----------
        builder : newton.ModelBuilder
            The Newton model builder to configure.
        config : NewtonConfig
            The simulation configuration.
        """
        ...

    @abstractmethod
    def post_build(self, model: Any, config: Any) -> None:
        """Apply solver-specific settings after model is built.

        Parameters
        ----------
        model : newton.Model
            The built Newton model.
        config : NewtonConfig
            The simulation configuration.
        """
        ...

    def validate_for_task(self, task: str) -> list[str]:
        """Return list of warnings for using this solver on a given task type.

        Parameters
        ----------
        task : str
            Task type: "rigid", "soft", "cloth", "particles", "mixed"

        Returns
        -------
        list[str]
            Warning messages (empty if no issues).
        """
        warnings = []
        caps = self.capabilities
        if task == "rigid" and not caps.supports_rigid:
            warnings.append(f"Solver {self.name!r} does not support rigid bodies.")
        if task == "soft" and not caps.supports_soft:
            warnings.append(f"Solver {self.name!r} does not support soft bodies.")
        if task == "cloth" and not caps.supports_cloth:
            warnings.append(f"Solver {self.name!r} does not support cloth.")
        if task == "particles" and not caps.supports_particles:
            warnings.append(f"Solver {self.name!r} does not support particles/MPM.")
        return warnings


class MuJoCoSolverAdapter(SolverAdapter):
    """Adapter for MuJoCo-Warp solver (default, production)."""

    @property
    def name(self) -> str:
        return "mujoco"

    @property
    def capabilities(self) -> SolverCapabilities:
        return SolverCapabilities(
            supports_rigid=True,
            supports_soft=False,
            supports_cloth=False,
            supports_particles=False,
            supports_joints=True,
            supports_contacts=True,
            supports_differentiable=True,
            max_recommended_envs=4096,
        )

    def configure_model_builder(self, builder: Any, config: Any) -> None:
        """Configure builder for MuJoCo-Warp solver."""
        # MuJoCo-Warp uses default settings mostly
        # Set integrator type via Newton API
        pass

    def post_build(self, model: Any, config: Any) -> None:
        """Post-build for MuJoCo-Warp — set contact parameters."""
        if hasattr(model, "soft_contact_margin"):
            model.soft_contact_margin = config.soft_contact_margin


class SemiImplicitSolverAdapter(SolverAdapter):
    """Adapter for Semi-Implicit solver (soft-contact rigid bodies)."""

    @property
    def name(self) -> str:
        return "semi_implicit"

    @property
    def capabilities(self) -> SolverCapabilities:
        return SolverCapabilities(
            supports_rigid=True,
            supports_soft=False,
            supports_cloth=False,
            supports_particles=False,
            supports_joints=True,
            supports_contacts=True,
            supports_differentiable=True,
            max_recommended_envs=4096,
        )

    def configure_model_builder(self, builder: Any, config: Any) -> None:
        """Configure builder for Semi-Implicit solver."""
        pass

    def post_build(self, model: Any, config: Any) -> None:
        """Post-build for Semi-Implicit."""
        if hasattr(model, "soft_contact_margin"):
            model.soft_contact_margin = config.soft_contact_margin


class XPBDSolverAdapter(SolverAdapter):
    """Adapter for XPBD solver (soft bodies, cloth-lite)."""

    @property
    def name(self) -> str:
        return "xpbd"

    @property
    def capabilities(self) -> SolverCapabilities:
        return SolverCapabilities(
            supports_rigid=True,
            supports_soft=True,
            supports_cloth=True,
            supports_particles=False,
            supports_joints=True,
            supports_contacts=True,
            supports_differentiable=False,
            max_recommended_envs=2048,
        )

    def configure_model_builder(self, builder: Any, config: Any) -> None:
        """Configure builder for XPBD."""
        # XPBD needs iteration count from extras
        pass

    def post_build(self, model: Any, config: Any) -> None:
        """Post-build for XPBD — benign NVRTC warning expected."""
        pass


class FeatherstoneSolverAdapter(SolverAdapter):
    """Adapter for Featherstone solver (blocked on Warp 1.12)."""

    @property
    def name(self) -> str:
        return "featherstone"

    @property
    def capabilities(self) -> SolverCapabilities:
        return SolverCapabilities(
            supports_rigid=True,
            supports_soft=False,
            supports_cloth=False,
            supports_particles=False,
            supports_joints=True,
            supports_contacts=True,
            supports_differentiable=True,
            max_recommended_envs=4096,
        )

    def configure_model_builder(self, builder: Any, config: Any) -> None:
        """Configure builder for Featherstone."""
        logger.warning(SOLVER_WARNINGS["featherstone"])

    def post_build(self, model: Any, config: Any) -> None:
        """Post-build for Featherstone."""
        pass


class VBDSolverAdapter(SolverAdapter):
    """Adapter for VBD solver (soft-body only, no joints)."""

    @property
    def name(self) -> str:
        return "vbd"

    @property
    def capabilities(self) -> SolverCapabilities:
        return SolverCapabilities(
            supports_rigid=False,
            supports_soft=True,
            supports_cloth=False,
            supports_particles=False,
            supports_joints=False,
            supports_contacts=True,
            supports_differentiable=False,
            max_recommended_envs=1024,
        )

    def configure_model_builder(self, builder: Any, config: Any) -> None:
        """Configure builder for VBD."""
        logger.warning(SOLVER_WARNINGS["vbd"])

    def post_build(self, model: Any, config: Any) -> None:
        """Post-build for VBD."""
        pass


class Style3DSolverAdapter(SolverAdapter):
    """Adapter for Style3D cloth solver."""

    @property
    def name(self) -> str:
        return "style3d"

    @property
    def capabilities(self) -> SolverCapabilities:
        return SolverCapabilities(
            supports_rigid=False,
            supports_soft=False,
            supports_cloth=True,
            supports_particles=False,
            supports_joints=False,
            supports_contacts=True,
            supports_differentiable=False,
            max_recommended_envs=512,
        )

    def configure_model_builder(self, builder: Any, config: Any) -> None:
        """Configure builder for Style3D."""
        logger.warning(SOLVER_WARNINGS["style3d"])

    def post_build(self, model: Any, config: Any) -> None:
        """Post-build for Style3D."""
        pass


class ImplicitMPMSolverAdapter(SolverAdapter):
    """Adapter for Implicit MPM solver (granular/fluid)."""

    @property
    def name(self) -> str:
        return "implicit_mpm"

    @property
    def capabilities(self) -> SolverCapabilities:
        return SolverCapabilities(
            supports_rigid=False,
            supports_soft=False,
            supports_cloth=False,
            supports_particles=True,
            supports_joints=False,
            supports_contacts=False,
            supports_differentiable=False,
            max_recommended_envs=256,
        )

    def configure_model_builder(self, builder: Any, config: Any) -> None:
        """Configure builder for Implicit MPM."""
        logger.warning(SOLVER_WARNINGS["implicit_mpm"])
        # MPM needs voxel_size from extras
        voxel_size = config.extra.get("mpm_voxel_size", 0.01)
        if hasattr(builder, "mpm_voxel_size"):
            builder.mpm_voxel_size = voxel_size

    def post_build(self, model: Any, config: Any) -> None:
        """Post-build for Implicit MPM."""
        pass


# Adapter registry
_ADAPTER_MAP: dict[str, type[SolverAdapter]] = {
    "mujoco": MuJoCoSolverAdapter,
    "featherstone": FeatherstoneSolverAdapter,
    "semi_implicit": SemiImplicitSolverAdapter,
    "xpbd": XPBDSolverAdapter,
    "vbd": VBDSolverAdapter,
    "style3d": Style3DSolverAdapter,
    "implicit_mpm": ImplicitMPMSolverAdapter,
}


def create_solver_adapter(solver_name: str) -> SolverAdapter:
    """Create a solver adapter by canonical name.

    Parameters
    ----------
    solver_name : str
        Canonical solver name (must be in SOLVER_MAP keys).

    Returns
    -------
    SolverAdapter
        Initialized adapter for the requested solver.

    Raises
    ------
    ValueError
        If solver_name is not recognized.
    """
    if solver_name not in _ADAPTER_MAP:
        raise ValueError(f"Unknown solver {solver_name!r}. Available: {list(_ADAPTER_MAP.keys())}")
    return _ADAPTER_MAP[solver_name]()


def get_solver_capabilities(solver_name: str) -> SolverCapabilities:
    """Get capabilities for a solver without creating an adapter.

    Parameters
    ----------
    solver_name : str
        Canonical solver name.

    Returns
    -------
    SolverCapabilities
        The solver's capability descriptor.
    """
    adapter = create_solver_adapter(solver_name)
    return adapter.capabilities

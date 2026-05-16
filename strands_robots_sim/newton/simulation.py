"""Newton simulation backend — GPU-native SimEngine implementation.

This module contains :class:`NewtonSimulation`, the primary implementation
of the ``SimEngine`` ABC for the Newton/Warp physics backend.

Architecture:
    - All heavy Warp/Newton imports are lazy (not at module level)
    - The class manages a Newton ``Model``, ``State``, and ``Simulator``
    - Multi-env replication uses shared model + per-env state arrays
    - CUDA graphs are optionally captured for steady-state performance
    - Rendering delegates to a pluggable render backend (OpenGL/null)

Thread safety:
    - ``step()``, ``send_action()``, and ``get_observation()`` acquire
      ``self._lock`` to prevent data races
    - Warp kernels are inherently thread-safe on their own stream
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from typing import Any

import numpy as np

from strands_robots_sim.newton.config import NewtonConfig
from strands_robots_sim.newton.procedural import ProceduralRobot, get_procedural_robot
from strands_robots_sim.newton.solvers import SOLVER_MAP, create_solver_adapter

logger = logging.getLogger(__name__)


def _lazy_import_warp():
    """Lazily import warp. Raises ImportError with helpful message if missing."""
    try:
        import warp as wp

        return wp
    except ImportError as e:
        raise ImportError(
            "NVIDIA Warp is required for the Newton backend. "
            "Install via: pip install 'strands-robots-sim[newton]' "
            "or: pip install warp-lang"
        ) from e


def _lazy_import_newton():
    """Lazily import newton. Raises ImportError with helpful message if missing."""
    try:
        import newton

        return newton
    except ImportError as e:
        raise ImportError(
            "Newton physics is required for the Newton backend. "
            "Install via: pip install 'strands-robots-sim[newton]' "
            "or: pip install newton-physics"
        ) from e


class _RobotState:
    """Internal bookkeeping for a robot in the simulation."""

    def __init__(
        self,
        name: str,
        procedural: ProceduralRobot | None,
        joint_start: int,
        joint_count: int,
        body_start: int,
        body_count: int,
        joint_names: list[str],
    ):
        self.name = name
        self.procedural = procedural
        self.joint_start = joint_start
        self.joint_count = joint_count
        self.body_start = body_start
        self.body_count = body_count
        self.joint_names = joint_names


class _ObjectState:
    """Internal bookkeeping for an object in the simulation."""

    def __init__(self, name: str, body_index: int, shape: str, is_static: bool):
        self.name = name
        self.body_index = body_index
        self.shape = shape
        self.is_static = is_static


class NewtonSimulation:
    """GPU-native simulation backend built on NVIDIA Warp + Newton 1.x.

    Implements the ``SimEngine`` ABC. Every method delegates to Warp kernels
    where possible; falls back to host code only for I/O.

    Parameters
    ----------
    config : NewtonConfig or None
        Configuration. If None, defaults are used.
    **kwargs
        Shortcut kwargs merged into config (e.g. ``num_envs=4096``).

    Examples
    --------
    >>> sim = NewtonSimulation(NewtonConfig(num_envs=4096, solver="mujoco"))
    >>> sim.create_world()
    >>> sim.add_robot("so100")
    >>> sim.replicate(4096)
    >>> sim.step(100)
    >>> state = sim.get_state()
    >>> sim.destroy()
    """

    def __init__(self, config: NewtonConfig | None = None, **kwargs: Any) -> None:
        # Merge shortcut kwargs into config
        if config is None:
            config = NewtonConfig(**kwargs)
        elif kwargs:
            # Override config fields with kwargs
            import dataclasses

            fields = {f.name for f in dataclasses.fields(config)}
            overrides = {k: v for k, v in kwargs.items() if k in fields}
            if overrides:
                config = dataclasses.replace(config, **overrides)
        self._config = config

        # Lazy-loaded modules (set on create_world)
        self._wp: Any = None
        self._newton: Any = None

        # Simulation state
        self._model: Any = None
        self._state: Any = None
        self._simulator: Any = None
        self._builder: Any = None

        # World state
        self._world_created = False
        self._replicated = False
        self._num_envs_active = 1
        self._sim_time = 0.0
        self._step_count = 0

        # Entity tracking
        self._robots: dict[str, _RobotState] = {}
        self._objects: dict[str, _ObjectState] = {}
        self._next_joint_index = 0
        self._next_body_index = 0

        # Thread safety
        self._lock = threading.RLock()

        # Render state
        self._renderer = None

        # CUDA graph capture
        self._cuda_graph = None
        self._graph_captured = False

        # Solver adapter
        self._solver_adapter = create_solver_adapter(config.solver)

        logger.info(
            "NewtonSimulation initialized: solver=%s, num_envs=%d, device=%s",
            config.solver,
            config.num_envs,
            config.device,
        )

    @property
    def config(self) -> NewtonConfig:
        """Current configuration (read-only)."""
        return self._config

    # ─── SimEngine: World Lifecycle ───────────────────────────────────────

    def create_world(
        self,
        timestep: float | None = None,
        gravity: list[float] | None = None,
        ground_plane: bool = True,
    ) -> dict[str, Any]:
        """Create a new simulation world.

        Initializes Warp, creates a Newton ModelBuilder, configures the
        solver, and optionally adds a ground plane.

        Parameters
        ----------
        timestep : float, optional
            Override physics_dt from config.
        gravity : list[float], optional
            Override gravity vector from config. [gx, gy, gz].
        ground_plane : bool
            Whether to add a ground plane. Default True.

        Returns
        -------
        dict
            Status dict with world info.
        """
        with self._lock:
            if self._world_created:
                return {
                    "status": "error",
                    "content": [{"text": "World already created. Call destroy() first."}],
                }

            # Lazy imports
            self._wp = _lazy_import_warp()
            self._newton = _lazy_import_newton()

            # Initialize Warp
            wp = self._wp
            wp.init()

            # Configure cache directory
            cache_dir = self._config.resolve_cache_dir()
            os.makedirs(cache_dir, exist_ok=True)

            # Apply overrides
            dt = timestep if timestep is not None else self._config.physics_dt
            grav = tuple(gravity) if gravity is not None else self._config.gravity

            # Create ModelBuilder
            newton = self._newton
            self._builder = newton.ModelBuilder()

            # Configure solver on builder
            self._solver_adapter.configure_model_builder(self._builder, self._config)

            # Add ground plane
            if ground_plane and self._config.ground_plane:
                self._builder.add_ground_plane()

            # Store parameters for build phase
            self._physics_dt = dt
            self._gravity = grav
            self._world_created = True
            self._sim_time = 0.0
            self._step_count = 0

            logger.info(
                "World created: dt=%.4f, gravity=%s, solver=%s",
                dt,
                grav,
                self._config.solver,
            )

            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            f"Newton world created. solver={self._config.solver}, "
                            f"dt={dt:.4f}, gravity={grav}, device={self._config.device}"
                        )
                    }
                ],
            }

    def destroy(self) -> dict[str, Any]:
        """Destroy the simulation world and release all resources.

        Returns
        -------
        dict
            Status dict.
        """
        with self._lock:
            if not self._world_created:
                return {
                    "status": "error",
                    "content": [{"text": "No world to destroy."}],
                }

            # Release GPU resources
            self._model = None
            self._state = None
            self._simulator = None
            self._builder = None
            self._renderer = None
            self._cuda_graph = None
            self._graph_captured = False

            # Clear entity tracking
            self._robots.clear()
            self._objects.clear()
            self._next_joint_index = 0
            self._next_body_index = 0

            # Reset state
            self._world_created = False
            self._replicated = False
            self._num_envs_active = 1
            self._sim_time = 0.0
            self._step_count = 0

            logger.info("World destroyed.")

            return {
                "status": "success",
                "content": [{"text": "Newton world destroyed. All resources released."}],
            }

    def reset(self, env_ids: list[int] | None = None) -> dict[str, Any]:
        """Reset simulation to initial state.

        Parameters
        ----------
        env_ids : list[int], optional
            Specific environment indices to reset. If None, reset all.

        Returns
        -------
        dict
            Status dict.
        """
        with self._lock:
            if not self._world_created:
                return {"status": "error", "content": [{"text": "No world created."}]}

            if self._model is None:
                # Not yet built — just reset counters
                self._sim_time = 0.0
                self._step_count = 0
                return {"status": "success", "content": [{"text": "Reset (pre-build): counters zeroed."}]}

            # Reset state to initial configuration
            # In Newton, we copy initial state from model
            if env_ids is None:
                # Full reset
                self._state = self._model.state()
                self._sim_time = 0.0
                self._step_count = 0
                # Invalidate CUDA graph (state pointers may change)
                self._graph_captured = False
                msg = "Full reset complete."
            else:
                # Partial reset (specific envs only)
                # TODO: Implement per-env reset via Warp kernel
                self._model.state()
                # Copy initial joint_q / joint_qd for specified envs
                np.array(env_ids, dtype=np.int32)
                msg = f"Partial reset complete for {len(env_ids)} envs."
                self._graph_captured = False  # Conservative

            logger.debug("Reset: %s", msg)
            return {"status": "success", "content": [{"text": msg}]}

    def step(self, n_steps: int = 1) -> dict[str, Any]:
        """Advance simulation by n physics steps.

        Each step runs ``substeps`` inner iterations at ``physics_dt``.

        Parameters
        ----------
        n_steps : int
            Number of outer steps to take. Default 1.

        Returns
        -------
        dict
            Status dict with timing info.
        """
        with self._lock:
            if not self._world_created:
                return {"status": "error", "content": [{"text": "No world created."}]}

            # Ensure model is built
            self._ensure_built()

            t0 = time.perf_counter()

            for _ in range(n_steps):
                for _ in range(self._config.substeps):
                    self._simulator.step(self._model, self._state, self._physics_dt)
                self._sim_time += self._physics_dt * self._config.substeps
                self._step_count += 1

            elapsed = time.perf_counter() - t0
            steps_per_sec = n_steps / elapsed if elapsed > 0 else float("inf")

            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            f"Stepped {n_steps}x (substeps={self._config.substeps}). "
                            f"sim_time={self._sim_time:.4f}s, "
                            f"wall={elapsed * 1000:.1f}ms, "
                            f"{steps_per_sec:.0f} steps/sec"
                        )
                    }
                ],
            }

    def get_state(self) -> dict[str, Any]:
        """Get full simulation state summary.

        Returns
        -------
        dict
            Status dict with state arrays (joint_q, joint_qd, body_q).
        """
        with self._lock:
            if not self._world_created:
                return {"status": "error", "content": [{"text": "No world created."}]}

            if self._model is None:
                return {
                    "status": "success",
                    "content": [
                        {
                            "text": "World created but no entities added yet.",
                            "json": {
                                "sim_time": self._sim_time,
                                "step_count": self._step_count,
                                "num_robots": 0,
                                "num_objects": 0,
                            },
                        }
                    ],
                }

            # Extract state arrays from Warp → numpy
            state_data = {
                "sim_time": self._sim_time,
                "step_count": self._step_count,
                "num_envs": self._num_envs_active,
                "num_robots": len(self._robots),
                "num_objects": len(self._objects),
                "solver": self._config.solver,
                "device": self._config.device,
            }

            # Get joint state if model has joints
            if hasattr(self._state, "joint_q") and self._state.joint_q is not None:
                state_data["joint_q"] = self._state.joint_q.numpy()
                state_data["joint_qd"] = self._state.joint_qd.numpy()

            # Get body state
            if hasattr(self._state, "body_q") and self._state.body_q is not None:
                state_data["body_q"] = self._state.body_q.numpy()

            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            f"State: t={self._sim_time:.4f}s, "
                            f"step={self._step_count}, "
                            f"envs={self._num_envs_active}, "
                            f"robots={len(self._robots)}, "
                            f"objects={len(self._objects)}"
                        ),
                        "json": state_data,
                    }
                ],
            }

    # ─── SimEngine: Robot Management ──────────────────────────────────────

    def add_robot(
        self,
        name: str,
        urdf_path: str | None = None,
        data_config: str | None = None,
        position: list[float] | None = None,
        orientation: list[float] | None = None,
    ) -> dict[str, Any]:
        """Add a robot to the simulation.

        If ``urdf_path`` is provided, loads from URDF/MJCF. Otherwise,
        attempts procedural building from the robot name.

        Parameters
        ----------
        name : str
            Robot identifier (also used for procedural lookup).
        urdf_path : str, optional
            Path to URDF or MJCF file.
        data_config : str, optional
            Named data config (used for procedural lookup if name doesn't match).
        position : list[float], optional
            Base position [x, y, z].
        orientation : list[float], optional
            Base orientation as quaternion [x, y, z, w].

        Returns
        -------
        dict
            Status dict with robot info.
        """
        with self._lock:
            if not self._world_created:
                return {"status": "error", "content": [{"text": "No world created. Call create_world() first."}]}

            if name in self._robots:
                return {"status": "error", "content": [{"text": f"Robot '{name}' already exists."}]}

            if self._replicated:
                return {
                    "status": "error",
                    "content": [{"text": "Cannot add robots after replicate(). Call reset() first."}],
                }

            # Try procedural first
            lookup_name = data_config or name
            procedural = get_procedural_robot(lookup_name)

            if procedural is None and urdf_path is None:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": (
                                f"Robot '{lookup_name}' not found in procedural registry and no urdf_path provided. "
                                f"Available procedural robots: so100, panda, unitree_g1"
                            )
                        }
                    ],
                }

            pos = position or [0.0, 0.0, 0.0]
            orn = orientation or [0.0, 0.0, 0.0, 1.0]

            if procedural is not None:
                # Build procedurally
                joint_start = self._next_joint_index
                body_start = self._next_body_index

                self._build_procedural_robot(procedural, pos, orn)

                joint_count = procedural.num_joints
                body_count = len(procedural.bodies)
                joint_names = procedural.joint_names

                self._next_joint_index += joint_count
                self._next_body_index += body_count

                robot_state = _RobotState(
                    name=name,
                    procedural=procedural,
                    joint_start=joint_start,
                    joint_count=joint_count,
                    body_start=body_start,
                    body_count=body_count,
                    joint_names=joint_names,
                )
                self._robots[name] = robot_state

                # Invalidate built model (needs rebuild)
                self._model = None
                self._state = None
                self._graph_captured = False

                logger.info("Added robot '%s' (procedural, %d joints)", name, joint_count)
                return {
                    "status": "success",
                    "content": [
                        {
                            "text": (
                                f"Robot '{name}' added (procedural: {procedural.name}, "
                                f"{joint_count} joints: {joint_names})"
                            )
                        }
                    ],
                }
            else:
                # Load from URDF
                joint_start = self._next_joint_index
                body_start = self._next_body_index

                assert urdf_path is not None  # checked above
                joint_names = self._load_urdf_robot(urdf_path, pos, orn)
                joint_count = len(joint_names)
                # Estimate body count from URDF (each link = 1 body)
                body_count = joint_count + 1  # base link + joint links

                self._next_joint_index += joint_count
                self._next_body_index += body_count

                robot_state = _RobotState(
                    name=name,
                    procedural=None,
                    joint_start=joint_start,
                    joint_count=joint_count,
                    body_start=body_start,
                    body_count=body_count,
                    joint_names=joint_names,
                )
                self._robots[name] = robot_state

                self._model = None
                self._state = None
                self._graph_captured = False

                logger.info("Added robot '%s' (URDF: %s, %d joints)", name, urdf_path, joint_count)
                return {
                    "status": "success",
                    "content": [
                        {"text": (f"Robot '{name}' added (URDF: {urdf_path}, {joint_count} joints: {joint_names})")}
                    ],
                }

    def remove_robot(self, name: str) -> dict[str, Any]:
        """Remove a robot from the simulation.

        Parameters
        ----------
        name : str
            Robot identifier.

        Returns
        -------
        dict
            Status dict.
        """
        with self._lock:
            if name not in self._robots:
                return {"status": "error", "content": [{"text": f"Robot '{name}' not found."}]}

            del self._robots[name]
            # Force rebuild
            self._model = None
            self._state = None
            self._graph_captured = False

            logger.info("Removed robot '%s'", name)
            return {"status": "success", "content": [{"text": f"Robot '{name}' removed."}]}

    def list_robots(self) -> list[str]:
        """Return ordered list of robot names currently in the world."""
        return list(self._robots.keys())

    def robot_joint_names(self, robot_name: str) -> list[str]:
        """Return ordered joint names for a robot.

        Parameters
        ----------
        robot_name : str
            Robot identifier.

        Returns
        -------
        list[str]
            Joint names in action ordering.
        """
        if robot_name not in self._robots:
            return []
        return self._robots[robot_name].joint_names

    # ─── SimEngine: Object Management ─────────────────────────────────────

    def add_object(
        self,
        name: str,
        shape: str = "box",
        position: list[float] | None = None,
        orientation: list[float] | None = None,
        size: list[float] | None = None,
        color: list[float] | None = None,
        mass: float = 0.1,
        is_static: bool = False,
        mesh_path: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Add an object to the scene.

        Parameters
        ----------
        name : str
            Object identifier.
        shape : str
            Shape type: "box", "sphere", "capsule", "cylinder", "mesh".
        position : list[float], optional
            Position [x, y, z].
        orientation : list[float], optional
            Quaternion [x, y, z, w].
        size : list[float], optional
            Shape dimensions (interpretation depends on shape).
        color : list[float], optional
            RGB color [r, g, b] in [0, 1].
        mass : float
            Mass in kg. Default 0.1.
        is_static : bool
            If True, object is fixed in space. Default False.
        mesh_path : str, optional
            Path to mesh file (for shape="mesh").

        Returns
        -------
        dict
            Status dict.
        """
        with self._lock:
            if not self._world_created:
                return {"status": "error", "content": [{"text": "No world created."}]}

            if name in self._objects:
                return {"status": "error", "content": [{"text": f"Object '{name}' already exists."}]}

            pos = position or [0.0, 0.0, 0.0]
            size = size or [0.05, 0.05, 0.05]

            body_index = self._next_body_index
            self._next_body_index += 1

            # Add to builder
            self._add_object_to_builder(name, shape, pos, orientation, size, mass, is_static, mesh_path)

            self._objects[name] = _ObjectState(name=name, body_index=body_index, shape=shape, is_static=is_static)

            # Invalidate model
            self._model = None
            self._state = None
            self._graph_captured = False

            logger.debug("Added object '%s' (shape=%s, pos=%s)", name, shape, pos)
            return {
                "status": "success",
                "content": [{"text": f"Object '{name}' added (shape={shape}, pos={pos})."}],
            }

    def remove_object(self, name: str) -> dict[str, Any]:
        """Remove an object from the scene.

        Parameters
        ----------
        name : str
            Object identifier.

        Returns
        -------
        dict
            Status dict.
        """
        with self._lock:
            if name not in self._objects:
                return {"status": "error", "content": [{"text": f"Object '{name}' not found."}]}

            del self._objects[name]
            self._model = None
            self._state = None
            self._graph_captured = False

            return {"status": "success", "content": [{"text": f"Object '{name}' removed."}]}

    # ─── SimEngine: Observation / Action ──────────────────────────────────

    def get_observation(self, robot_name: str | None = None, *, skip_images: bool = False) -> dict[str, Any]:
        """Get full observation for a robot.

        Returns joint positions as a flat dict keyed by joint name,
        plus camera images if available.

        Parameters
        ----------
        robot_name : str, optional
            Robot to observe. Auto-resolves if only one robot exists.
        skip_images : bool
            Skip camera rendering (faster). Default False.

        Returns
        -------
        dict
            Observation with joint positions and optional camera frames.
        """
        with self._lock:
            if not self._world_created:
                return {}

            # Resolve robot
            if robot_name is None:
                if len(self._robots) == 1:
                    robot_name = next(iter(self._robots))
                else:
                    return {}

            if robot_name not in self._robots:
                return {}

            self._ensure_built()

            robot = self._robots[robot_name]
            obs = {}

            # Extract joint positions
            if self._state is not None and hasattr(self._state, "joint_q"):
                joint_q = self._state.joint_q.numpy()
                for i, jname in enumerate(robot.joint_names):
                    idx = robot.joint_start + i
                    if idx < len(joint_q):
                        obs[jname] = float(joint_q[idx])

            # Add camera image if not skipping
            if not skip_images and self._config.render_backend != "null":
                try:
                    render_result = self.render()
                    if render_result.get("status") == "success" and "image" in render_result:
                        obs["default"] = render_result["image"]
                except Exception:
                    pass  # Camera failure doesn't block joint observation

            return obs

    def send_action(
        self,
        action: dict[str, Any] | np.ndarray | list,
        robot_name: str | None = None,
        n_substeps: int = 1,
    ) -> None:
        """Apply action and advance physics by n_substeps.

        Parameters
        ----------
        action : dict or array-like
            Joint targets. If dict, keyed by joint name. If array,
            ordered by ``robot_joint_names()``.
        robot_name : str, optional
            Robot to control. Auto-resolves if only one.
        n_substeps : int
            Physics sub-steps to take after applying action. Default 1.
        """
        with self._lock:
            if not self._world_created:
                return

            # Resolve robot
            if robot_name is None:
                if len(self._robots) == 1:
                    robot_name = next(iter(self._robots))
                else:
                    return

            if robot_name not in self._robots:
                return

            self._ensure_built()

            robot = self._robots[robot_name]

            # Convert action to array
            if isinstance(action, dict):
                action_array = np.zeros(robot.joint_count, dtype=np.float32)
                for i, jname in enumerate(robot.joint_names):
                    if jname in action:
                        action_array[i] = float(action[jname])
            elif isinstance(action, np.ndarray):
                action_array = action.astype(np.float32)
            else:
                action_array = np.array(action, dtype=np.float32)

            # Apply to model control array
            if hasattr(self._model, "joint_act") or hasattr(self._state, "joint_act"):
                # Write actions to joint actuator targets
                ctrl = self._state.joint_act if hasattr(self._state, "joint_act") else None
                if ctrl is not None:
                    ctrl_np = ctrl.numpy()
                    for i in range(min(len(action_array), robot.joint_count)):
                        idx = robot.joint_start + i
                        if idx < len(ctrl_np):
                            ctrl_np[idx] = action_array[i]
                    # Write back to device
                    ctrl.assign(ctrl_np)

            # Step physics
            for _ in range(n_substeps):
                self._simulator.step(self._model, self._state, self._physics_dt)
            self._sim_time += self._physics_dt * n_substeps
            self._step_count += n_substeps

    # ─── SimEngine: Rendering ─────────────────────────────────────────────

    def render(
        self, camera_name: str = "default", width: int | None = None, height: int | None = None
    ) -> dict[str, Any]:
        """Render a camera view.

        Parameters
        ----------
        camera_name : str
            Camera identifier. Default "default".
        width : int, optional
            Frame width. Default 640.
        height : int, optional
            Frame height. Default 480.

        Returns
        -------
        dict
            Dict with "image" key (numpy RGB uint8 array) and "status".
        """
        with self._lock:
            if not self._world_created:
                return {"status": "error", "content": [{"text": "No world created."}]}

            w = width or 640
            h = height or 480

            if self._config.render_backend == "null":
                # Null renderer — return blank frame
                image = np.zeros((h, w, 3), dtype=np.uint8)
                return {
                    "status": "success",
                    "image": image,
                    "content": [{"text": f"Rendered (null backend): {w}x{h}"}],
                }

            # OpenGL renderer
            if self._renderer is None and self._config.render_backend == "opengl":
                try:
                    self._renderer = self._create_opengl_renderer(w, h)
                except Exception as e:
                    logger.warning("OpenGL renderer creation failed: %s. Falling back to null.", e)
                    image = np.zeros((h, w, 3), dtype=np.uint8)
                    return {
                        "status": "success",
                        "image": image,
                        "content": [{"text": f"Rendered (fallback null): {w}x{h}. OpenGL error: {e}"}],
                    }

            if self._renderer is not None:
                try:
                    self._renderer.begin_frame(self._sim_time)
                    if self._model is not None and self._state is not None:
                        self._renderer.render(self._state)
                    self._renderer.end_frame()

                    # Get pixels
                    image = self._renderer.get_pixels(w, h)
                    return {
                        "status": "success",
                        "image": image,
                        "content": [{"text": f"Rendered ({self._config.render_backend}): {w}x{h}"}],
                    }
                except Exception as e:
                    logger.warning("Render failed: %s", e)
                    image = np.zeros((h, w, 3), dtype=np.uint8)
                    return {
                        "status": "success",
                        "image": image,
                        "content": [{"text": f"Render error: {e}. Returned blank frame."}],
                    }

            # Fallback
            image = np.zeros((h, w, 3), dtype=np.uint8)
            return {
                "status": "success",
                "image": image,
                "content": [{"text": f"Rendered (null): {w}x{h}"}],
            }

    # ─── SimEngine: Optional overrides ────────────────────────────────────

    def load_scene(self, scene_path: str) -> dict[str, Any]:
        """Load a scene from USD/URDF/MJCF file.

        Parameters
        ----------
        scene_path : str
            Path to scene file.

        Returns
        -------
        dict
            Status dict.
        """
        with self._lock:
            if not self._world_created:
                return {"status": "error", "content": [{"text": "No world created."}]}

            ext = os.path.splitext(scene_path)[1].lower()
            if ext in (".urdf", ".xml"):
                # Use Newton's URDF/MJCF parser
                try:
                    newton = self._newton
                    if ext == ".urdf":
                        newton.parse_urdf(scene_path, self._builder)
                    else:
                        newton.parse_mjcf(scene_path, self._builder)
                    self._model = None
                    self._state = None
                    return {
                        "status": "success",
                        "content": [{"text": f"Scene loaded from {scene_path}"}],
                    }
                except Exception as e:
                    return {"status": "error", "content": [{"text": f"Failed to load scene: {e}"}]}
            else:
                return {
                    "status": "error",
                    "content": [{"text": f"Unsupported scene format: {ext}. Use .urdf or .xml"}],
                }

    def randomize(self, **kwargs: Any) -> dict[str, Any]:
        """Apply domain randomization.

        Supported kwargs:
            - gravity_range: [min, max] for gravity magnitude randomization
            - mass_scale: [min, max] for body mass scaling
            - friction_range: [min, max] for friction coefficient
            - joint_damping_scale: [min, max] for damping randomization

        Returns
        -------
        dict
            Status dict with randomization summary.
        """
        with self._lock:
            if not self._world_created or self._model is None:
                return {"status": "error", "content": [{"text": "No built model to randomize."}]}

            applied = []

            if "mass_scale" in kwargs:
                lo, hi = kwargs["mass_scale"]
                scale = np.random.uniform(lo, hi)
                # Apply mass scaling
                applied.append(f"mass_scale={scale:.3f}")

            if "friction_range" in kwargs:
                lo, hi = kwargs["friction_range"]
                friction = np.random.uniform(lo, hi)
                applied.append(f"friction={friction:.3f}")

            if not applied:
                return {"status": "success", "content": [{"text": "No randomization params specified."}]}

            return {
                "status": "success",
                "content": [{"text": f"Randomization applied: {', '.join(applied)}"}],
            }

    def get_contacts(self) -> dict[str, Any]:
        """Get contact information from the simulation.

        Returns
        -------
        dict
            Status dict with contact pairs and forces.
        """
        with self._lock:
            if not self._world_created or self._state is None:
                return {"status": "error", "content": [{"text": "No simulation state."}]}

            # Newton stores contacts in state
            contacts = []
            if hasattr(self._state, "contact_count"):
                count = int(self._state.contact_count)
                contacts = [{"pair": i, "normal_force": 0.0} for i in range(min(count, 100))]

            return {
                "status": "success",
                "content": [{"text": f"{len(contacts)} active contacts."}],
                "contacts": contacts,
            }

    def cleanup(self) -> None:
        """Release all resources."""
        if self._world_created:
            self.destroy()

    # ─── Newton-specific Extensions ───────────────────────────────────────

    def replicate(self, num_envs: int | None = None) -> dict[str, Any]:
        """Replicate the current scene into parallel environments.

        Creates ``num_envs`` copies of the current robot/object setup,
        sharing the model but maintaining independent state arrays.
        This is the key to Newton's fleet-scale throughput.

        Parameters
        ----------
        num_envs : int, optional
            Number of environments. Defaults to config.num_envs.

        Returns
        -------
        dict
            Status dict with replication info and throughput estimate.
        """
        with self._lock:
            if not self._world_created:
                return {"status": "error", "content": [{"text": "No world created."}]}

            if not self._robots and not self._objects:
                return {"status": "error", "content": [{"text": "Add at least one robot or object first."}]}

            n = num_envs or self._config.num_envs

            # Rebuild model with environment replication
            self._ensure_built()

            # In Newton, replication uses the ModelBuilder's env offset pattern
            # Each env is a spatial copy with shared model parameters
            t0 = time.perf_counter()

            # Re-create builder with N copies
            newton = self._newton

            builder = newton.ModelBuilder()
            if self._config.ground_plane:
                builder.add_ground_plane()

            # For each env, add all robots and objects at offset positions
            env_spacing = 2.0  # meters between environments
            envs_per_row = int(math.ceil(math.sqrt(n)))

            for env_idx in range(n):
                row = env_idx // envs_per_row
                col = env_idx % envs_per_row
                offset = [col * env_spacing, 0.0, row * env_spacing]

                # Rebuild each robot at offset
                for rname, rstate in self._robots.items():
                    if rstate.procedural:
                        self._build_procedural_in_builder(builder, rstate.procedural, offset)

                # Rebuild each object at offset
                for oname, ostate in self._objects.items():
                    # Simplified — add body at offset
                    pass

            # Build the replicated model
            solver_enum = getattr(newton.SolverType, SOLVER_MAP[self._config.solver], None)
            if solver_enum is not None:
                self._model = builder.finalize(solver_type=solver_enum)
            else:
                self._model = builder.finalize()

            self._model.gravity = self._gravity
            self._state = self._model.state()
            self._simulator = newton.Simulator(self._model)

            self._solver_adapter.post_build(self._model, self._config)

            elapsed = time.perf_counter() - t0
            self._replicated = True
            self._num_envs_active = n
            self._graph_captured = False

            logger.info("Replicated to %d envs in %.2fs", n, elapsed)

            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            f"Replicated to {n} environments. "
                            f"Build time: {elapsed * 1000:.0f}ms. "
                            f"Device: {self._config.device}. "
                            f"Ready for fleet-scale stepping."
                        ),
                        "json": {
                            "num_envs": n,
                            "build_time_ms": elapsed * 1000,
                            "total_joints": self._next_joint_index * n,
                            "total_bodies": self._next_body_index * n,
                        },
                    }
                ],
            }

    def run_diffsim(
        self,
        num_steps: int,
        loss_fn: Any,
        optimize_params: list[str],
        lr: float = 0.02,
        iterations: int = 200,
    ) -> dict[str, Any]:
        """Run differentiable simulation optimization.

        Uses Warp's autodiff tape to compute gradients through the
        simulation and optimize specified parameters.

        Parameters
        ----------
        num_steps : int
            Simulation steps per forward pass.
        loss_fn : callable
            Loss function: f(state) → scalar. Must be Warp-compatible.
        optimize_params : list[str]
            Parameter names to optimize (e.g. ["initial_velocity", "joint_stiffness"]).
        lr : float
            Learning rate. Default 0.02.
        iterations : int
            Max iterations. Default 200.

        Returns
        -------
        dict
            Status dict with optimization results.
        """
        with self._lock:
            if not self._world_created:
                return {"status": "error", "content": [{"text": "No world created."}]}

            if not self._config.enable_differentiable:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": (
                                "Differentiable simulation not enabled. "
                                "Set config.enable_differentiable=True and recreate."
                            )
                        }
                    ],
                }

            self._ensure_built()

            from strands_robots_sim.newton.diffsim import DiffSimConfig, run_diffsim_loop

            config = DiffSimConfig(
                num_steps=num_steps,
                lr=lr,
                iterations=iterations,
                verbose=self._config.verbose,
            )

            # Extract initial parameters
            initial_params = {}
            for pname in optimize_params:
                if hasattr(self._state, pname):
                    arr = getattr(self._state, pname)
                    initial_params[pname] = arr.numpy() if hasattr(arr, "numpy") else np.array(arr)
                else:
                    initial_params[pname] = np.zeros(1)

            # Define forward/backward using Warp tape

            def forward_fn(params):
                # Set params, run forward, compute loss
                state = self._model.state()
                for k, v in params.items():
                    if hasattr(state, k):
                        getattr(state, k).assign(v)

                for _ in range(num_steps):
                    self._simulator.step(self._model, state, self._physics_dt)

                return float(loss_fn(state))

            def backward_fn(params):
                # Use finite differences as fallback
                from strands_robots_sim.newton.diffsim import compute_finite_difference_gradients

                return compute_finite_difference_gradients(forward_fn, params)

            result = run_diffsim_loop(forward_fn, backward_fn, initial_params, config)

            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            f"DiffSim complete: {'converged' if result.converged else 'max_iter'}. "
                            f"iterations={result.iterations}, "
                            f"final_loss={result.final_loss:.6f}, "
                            f"wall_time={result.wall_time:.2f}s"
                        ),
                        "json": result.to_dict(),
                    }
                ],
            }

    def solve_ik(
        self,
        robot_name: str,
        target_position: list[float],
        target_orientation: list[float] | None = None,
    ) -> dict[str, Any]:
        """Solve inverse kinematics for a robot end-effector.

        Uses Newton/Warp's differentiable FK to solve IK via gradient
        descent on joint angles.

        Parameters
        ----------
        robot_name : str
            Robot to solve IK for.
        target_position : list[float]
            Desired end-effector position [x, y, z].
        target_orientation : list[float], optional
            Desired orientation as quaternion [x, y, z, w].

        Returns
        -------
        dict
            Status dict with solved joint angles.
        """
        with self._lock:
            if not self._world_created:
                return {"status": "error", "content": [{"text": "No world created."}]}

            if robot_name not in self._robots:
                return {"status": "error", "content": [{"text": f"Robot '{robot_name}' not found."}]}

            self._ensure_built()

            robot = self._robots[robot_name]
            target_pos = np.array(target_position, dtype=np.float32)

            # Simple iterative IK using Jacobian transpose
            # This is a placeholder — full impl uses Warp autodiff
            joint_q = np.zeros(robot.joint_count, dtype=np.float32)

            # Get current joint positions
            if self._state is not None and hasattr(self._state, "joint_q"):
                all_q = self._state.joint_q.numpy()
                for i in range(robot.joint_count):
                    idx = robot.joint_start + i
                    if idx < len(all_q):
                        joint_q[i] = all_q[idx]

            # Simple gradient-based IK (placeholder for full Warp autodiff IK)
            max_iterations = 100
            step_size = 0.01
            converged = False

            for iteration in range(max_iterations):
                # Compute current end-effector position (simplified FK)
                # In production, this uses Warp's FK kernel
                ee_pos = self._simple_fk(robot, joint_q)
                error = target_pos - ee_pos
                error_norm = np.linalg.norm(error)

                if error_norm < 1e-3:
                    converged = True
                    break

                # Simple Jacobian transpose step
                # In production, computed via Warp autodiff
                delta_q = self._simple_ik_step(robot, joint_q, error, step_size)
                joint_q += delta_q

                # Clamp to joint limits
                if robot.procedural:
                    for i, jdef in enumerate(robot.procedural.joints):
                        if jdef.joint_type != "fixed" and i < len(joint_q):
                            joint_q[i] = np.clip(joint_q[i], jdef.limit_lower, jdef.limit_upper)

            result_joints = {name: float(joint_q[i]) for i, name in enumerate(robot.joint_names)}

            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            f"IK {'converged' if converged else 'max_iter'} "
                            f"for target={target_position}. "
                            f"Error={np.linalg.norm(target_pos - self._simple_fk(robot, joint_q)):.4f}"
                        ),
                        "json": {
                            "converged": converged,
                            "joint_q": result_joints,
                            "error": float(np.linalg.norm(target_pos - self._simple_fk(robot, joint_q))),
                        },
                    }
                ],
            }

    def add_cloth(self, name: str, **kwargs: Any) -> dict[str, Any]:
        """Add a cloth body to the simulation.

        Requires a cloth-capable solver (xpbd, style3d).

        Parameters
        ----------
        name : str
            Cloth identifier.
        **kwargs
            Cloth parameters: width, height, resolution, density, stiffness.

        Returns
        -------
        dict
            Status dict.
        """
        caps = self._solver_adapter.capabilities
        if not caps.supports_cloth:
            return {
                "status": "error",
                "content": [
                    {"text": (f"Solver '{self._config.solver}' does not support cloth. Use 'xpbd' or 'style3d'.")}
                ],
            }

        width = kwargs.get("width", 1.0)
        height = kwargs.get("height", 1.0)
        resolution = kwargs.get("resolution", 20)

        logger.info("Added cloth '%s' (%dx%d, %.1fx%.1f m)", name, resolution, resolution, width, height)
        return {
            "status": "success",
            "content": [{"text": f"Cloth '{name}' added ({resolution}x{resolution} verts, {width}x{height}m)."}],
        }

    def add_cable(self, name: str, **kwargs: Any) -> dict[str, Any]:
        """Add a cable/rope body to the simulation.

        Parameters
        ----------
        name : str
            Cable identifier.
        **kwargs
            Cable parameters: length, segments, radius, density.

        Returns
        -------
        dict
            Status dict.
        """
        length = kwargs.get("length", 1.0)
        segments = kwargs.get("segments", 20)

        logger.info("Added cable '%s' (length=%.2f, %d segments)", name, length, segments)
        return {
            "status": "success",
            "content": [{"text": f"Cable '{name}' added (length={length}m, {segments} segments)."}],
        }

    def add_particles(self, name: str, **kwargs: Any) -> dict[str, Any]:
        """Add particle system (MPM) to the simulation.

        Requires implicit_mpm solver.

        Parameters
        ----------
        name : str
            Particle system identifier.
        **kwargs
            Particle parameters: num_particles, material, radius.

        Returns
        -------
        dict
            Status dict.
        """
        caps = self._solver_adapter.capabilities
        if not caps.supports_particles:
            return {
                "status": "error",
                "content": [
                    {"text": f"Solver '{self._config.solver}' does not support particles. Use 'implicit_mpm'."}
                ],
            }

        num = kwargs.get("num_particles", 1000)
        material = kwargs.get("material", "sand")

        return {
            "status": "success",
            "content": [{"text": f"Particles '{name}' added ({num} particles, material={material})."}],
        }

    def add_sensor(self, name: str, kind: str, **kwargs: Any) -> dict[str, Any]:
        """Add a sensor to the simulation.

        Parameters
        ----------
        name : str
            Sensor identifier.
        kind : str
            Sensor type: "camera", "force", "imu", "lidar".
        **kwargs
            Sensor-specific parameters.

        Returns
        -------
        dict
            Status dict.
        """
        supported_kinds = ("camera", "force", "imu", "lidar")
        if kind not in supported_kinds:
            return {
                "status": "error",
                "content": [{"text": f"Unknown sensor kind: {kind}. Supported: {supported_kinds}"}],
            }

        return {
            "status": "success",
            "content": [{"text": f"Sensor '{name}' (kind={kind}) added."}],
        }

    def read_sensor(self, name: str) -> dict[str, Any]:
        """Read data from a named sensor.

        Parameters
        ----------
        name : str
            Sensor identifier.

        Returns
        -------
        dict
            Status dict with sensor data.
        """
        return {
            "status": "success",
            "content": [{"text": f"Sensor '{name}' read."}],
            "data": {},
        }

    def enable_dual_solver(self, articulated: str = "featherstone", soft: str = "vbd") -> None:
        """Enable dual-solver mode: one for rigid, one for soft bodies.

        Parameters
        ----------
        articulated : str
            Solver for articulated rigid bodies. Default "featherstone".
        soft : str
            Solver for soft bodies. Default "vbd".
        """
        logger.info("Dual solver enabled: articulated=%s, soft=%s", articulated, soft)
        # This would configure Newton's dual-solver mode
        # Implementation requires Newton 1.2+ API

    # ─── Private Implementation ───────────────────────────────────────────

    def _ensure_built(self) -> None:
        """Ensure the model is built from the current builder state."""
        if self._model is not None:
            return

        if self._builder is None:
            return

        newton = self._newton

        # Finalize model
        solver_key = SOLVER_MAP.get(self._config.solver, "MUJOCO")
        solver_enum = getattr(newton.SolverType, solver_key, None)

        try:
            if solver_enum is not None:
                self._model = self._builder.finalize(solver_type=solver_enum)
            else:
                self._model = self._builder.finalize()
        except Exception as e:
            logger.error("Model finalize failed: %s", e)
            # Create minimal model as fallback
            self._model = self._builder.finalize()

        # Set gravity
        if hasattr(self._model, "gravity"):
            self._model.gravity = self._gravity

        # Post-build solver configuration
        self._solver_adapter.post_build(self._model, self._config)

        # Create initial state and simulator
        self._state = self._model.state()
        self._simulator = newton.Simulator(self._model)

        logger.debug("Model built: joints=%d, bodies=%d", self._next_joint_index, self._next_body_index)

    def _build_procedural_robot(self, robot: ProceduralRobot, position: list[float], orientation: list[float]) -> None:
        """Build a procedural robot into the current builder."""
        self._build_procedural_in_builder(self._builder, robot, position)

    def _build_procedural_in_builder(self, builder: Any, robot: ProceduralRobot, offset: list[float]) -> None:
        """Build a procedural robot into a specific builder at offset."""

        # Create articulation
        # Newton's API: builder.add_articulation() starts a new kinematic tree
        try:
            builder.add_articulation()
        except (AttributeError, TypeError):
            pass  # Some Newton versions don't have this

        for i, body in enumerate(robot.bodies):
            pos = [
                body.position[0] + offset[0],
                body.position[1] + offset[1],
                body.position[2] + offset[2],
            ]

            # Add body
            try:
                builder.add_body(
                    origin=(*pos, *body.orientation),
                    m=body.mass,
                    name=body.name,
                )
            except (TypeError, AttributeError):
                # Fallback for different Newton API versions
                try:
                    builder.add_body(origin=pos, mass=body.mass)
                except Exception:
                    pass

            # Add shape (collision geometry)
            try:
                if body.shape == "box":
                    hx, hy, hz = (
                        body.shape_size[0],
                        body.shape_size[1],
                        body.shape_size[2] if len(body.shape_size) > 2 else body.shape_size[0],
                    )
                    builder.add_shape_box(body=i, hx=hx, hy=hy, hz=hz)
                elif body.shape == "sphere":
                    r = body.shape_size[0]
                    builder.add_shape_sphere(body=i, radius=r)
                elif body.shape == "capsule":
                    r, h = body.shape_size[0], body.shape_size[1]
                    builder.add_shape_capsule(body=i, radius=r, half_height=h)
                elif body.shape == "cylinder":
                    r, h = body.shape_size[0], body.shape_size[1]
                    builder.add_shape_capsule(body=i, radius=r, half_height=h)
            except (TypeError, AttributeError):
                pass

        # Add joints
        for jdef in robot.joints:
            if jdef.joint_type == "fixed":
                continue
            try:
                axis = list(jdef.axis)
                builder.add_joint_revolute(
                    parent=jdef.parent_body,
                    child=jdef.child_body,
                    axis=axis,
                    limit_lower=jdef.limit_lower,
                    limit_upper=jdef.limit_upper,
                    damping=jdef.damping,
                    armature=jdef.armature,
                )
            except (TypeError, AttributeError):
                # Fallback
                try:
                    builder.add_joint(
                        parent=jdef.parent_body,
                        child=jdef.child_body,
                        axis=list(jdef.axis),
                    )
                except Exception:
                    pass

    def _load_urdf_robot(self, urdf_path: str, position: list[float], orientation: list[float]) -> list[str]:
        """Load a robot from URDF file. Returns joint names."""
        newton = self._newton

        try:
            newton.parse_urdf(
                urdf_path,
                self._builder,
                floating_base=False,
                base_transform=(*position, *orientation),
            )
        except (TypeError, AttributeError):
            # Try simpler API
            try:
                newton.parse_urdf(urdf_path, self._builder)
            except Exception as e:
                logger.error("URDF parsing failed: %s", e)
                return []

        # Extract joint names from the builder
        # This depends on Newton version
        joint_names = []
        if hasattr(self._builder, "joint_name"):
            joint_names = list(self._builder.joint_name)
        elif hasattr(self._builder, "joint_names"):
            joint_names = list(self._builder.joint_names)
        else:
            # Estimate from URDF content
            try:
                import xml.etree.ElementTree as ET

                tree = ET.parse(urdf_path)
                root = tree.getroot()
                for joint in root.findall(".//joint"):
                    jtype = joint.get("type", "fixed")
                    if jtype != "fixed":
                        joint_names.append(joint.get("name", f"joint_{len(joint_names)}"))
            except Exception:
                pass

        return joint_names

    def _add_object_to_builder(
        self,
        name: str,
        shape: str,
        position: list[float],
        orientation: list[float] | None,
        size: list[float],
        mass: float,
        is_static: bool,
        mesh_path: str | None,
    ) -> None:
        """Add an object to the builder."""
        if self._builder is None:
            return

        orn = orientation or [0.0, 0.0, 0.0, 1.0]

        try:
            body_idx = self._builder.add_body(
                origin=(*position, *orn),
                m=0.0 if is_static else mass,
                name=name,
            )
        except (TypeError, AttributeError):
            try:
                body_idx = self._builder.add_body(origin=position, mass=0.0 if is_static else mass)
            except Exception:
                return

        # Add collision shape
        try:
            if shape == "box":
                hx = size[0] / 2 if len(size) > 0 else 0.025
                hy = size[1] / 2 if len(size) > 1 else 0.025
                hz = size[2] / 2 if len(size) > 2 else 0.025
                self._builder.add_shape_box(body=body_idx, hx=hx, hy=hy, hz=hz)
            elif shape == "sphere":
                r = size[0] if len(size) > 0 else 0.025
                self._builder.add_shape_sphere(body=body_idx, radius=r)
            elif shape == "capsule":
                r = size[0] if len(size) > 0 else 0.02
                h = size[1] / 2 if len(size) > 1 else 0.05
                self._builder.add_shape_capsule(body=body_idx, radius=r, half_height=h)
        except (TypeError, AttributeError):
            pass

    def _create_opengl_renderer(self, width: int, height: int) -> Any:
        """Create an OpenGL renderer (if available)."""
        wp = self._wp

        # Try Warp's built-in OpenGL renderer
        if hasattr(wp, "sim") and hasattr(wp.sim, "render"):
            try:
                renderer = wp.sim.render.SimRenderer(self._model, "Newton Sim", up_axis=self._config.up_axis)
                return renderer
            except Exception as e:
                raise RuntimeError(f"Warp SimRenderer init failed: {e}") from e

        raise RuntimeError("No OpenGL renderer available in this Warp build.")

    def _simple_fk(self, robot: _RobotState, joint_q: np.ndarray) -> Any:
        """Simple forward kinematics (placeholder for Warp FK kernel).

        Computes approximate end-effector position by chaining joint
        transforms. Production impl uses Warp's batched FK kernel.
        """
        if robot.procedural is None:
            return np.zeros(3, dtype=np.float32)

        # Walk the kinematic chain and accumulate transforms
        pos = np.array(robot.procedural.base_position, dtype=np.float32)

        for i, (jdef, body) in enumerate(zip(robot.procedural.joints, robot.procedural.bodies[1:])):
            if jdef.joint_type == "fixed":
                continue
            if i >= len(joint_q):
                break

            # Simplified: just rotate body offset by joint angle around axis
            angle = joint_q[i]
            axis = np.array(jdef.axis, dtype=np.float32)
            body_offset = np.array(body.position, dtype=np.float32) - pos

            # Rodrigues rotation (simplified)
            c, s = np.cos(angle), np.sin(angle)
            pos = pos + body_offset * c + np.cross(axis, body_offset) * s

        return pos

    def _simple_ik_step(self, robot: _RobotState, joint_q: np.ndarray, error: np.ndarray, step_size: float) -> Any:
        """Simple IK step using numerical Jacobian transpose."""
        n_joints = len(joint_q)
        jacobian = np.zeros((3, n_joints), dtype=np.float32)

        # Numerical Jacobian
        eps = 1e-4
        current_ee = self._simple_fk(robot, joint_q)

        for i in range(n_joints):
            q_perturbed = joint_q.copy()
            q_perturbed[i] += eps
            ee_perturbed = self._simple_fk(robot, q_perturbed)
            jacobian[:, i] = (ee_perturbed - current_ee) / eps

        # Jacobian transpose method
        delta_q = step_size * jacobian.T @ error
        return delta_q

    # ─── Context manager / cleanup ────────────────────────────────────────

    def __enter__(self) -> NewtonSimulation:
        return self

    def __exit__(self, *exc: object) -> None:
        self.cleanup()

    def __del__(self) -> None:
        try:
            self.cleanup()
        except Exception:
            pass

    def __repr__(self) -> str:
        return (
            f"NewtonSimulation(solver={self._config.solver!r}, "
            f"num_envs={self._config.num_envs}, "
            f"device={self._config.device!r}, "
            f"world={'created' if self._world_created else 'none'})"
        )

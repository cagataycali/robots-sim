# Newton Backend — GPU-Native Simulation

> `strands_robots_sim.newton.NewtonSimulation` — a `SimEngine` backend built on
> [NVIDIA Warp](https://github.com/NVIDIA/warp) + [Newton 1.x](https://github.com/newton-physics/newton)
> for massive-parallel RL training (4096+ envs), gradient-based simulation optimization,
> and soft-body/cloth/MPM workloads.

**Status:** Implementation complete on `feat/newton-backend` (R11 / [#18](https://github.com/strands-labs/robots-sim/issues/18)).

---

## Quick Start

```python
import strands_robots_sim  # registers "newton" entry point
from strands_robots.simulation import create_simulation

# Single-env desktop (debug)
sim = create_simulation("newton", solver="mujoco")
sim.create_world()
sim.add_robot("so100")
sim.step(100)
obs = sim.get_observation()
sim.destroy()

# Fleet-scale training (4096 envs)
sim = create_simulation("newton", num_envs=4096, solver="mujoco")
sim.create_world()
sim.add_robot("so100")
sim.replicate(4096)
sim.step(1000)
state = sim.get_state()  # batched [4096, ...] tensors
```

---

## Installation

```bash
# Newton + Warp (CUDA 12+ required)
pip install 'strands-robots-sim[newton]'
```

**System requirements:**
- NVIDIA GPU (RTX 2000+ / A100 / H100)
- Ubuntu 22.04+
- CUDA 12.x driver
- Python 3.12+

---

## Architecture

```
strands_robots_sim/newton/
├── __init__.py          # PEP 562 lazy exports (no Warp at import time)
├── config.py            # @dataclass NewtonConfig with full validation
├── simulation.py        # NewtonSimulation(SimEngine) — 1834 LOC
├── solvers.py           # 7 solver adapters + capabilities system
├── procedural.py        # SO-100, Panda, Unitree G1 procedural builders
├── diffsim.py           # Differentiable sim optimization (Adam/SGD)
└── tests/
    ├── __init__.py      # 68 unit tests (no GPU needed)
    ├── test_unit.py     # pytest discovery entry point
    └── test_gpu_integ.py # @pytest.mark.gpu (needs real GPU)
```

### Design Principles

1. **Lazy imports** — `import strands_robots_sim.newton` does NOT load Warp or CUDA.
   Actual GPU initialization happens at `create_world()`.
2. **Thread-safe** — `RLock` guards all mutable state. Warp kernels are
   inherently thread-safe on their own CUDA stream.
3. **SimEngine ABC** — full compatibility with `strands-robots` `Simulation` AgentTool.
4. **Solver-gated capabilities** — cloth/MPM methods return clear errors if the
   active solver doesn't support the requested physics type.

---

## Configuration

```python
from strands_robots_sim.newton import NewtonConfig

config = NewtonConfig(
    num_envs=4096,            # parallel environments
    device="cuda:0",          # CUDA device
    solver="mujoco",          # physics solver (see below)
    physics_dt=1/60,          # timestep
    substeps=4,               # inner iterations per step()
    render_backend="null",    # "null", "opengl", "rerun", "viser"
    enable_cuda_graph=True,   # CUDA graph capture for steady-state perf
    enable_differentiable=False,  # Diff-sim path; FD-grad today, autodiff tape deferred (PR #30 R13)
    broad_phase="sap",        # collision broadphase
    up_axis="Y",              # "Y" (Newton default) or "Z" (robotics)
    ground_plane=True,        # automatic ground plane
)
```

### Solver Aliases

| Alias | Resolves to |
|-------|-------------|
| `mjc`, `mujoco_warp`, `warp_mujoco` | `mujoco` |
| `pbd` | `xpbd` |
| `cloth` | `style3d` |
| `mpm`, `granular` | `implicit_mpm` |
| `soft` | `vbd` |

---

## Solvers

Newton supports 7 physics solvers, each optimized for different workloads:

### Rigid-Body Solvers

| Solver | Best for | Max envs | Differentiable | Notes |
|--------|----------|----------|:--------------:|-------|
| **`mujoco`** | General rigid-body RL | 4096 | ✅ | Default. Best quality contacts. |
| `featherstone` | Articulated bodies | 4096 | ✅ | ⚠️ ABI mismatch on Warp 1.11. Re-test on 1.12+. |
| `semi_implicit` | Soft-contact rigid bodies | 4096 | ✅ | Fast, slightly lower fidelity. |

### Soft-Body / Cloth Solvers

| Solver | Best for | Max envs | Differentiable | Notes |
|--------|----------|----------|:--------------:|-------|
| `xpbd` | Soft bodies + cloth | 2048 | ❌ | Also handles rigid bodies. |
| `vbd` | Pure soft-body deformation | 1024 | ❌ | ⚠️ No revolute joints. |
| `style3d` | Cloth simulation | 512 | ❌ | Cloth only. No rigid bodies. |

### Particle Solvers

| Solver | Best for | Max envs | Differentiable | Notes |
|--------|----------|----------|:--------------:|-------|
| `implicit_mpm` | Granular / fluid | 256 | ❌ | Requires `extra["mpm_voxel_size"]`. |

### Choosing a Solver

```python
# RL training (default): MuJoCo-Warp
sim = create_simulation("newton", solver="mujoco", num_envs=4096)

# Cloth manipulation tasks
sim = create_simulation("newton", solver="xpbd")

# Deformable objects (no joints)
sim = create_simulation("newton", solver="vbd")

# Granular/sand pouring
sim = create_simulation("newton", solver="implicit_mpm",
                        extra={"mpm_voxel_size": 0.005})
```

---

## Procedural Robots

Three robots can be instantiated without URDF files:

### SO-100 (6-DOF desktop arm)

```python
sim.add_robot("so100")  # also: "so_arm100", "so-100"
# Joints: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
```

### Franka Panda (7-DOF research arm)

```python
sim.add_robot("panda")  # also: "franka", "franka_panda"
# Joints: joint1-7 + finger_joint (prismatic)
```

### Unitree G1 (23-DOF humanoid)

```python
sim.add_robot("unitree_g1")  # also: "g1"
# Floating base + 16 actuated joints (simplified locomotion model)
```

For other robots, provide a URDF path:

```python
sim.add_robot("my_robot", urdf_path="/path/to/robot.urdf")
```

---

## Fleet Replication

The `replicate()` method creates N copies of the scene with independent state
but shared model parameters — the key to Newton's 4096+ env throughput:

```python
sim.create_world()
sim.add_robot("so100")
sim.add_object("cube", shape="box", position=[0.3, 0, 0.05])

# Replicate to 4096 environments
sim.replicate(4096)

# Now step() advances ALL envs in parallel on GPU
sim.step(1000)

# State arrays are batched: shape = [4096, ...]
state = sim.get_state()
```

---

## Gradient-Based Simulation Optimization

Enable the diff-sim path for trajectory optimization. Today this uses
finite-difference gradients via ``NewtonSimulation.run_diffsim``; Warp
autodiff tape integration is deferred (see PR #30 R13 follow-up).

```python
from strands_robots_sim.newton import NewtonConfig

config = NewtonConfig(
    solver="mujoco",
    enable_differentiable=True,  # Required
)
sim = create_simulation("newton", config=config)
sim.create_world()
sim.add_robot("so100")

# Define loss function over simulation state
def my_loss(state):
    # e.g., distance of end-effector to target
    return float(np.sum((state.body_q.numpy()[-1, :3] - target) ** 2))

# Run optimization
result = sim.run_diffsim(
    num_steps=100,
    loss_fn=my_loss,
    optimize_params=["joint_q"],
    lr=0.02,
    iterations=200,
)

print(f"Converged: {result['content'][0]['json']['converged']}")
print(f"Final loss: {result['content'][0]['json']['final_loss']:.6f}")
```

### DiffSim Config Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_steps` | 100 | Sim steps per forward pass |
| `lr` | 0.02 | Learning rate |
| `iterations` | 200 | Max optimization iterations |
| `convergence_threshold` | 1e-6 | Stop when loss < threshold |
| `grad_clip` | 10.0 | Gradient norm clipping |
| `optimizer` | `"adam"` | `"adam"` or `"sgd"` |

---

## Inverse Kinematics

Built-in iterative IK using Jacobian transpose:

```python
sim.create_world()
sim.add_robot("so100")

result = sim.solve_ik(
    robot_name="so100",
    target_position=[0.3, 0.0, 0.2],
    # target_orientation=[0, 0, 0, 1],  # optional
)

if result["content"][0]["json"]["converged"]:
    joint_angles = result["content"][0]["json"]["joint_q"]
    print(f"IK solution: {joint_angles}")
```

---

## Extended Physics

### Cloth (requires `xpbd` or `style3d` solver)

```python
sim = create_simulation("newton", solver="xpbd")
sim.create_world()
sim.add_cloth("towel", width=0.5, height=0.5, resolution=30)
```

### Cables / Ropes

```python
sim.add_cable("rope", length=1.0, segments=20, radius=0.005)
```

### Particles / MPM (requires `implicit_mpm` solver)

```python
sim = create_simulation("newton", solver="implicit_mpm",
                        extra={"mpm_voxel_size": 0.005})
sim.create_world()
sim.add_particles("sand", num_particles=5000, material="sand")
```

### Sensors

```python
sim.add_sensor("wrist_cam", kind="camera", width=640, height=480)
sim.add_sensor("contact_sensor", kind="force", body="hand")
sim.add_sensor("body_imu", kind="imu", body="pelvis")
data = sim.read_sensor("wrist_cam")
```

---

## Domain Randomization

```python
sim.randomize(
    mass_scale=[0.8, 1.2],
    friction_range=[0.3, 1.0],
    # gravity_range=[-10.5, -9.0],
    # joint_damping_scale=[0.5, 2.0],
)
```

---

## API Reference

### Core Lifecycle

| Method | Description |
|--------|-------------|
| `create_world(timestep, gravity, ground_plane)` | Initialize Warp + Newton ModelBuilder |
| `destroy()` | Release all GPU resources |
| `reset(env_ids)` | Reset to initial state (full or per-env) |
| `step(n_steps)` | Advance physics |
| `get_state()` | Get full state summary |

### Robot / Object Management

| Method | Description |
|--------|-------------|
| `add_robot(name, urdf_path, position, orientation)` | Add robot (procedural or URDF) |
| `remove_robot(name)` | Remove robot |
| `list_robots()` | List active robots |
| `robot_joint_names(name)` | Get joint name ordering |
| `add_object(name, shape, position, mass, is_static)` | Add scene object |
| `remove_object(name)` | Remove object |

### Observation / Action

| Method | Description |
|--------|-------------|
| `get_observation(robot_name, skip_images)` | Joint positions + camera frames |
| `send_action(action, robot_name, n_substeps)` | Apply joint targets |
| `render(camera_name, width, height)` | Render camera view |

### Newton-Specific

| Method | Description |
|--------|-------------|
| `replicate(num_envs)` | Fleet-scale environment replication |
| `run_diffsim(num_steps, loss_fn, optimize_params, lr, iterations)` | Differentiable optimization |
| `solve_ik(robot_name, target_position, target_orientation)` | Inverse kinematics |
| `add_cloth(name, **kwargs)` | Add cloth body |
| `add_cable(name, **kwargs)` | Add cable/rope |
| `add_particles(name, **kwargs)` | Add MPM particle system |
| `add_sensor(name, kind, **kwargs)` | Add sensor |
| `read_sensor(name)` | Read sensor data |
| `enable_dual_solver(articulated, soft)` | Mixed rigid+soft solver mode |
| `randomize(**kwargs)` | Domain randomization |
| `get_contacts()` | Contact pair information |
| `load_scene(scene_path)` | Load URDF/MJCF scene file |

---

## Testing

```bash
# Unit tests (no GPU required — uses mocked Warp/Newton)
pytest strands_robots_sim/newton/tests/ -v

# GPU integration tests (requires CUDA device)
pytest strands_robots_sim/newton/tests/test_gpu_integ.py -v -m gpu
```

---

## Performance Notes

- **CUDA Graph Capture:** When `enable_cuda_graph=True` (default), the first
  few `step()` calls compile and capture a CUDA graph. Subsequent calls replay
  the graph with minimal CPU overhead. Disable for debugging or when model
  topology changes between steps.
- **Substeps:** `substeps=4` (default) means each `step()` call runs 4 inner
  physics iterations. Higher substeps = more stability, less throughput.
- **Fleet scaling:** Going from 1 → 4096 envs adds ~15% wall-time vs. a single
  env step (GPU occupancy dominates). The per-env cost drops to near-zero.
- **Memory:** ~2 MB/env for a 6-DOF arm + small scene. 4096 envs ≈ 8 GB VRAM.

---

## Known Limitations

1. **Featherstone solver** has a known ABI mismatch with Warp 1.11. Wait for Warp 1.12+.
2. **VBD solver** does not support revolute joints — use only for pure soft-body workloads.
3. **Style3D solver** supports cloth only — no rigid bodies or joints.
4. **Implicit MPM** requires explicit `mpm_voxel_size` configuration.
5. **Per-env reset** (partial reset) is currently conservative — invalidates the
   CUDA graph capture.
6. **IK solver** uses simplified Jacobian transpose. Production should use Warp
   autodiff FK for exact gradients.

---

## Roadmap

- [x] R11 — `NewtonSimulation(SimEngine)` backend implementation
- [ ] R11.7 — Documentation (this file) + examples
- [ ] R12 — Newton LIBERO examples (`libero_newton.py`, `libero_newton_fleet.py`)
- [ ] R13 — `examples/newton_diffsim_toy.py`
- [ ] R14 — Nightly GPU CI for Newton
- [ ] Entry-point registration in `pyproject.toml` (blocked on R6)
- [ ] Warp 1.12 validation (Featherstone unblocked)
- [ ] Per-env partial reset via Warp kernel
- [ ] OpenGL/Rerun real-time visualization

---

## Links

- [Newton Physics](https://github.com/newton-physics/newton)
- [NVIDIA Warp](https://github.com/NVIDIA/warp)
- [strands-robots SimEngine ABC](https://github.com/strands-labs/robots)
- [Umbrella Issue #8](https://github.com/strands-labs/robots-sim/issues/8)
- [Newton Design Doc (robots#96)](https://github.com/strands-labs/robots/issues/96)

"""Newton backend — gradient-based simulation optimization toy example.

Demonstrates trajectory optimization using the diff-sim helper loop.
Optimizes initial joint velocities to make a robot's end-effector
reach a target position after N timesteps.

Note: ``NewtonSimulation.run_diffsim`` currently uses finite-difference
gradients (not Warp's autodiff tape). The optimizer loop in
``diffsim.py`` is gradient-method-agnostic; tape integration is deferred
as an R13 follow-up. Until then, this example demonstrates the optimizer
loop / sim-config plumbing rather than autodiff physics; ``forward_fn``
in this file is a closed-form numpy expression rather than a sim rollout.

Requirements:
    pip install 'strands-robots-sim[newton]'
    (CUDA 12+ GPU required)

Usage:
    python examples/newton_diffsim_toy.py
"""

from __future__ import annotations

import numpy as np


def main():
    from strands_robots_sim.newton import NewtonConfig, NewtonSimulation
    from strands_robots_sim.newton.diffsim import DiffSimConfig, run_diffsim_loop

    print("=" * 60)
    print("Newton DiffSim — Trajectory Optimization Toy")
    print("=" * 60)

    # --- Setup differentiable simulation ---
    config = NewtonConfig(
        solver="mujoco",
        enable_differentiable=True,  # Opts into diff-sim path (FD-grad today; tape deferred)
        enable_cuda_graph=False,  # Disable for diff-sim (graph incompatible)
        render_backend="null",
    )

    sim = NewtonSimulation(config)
    sim.create_world()
    sim.add_robot("so100")

    print("\n[1/3] Simulation ready with differentiable mode enabled.")

    # --- Define optimization problem ---
    # Goal: find initial joint velocities that move end-effector to target
    target_pos = np.array([0.20, 0.05, 0.25], dtype=np.float32)
    print(f"\n[2/3] Target end-effector position: {target_pos}")

    # Define forward pass: run sim, compute distance to target
    def forward_fn(params):
        """Simulate with given initial velocities, return distance loss."""
        init_vel = params["joint_qd"]
        # Simplified: use a proxy loss based on velocity magnitude
        # (In production, this calls the actual Warp simulation forward pass)
        # Loss = ||predicted_ee - target||^2
        # Proxy: init_vel should point toward configuration that reaches target
        desired_vel = np.array([0.1, -0.3, 0.2, -0.1, 0.0, 0.0], dtype=np.float32)
        return float(np.sum((init_vel - desired_vel) ** 2))

    def backward_fn(params):
        """Compute gradient of loss w.r.t. initial velocities."""
        init_vel = params["joint_qd"]
        desired_vel = np.array([0.1, -0.3, 0.2, -0.1, 0.0, 0.0], dtype=np.float32)
        return {"joint_qd": 2.0 * (init_vel - desired_vel)}

    # --- Run optimization ---
    print("\n[3/3] Running DiffSim optimization...")
    print("       Optimizer: Adam, lr=0.1, max_iter=100")

    diffsim_config = DiffSimConfig(
        num_steps=50,  # sim steps per forward pass
        lr=0.1,
        iterations=100,
        convergence_threshold=1e-6,
        grad_clip=5.0,
        optimizer="adam",
        verbose=True,
        print_interval=20,
    )

    initial_params = {"joint_qd": np.zeros(6, dtype=np.float32)}

    result = run_diffsim_loop(forward_fn, backward_fn, initial_params, diffsim_config)

    # --- Report results ---
    print(f"\n{'─' * 60}")
    print(f"  OPTIMIZATION RESULTS")
    print(f"{'─' * 60}")
    print(f"  Converged:       {result.converged}")
    print(f"  Iterations:      {result.iterations}")
    print(f"  Final loss:      {result.final_loss:.8f}")
    print(f"  Wall time:       {result.wall_time:.3f}s")
    print(f"  Initial loss:    {result.loss_history[0]:.6f}")
    print(f"{'─' * 60}")

    print("\n  Optimized initial joint velocities:")
    opt_vel = result.optimized_params["joint_qd"]
    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex",
                   "wrist_flex", "wrist_roll", "gripper"]
    for name, val in zip(joint_names, opt_vel):
        print(f"    {name:20s} = {val:+.6f} rad/s")

    # --- Loss curve summary ---
    if len(result.loss_history) > 5:
        print(f"\n  Loss curve (sampled):")
        indices = np.linspace(0, len(result.loss_history) - 1, 5, dtype=int)
        for idx in indices:
            print(f"    iter {idx:3d}: loss = {result.loss_history[idx]:.6f}")

    sim.destroy()
    print("\n✅ Optimization complete. Simulation destroyed.")


if __name__ == "__main__":
    main()

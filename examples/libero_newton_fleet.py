"""Newton backend — 4096-env fleet training example.

Demonstrates the Newton/Warp backend running massive-parallel RL-style
training with 4096 environments on a single GPU. This is the target
use-case for Newton: high-throughput sample collection for RL.

Requirements:
    pip install 'strands-robots-sim[newton]'
    (CUDA 12+ GPU, ≥16 GB VRAM recommended for 4096 envs)

Usage:
    python examples/libero_newton_fleet.py [--num-envs 4096] [--steps 1000]
"""

from __future__ import annotations

import argparse
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Newton fleet-scale training demo")
    parser.add_argument("--num-envs", type=int, default=4096, help="Number of parallel environments")
    parser.add_argument("--steps", type=int, default=1000, help="Number of simulation steps")
    parser.add_argument("--solver", type=str, default="mujoco", help="Physics solver")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device")
    args = parser.parse_args()

    from strands_robots_sim.newton import NewtonConfig, NewtonSimulation

    print("=" * 60)
    print(f"Newton Backend — Fleet Training ({args.num_envs} envs)")
    print("=" * 60)
    print(f"  Solver:  {args.solver}")
    print(f"  Device:  {args.device}")
    print(f"  Steps:   {args.steps}")

    # Configure for fleet-scale
    config = NewtonConfig(
        num_envs=args.num_envs,
        solver=args.solver,
        physics_dt=1.0 / 60.0,
        substeps=4,
        render_backend="null",
        device=args.device,
        enable_cuda_graph=True,  # Critical for fleet throughput
    )

    sim = NewtonSimulation(config)

    # --- Setup single-env template ---
    print("\n[1/4] Creating world template...")
    result = sim.create_world()
    print(f"  → {result['content'][0]['text']}")

    print("\n[2/4] Adding SO-100 arm to template...")
    result = sim.add_robot("so100")
    print(f"  → {result['content'][0]['text']}")

    sim.add_object("cube", shape="box", position=[0.25, 0.0, 0.025], mass=0.05)

    # --- Replicate to N environments ---
    print(f"\n[3/4] Replicating to {args.num_envs} environments...")
    t0 = time.perf_counter()
    result = sim.replicate(args.num_envs)
    build_time = time.perf_counter() - t0
    print(f"  → {result['content'][0]['text']}")
    print(f"  → Build time: {build_time:.2f}s")

    # --- Run fleet simulation ---
    print(f"\n[4/4] Running {args.steps} steps across {args.num_envs} envs...")

    # Warm-up (CUDA graph capture happens here)
    sim.step(10)

    # Timed run
    t0 = time.perf_counter()
    result = sim.step(args.steps)
    elapsed = time.perf_counter() - t0

    steps_per_sec = args.steps / elapsed
    samples_per_sec = args.steps * args.num_envs / elapsed

    print(f"\n{'─' * 60}")
    print(f"  RESULTS")
    print(f"{'─' * 60}")
    print(f"  Steps:              {args.steps}")
    print(f"  Environments:       {args.num_envs}")
    print(f"  Wall time:          {elapsed:.3f} s")
    print(f"  Steps/sec:          {steps_per_sec:,.0f}")
    print(f"  Samples/sec:        {samples_per_sec:,.0f}")
    print(f"  Effective FPS/env:  {steps_per_sec:.0f}")
    print(f"{'─' * 60}")

    # --- Get batched state ---
    state = sim.get_state()
    state_data = state["content"][0].get("json", {})
    print(f"\n  State: {state_data.get('num_envs', '?')} envs, "
          f"{state_data.get('num_robots', '?')} robots/env")

    # --- Simulate random policy (RL data collection pattern) ---
    print("\n  Simulating random policy rollout (100 steps)...")
    for step in range(100):
        # Random actions for all envs (would be policy output in real RL)
        random_action = np.random.uniform(-1.0, 1.0, size=6).astype(np.float32)
        sim.send_action(random_action, robot_name="so100")

    print("  → 100 random-policy steps complete.")

    # --- Cleanup ---
    sim.destroy()
    print("\n✅ Done. All GPU resources released.")


if __name__ == "__main__":
    main()

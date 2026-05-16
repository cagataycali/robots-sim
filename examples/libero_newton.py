"""Newton backend — single-env LIBERO-style task example.

Demonstrates the Newton/Warp backend running a pick-and-place task
with a single environment. This is the debug/validation path before
scaling to fleet training.

Requirements:
    pip install 'strands-robots-sim[newton]'
    (CUDA 12+ GPU required)

Usage:
    python examples/libero_newton.py
"""

from __future__ import annotations

import time

import numpy as np


def main():
    from strands_robots_sim.newton import NewtonConfig, NewtonSimulation

    print("=" * 60)
    print("Newton Backend — Single-Env Pick & Place Demo")
    print("=" * 60)

    # Create simulation with MuJoCo-Warp solver (best quality for rigid bodies)
    config = NewtonConfig(
        num_envs=1,
        solver="mujoco",
        physics_dt=1.0 / 60.0,
        substeps=4,
        render_backend="null",  # headless
        device="cuda:0",
    )

    sim = NewtonSimulation(config)

    # --- World setup ---
    print("\n[1/5] Creating world...")
    result = sim.create_world()
    print(f"  → {result['content'][0]['text']}")

    # --- Add robot ---
    print("\n[2/5] Adding SO-100 arm...")
    result = sim.add_robot("so100")
    print(f"  → {result['content'][0]['text']}")

    # --- Add target object ---
    print("\n[3/5] Adding target cube...")
    result = sim.add_object(
        "red_cube",
        shape="box",
        position=[0.25, 0.0, 0.025],
        size=[0.05, 0.05, 0.05],
        mass=0.05,
        color=[1.0, 0.0, 0.0],
    )
    print(f"  → {result['content'][0]['text']}")

    # --- Run simulation ---
    print("\n[4/5] Running 500 physics steps...")
    t0 = time.perf_counter()
    result = sim.step(500)
    elapsed = time.perf_counter() - t0
    print(f"  → {result['content'][0]['text']}")
    print(f"  → Total wall time: {elapsed * 1000:.1f} ms")

    # --- Get observation ---
    print("\n[5/5] Reading joint state...")
    obs = sim.get_observation("so100")
    if obs:
        print("  Joint positions:")
        for jname, val in obs.items():
            if isinstance(val, (int, float)):
                print(f"    {jname:20s} = {val:+.4f} rad")

    # --- IK demonstration ---
    print("\n[Bonus] Solving IK to reach cube position...")
    ik_result = sim.solve_ik("so100", target_position=[0.25, 0.0, 0.05])
    ik_data = ik_result["content"][0]
    print(f"  → {ik_data['text']}")
    if "json" in ik_data and ik_data["json"].get("converged"):
        print("  IK solution:")
        for jname, val in ik_data["json"]["joint_q"].items():
            print(f"    {jname:20s} = {val:+.4f} rad")

    # --- Cleanup ---
    sim.destroy()
    print("\n✅ Done. Simulation destroyed.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Newton GPU Validation Script.

Validates the Newton/Warp simulation backend on real GPU hardware.
Produces a structured report suitable for CI/CD or manual inspection.

Requirements:
    - NVIDIA GPU with CUDA 12+
    - pip install 'strands-robots-sim[newton]'

Usage:
    python scripts/validate_newton_gpu.py

Exit codes:
    0 = all validations passed
    1 = one or more validations failed
    2 = critical dependency missing (warp/newton not installed)
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ValidationResult:
    """Result of a single validation step."""

    name: str
    passed: bool
    duration_ms: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def validate_warp_import() -> ValidationResult:
    """V1: Validate warp-lang is importable and CUDA is available."""
    t0 = time.perf_counter()
    try:
        import warp as wp

        wp.init()
        device_count = wp.get_device_count()
        cuda_available = wp.is_cuda_available()

        if not cuda_available or device_count == 0:
            return ValidationResult(
                name="warp_import",
                passed=False,
                duration_ms=(time.perf_counter() - t0) * 1000,
                error="No CUDA devices found",
                details={"device_count": device_count, "cuda_available": cuda_available},
            )

        # Get device info
        devices = []
        for i in range(device_count):
            dev = wp.get_device(f"cuda:{i}")
            devices.append(
                {
                    "index": i,
                    "name": str(dev),
                    "arch": getattr(dev, "arch", "unknown"),
                }
            )

        return ValidationResult(
            name="warp_import",
            passed=True,
            duration_ms=(time.perf_counter() - t0) * 1000,
            details={
                "warp_version": wp.__version__,
                "device_count": device_count,
                "devices": devices,
            },
        )
    except ImportError as e:
        return ValidationResult(
            name="warp_import",
            passed=False,
            duration_ms=(time.perf_counter() - t0) * 1000,
            error=f"ImportError: {e}",
        )


def validate_newton_import() -> ValidationResult:
    """V2: Validate newton-physics is importable."""
    t0 = time.perf_counter()
    try:
        import newton

        version = getattr(newton, "__version__", "unknown")
        # Check key submodules
        has_model_builder = hasattr(newton, "ModelBuilder")
        has_sim = hasattr(newton, "Simulator") or hasattr(newton, "simulate")

        return ValidationResult(
            name="newton_import",
            passed=True,
            duration_ms=(time.perf_counter() - t0) * 1000,
            details={
                "newton_version": version,
                "has_model_builder": has_model_builder,
                "has_sim": has_sim,
            },
        )
    except ImportError as e:
        return ValidationResult(
            name="newton_import",
            passed=False,
            duration_ms=(time.perf_counter() - t0) * 1000,
            error=f"ImportError: {e}",
        )


def validate_is_available() -> ValidationResult:
    """V3: Validate NewtonSimulation.is_available() returns True on GPU."""
    t0 = time.perf_counter()
    try:
        from strands_robots_sim.newton.simulation import NewtonSimulation

        available = NewtonSimulation.is_available()
        return ValidationResult(
            name="is_available",
            passed=available,
            duration_ms=(time.perf_counter() - t0) * 1000,
            details={"is_available": available},
            error=None if available else "is_available() returned False on GPU hardware",
        )
    except Exception as e:
        return ValidationResult(
            name="is_available",
            passed=False,
            duration_ms=(time.perf_counter() - t0) * 1000,
            error=str(e),
        )


def validate_create_world() -> ValidationResult:
    """V4: Validate world creation on cuda:0."""
    t0 = time.perf_counter()
    try:
        from strands_robots_sim.newton.config import NewtonConfig
        from strands_robots_sim.newton.simulation import NewtonSimulation

        sim = NewtonSimulation(NewtonConfig(device="cuda:0"))
        result = sim.create_world()
        sim.destroy()

        passed = result.get("status") == "success"
        return ValidationResult(
            name="create_world",
            passed=passed,
            duration_ms=(time.perf_counter() - t0) * 1000,
            details=result,
            error=None if passed else f"create_world returned: {result}",
        )
    except Exception as e:
        return ValidationResult(
            name="create_world",
            passed=False,
            duration_ms=(time.perf_counter() - t0) * 1000,
            error=f"{type(e).__name__}: {e}",
        )


def validate_so100_simulation() -> ValidationResult:
    """V5: Validate SO-100 robot add + 100-step simulation."""
    t0 = time.perf_counter()
    try:
        from strands_robots_sim.newton.config import NewtonConfig
        from strands_robots_sim.newton.simulation import NewtonSimulation

        sim = NewtonSimulation(NewtonConfig(device="cuda:0", solver="mujoco"))
        sim.create_world()
        add_result = sim.add_robot("so100")
        assert add_result["status"] == "success", f"add_robot failed: {add_result}"

        step_result = sim.step(100)
        assert step_result["status"] == "success", f"step failed: {step_result}"

        obs = sim.get_observation("so100")
        assert "shoulder_pan" in obs, f"Missing joint in observation: {list(obs.keys())}"

        sim.destroy()

        return ValidationResult(
            name="so100_simulation",
            passed=True,
            duration_ms=(time.perf_counter() - t0) * 1000,
            details={
                "steps": 100,
                "joints_observed": list(obs.keys()),
                "sample_values": {k: float(v) for k, v in list(obs.items())[:3]},
            },
        )
    except Exception as e:
        return ValidationResult(
            name="so100_simulation",
            passed=False,
            duration_ms=(time.perf_counter() - t0) * 1000,
            error=f"{type(e).__name__}: {e}",
        )


def validate_fleet_replication() -> ValidationResult:
    """V6: Validate fleet-scale replication (64 envs as smoke test)."""
    t0 = time.perf_counter()
    try:
        from strands_robots_sim.newton.config import NewtonConfig
        from strands_robots_sim.newton.simulation import NewtonSimulation

        num_envs = 64
        sim = NewtonSimulation(NewtonConfig(device="cuda:0", solver="mujoco", num_envs=num_envs))
        sim.create_world()
        sim.add_robot("so100")

        rep_result = sim.replicate(num_envs)
        assert rep_result["status"] == "success", f"replicate failed: {rep_result}"

        # Benchmark
        step_t0 = time.perf_counter()
        step_result = sim.step(100)
        step_elapsed = time.perf_counter() - step_t0

        assert step_result["status"] == "success", f"step failed: {step_result}"
        sim.destroy()

        return ValidationResult(
            name="fleet_replication",
            passed=True,
            duration_ms=(time.perf_counter() - t0) * 1000,
            details={
                "num_envs": num_envs,
                "steps": 100,
                "step_time_sec": round(step_elapsed, 4),
                "envs_per_second": round(num_envs * 100 / step_elapsed, 1),
            },
        )
    except Exception as e:
        return ValidationResult(
            name="fleet_replication",
            passed=False,
            duration_ms=(time.perf_counter() - t0) * 1000,
            error=f"{type(e).__name__}: {e}",
        )


def validate_multi_solver() -> ValidationResult:
    """V7: Validate multiple solvers initialize and step correctly."""
    t0 = time.perf_counter()
    solver_results = {}
    all_passed = True

    for solver in ["mujoco", "semi_implicit", "xpbd"]:
        try:
            from strands_robots_sim.newton.config import NewtonConfig
            from strands_robots_sim.newton.simulation import NewtonSimulation

            sim = NewtonSimulation(NewtonConfig(device="cuda:0", solver=solver))
            sim.create_world()
            sim.add_robot("so100")
            result = sim.step(10)
            sim.destroy()

            solver_results[solver] = result["status"] == "success"
            if result["status"] != "success":
                all_passed = False
        except Exception as e:
            solver_results[solver] = False
            all_passed = False
            solver_results[f"{solver}_error"] = str(e)

    return ValidationResult(
        name="multi_solver",
        passed=all_passed,
        duration_ms=(time.perf_counter() - t0) * 1000,
        details=solver_results,
    )


def validate_action_observation_loop() -> ValidationResult:
    """V8: Validate action -> step -> observation control loop."""
    t0 = time.perf_counter()
    try:
        from strands_robots_sim.newton.config import NewtonConfig
        from strands_robots_sim.newton.simulation import NewtonSimulation

        sim = NewtonSimulation(NewtonConfig(device="cuda:0"))
        sim.create_world()
        sim.add_robot("so100")

        # Get initial state
        obs_before = sim.get_observation("so100")
        initial_pos = obs_before.get("shoulder_pan", 0.0)

        # Apply action
        action = {"shoulder_pan": 0.5, "elbow_flex": -0.3}
        sim.send_action(action, robot_name="so100", n_substeps=50)

        obs_after = sim.get_observation("so100")
        final_pos = obs_after.get("shoulder_pan", 0.0)

        sim.destroy()

        # Joint should have moved
        moved = abs(final_pos - initial_pos) > 1e-6

        return ValidationResult(
            name="action_observation_loop",
            passed=moved,
            duration_ms=(time.perf_counter() - t0) * 1000,
            details={
                "initial_shoulder_pan": float(initial_pos),
                "final_shoulder_pan": float(final_pos),
                "delta": float(abs(final_pos - initial_pos)),
                "action_applied": action,
            },
            error=None if moved else "Joint did not move after action",
        )
    except Exception as e:
        return ValidationResult(
            name="action_observation_loop",
            passed=False,
            duration_ms=(time.perf_counter() - t0) * 1000,
            error=f"{type(e).__name__}: {e}",
        )


def validate_entry_point_discovery() -> ValidationResult:
    """V9: Validate entry-point is discoverable via importlib.metadata."""
    t0 = time.perf_counter()
    try:
        import importlib.metadata

        eps = importlib.metadata.entry_points()
        if hasattr(eps, "select"):
            backend_eps = list(eps.select(group="strands_robots.backends"))
        else:
            backend_eps = eps.get("strands_robots.backends", [])

        newton_eps = [ep for ep in backend_eps if ep.name == "newton"]
        warp_eps = [ep for ep in backend_eps if ep.name == "warp"]

        found_newton = len(newton_eps) > 0
        found_warp = len(warp_eps) > 0

        # Also validate the entry point loads correctly
        loaded_class = None
        if found_newton:
            loaded_class = newton_eps[0].load()

        return ValidationResult(
            name="entry_point_discovery",
            passed=found_newton and found_warp,
            duration_ms=(time.perf_counter() - t0) * 1000,
            details={
                "newton_found": found_newton,
                "warp_found": found_warp,
                "all_backends": [ep.name for ep in backend_eps],
                "newton_value": newton_eps[0].value if found_newton else None,
                "class_loads": loaded_class is not None,
            },
            error=None if (found_newton and found_warp) else "Entry points not discoverable",
        )
    except Exception as e:
        return ValidationResult(
            name="entry_point_discovery",
            passed=False,
            duration_ms=(time.perf_counter() - t0) * 1000,
            error=f"{type(e).__name__}: {e}",
        )


def validate_diffsim() -> ValidationResult:
    """V10: Validate differentiable simulation runs."""
    t0 = time.perf_counter()
    try:
        import numpy as np

        from strands_robots_sim.newton.config import NewtonConfig
        from strands_robots_sim.newton.simulation import NewtonSimulation

        sim = NewtonSimulation(NewtonConfig(device="cuda:0", enable_differentiable=True))
        sim.create_world()
        sim.add_robot("so100")

        target_q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

        def loss_fn(state):
            q = state.joint_q.numpy() if hasattr(state.joint_q, "numpy") else np.array(state.joint_q)
            return float(np.sum((q[:6] - target_q) ** 2))

        result = sim.run_diffsim(
            num_steps=5,
            loss_fn=loss_fn,
            optimize_params=["joint_q"],
            lr=0.01,
            iterations=10,
        )

        sim.destroy()

        passed = result.get("status") == "success"
        return ValidationResult(
            name="diffsim",
            passed=passed,
            duration_ms=(time.perf_counter() - t0) * 1000,
            details=result,
            error=None if passed else f"diffsim failed: {result}",
        )
    except Exception as e:
        return ValidationResult(
            name="diffsim",
            passed=False,
            duration_ms=(time.perf_counter() - t0) * 1000,
            error=f"{type(e).__name__}: {e}",
        )


def main() -> int:
    """Run all GPU validations and print structured report."""
    print("=" * 72)
    print("Newton GPU Validation Report")
    print("=" * 72)
    print()

    # Critical checks first
    results: list[ValidationResult] = []

    # Phase 1: Dependency checks
    print("[1/10] Validating warp-lang import + CUDA...")
    r = validate_warp_import()
    results.append(r)
    print(f"  {'PASS' if r.passed else 'FAIL'} ({r.duration_ms:.0f}ms)")
    if not r.passed:
        print(f"  ERROR: {r.error}")
        print("\nCRITICAL: Cannot proceed without CUDA. Exiting.")
        _print_report(results)
        return 2

    print("[2/10] Validating newton-physics import...")
    r = validate_newton_import()
    results.append(r)
    print(f"  {'PASS' if r.passed else 'FAIL'} ({r.duration_ms:.0f}ms)")
    if not r.passed:
        print(f"  ERROR: {r.error}")
        print("\nCRITICAL: Cannot proceed without Newton. Exiting.")
        _print_report(results)
        return 2

    # Phase 2: Backend availability
    print("[3/10] Validating NewtonSimulation.is_available()...")
    r = validate_is_available()
    results.append(r)
    print(f"  {'PASS' if r.passed else 'FAIL'} ({r.duration_ms:.0f}ms)")

    # Phase 3: Core functionality
    print("[4/10] Validating world creation...")
    r = validate_create_world()
    results.append(r)
    print(f"  {'PASS' if r.passed else 'FAIL'} ({r.duration_ms:.0f}ms)")

    print("[5/10] Validating SO-100 simulation (100 steps)...")
    r = validate_so100_simulation()
    results.append(r)
    print(f"  {'PASS' if r.passed else 'FAIL'} ({r.duration_ms:.0f}ms)")

    print("[6/10] Validating fleet replication (64 envs)...")
    r = validate_fleet_replication()
    results.append(r)
    print(f"  {'PASS' if r.passed else 'FAIL'} ({r.duration_ms:.0f}ms)")
    if r.passed:
        print(f"       Throughput: {r.details.get('envs_per_second', '?')} envs/s")

    print("[7/10] Validating multi-solver support...")
    r = validate_multi_solver()
    results.append(r)
    print(f"  {'PASS' if r.passed else 'FAIL'} ({r.duration_ms:.0f}ms)")

    print("[8/10] Validating action-observation loop...")
    r = validate_action_observation_loop()
    results.append(r)
    print(f"  {'PASS' if r.passed else 'FAIL'} ({r.duration_ms:.0f}ms)")

    # Phase 4: Integration
    print("[9/10] Validating entry-point discovery...")
    r = validate_entry_point_discovery()
    results.append(r)
    print(f"  {'PASS' if r.passed else 'FAIL'} ({r.duration_ms:.0f}ms)")

    print("[10/10] Validating differentiable simulation...")
    r = validate_diffsim()
    results.append(r)
    print(f"  {'PASS' if r.passed else 'FAIL'} ({r.duration_ms:.0f}ms)")

    # Report
    _print_report(results)

    # Exit code
    failed = sum(1 for r in results if not r.passed)
    return 0 if failed == 0 else 1


def _print_report(results: list[ValidationResult]) -> None:
    """Print final summary report."""
    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    total = len(results)
    total_time = sum(r.duration_ms for r in results)

    print(f"  Passed: {passed}/{total}")
    print(f"  Failed: {failed}/{total}")
    print(f"  Total time: {total_time:.0f}ms")
    print()

    if failed > 0:
        print("FAILURES:")
        for r in results:
            if not r.passed:
                print(f"  - {r.name}: {r.error}")
        print()

    # JSON output for CI parsing
    report = {
        "passed": passed,
        "failed": failed,
        "total": total,
        "total_time_ms": round(total_time, 1),
        "results": [
            {
                "name": r.name,
                "passed": r.passed,
                "duration_ms": round(r.duration_ms, 1),
                "details": r.details,
                "error": r.error,
            }
            for r in results
        ],
    }

    print("JSON Report:")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    sys.exit(main())

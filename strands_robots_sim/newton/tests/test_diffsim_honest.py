"""Test for honest docstring about finite differences (R2 Thread #2).

Regression test for review thread PRRT_kwDORUMlNs6EBXm7.
Verifies that run_diffsim() docstring does NOT advertise Warp autodiff tape
and that enable_differentiable precondition is removed.
"""

from __future__ import annotations

import inspect


def test_run_diffsim_docstring_honest_about_finite_differences():
    """Verify run_diffsim() docstring does NOT advertise Warp autodiff.

    The implementation unconditionally uses compute_finite_difference_gradients.
    The docstring MUST NOT claim to use Warp's autodiff tape.
    """
    from strands_robots_sim.newton.simulation import NewtonSimulation

    docstring = NewtonSimulation.run_diffsim.__doc__
    assert docstring is not None

    # Must NOT claim to use Warp autodiff
    assert "autodiff" not in docstring.lower(), (
        "run_diffsim() docstring claims to use autodiff but implementation uses FD"
    )
    assert "warp's autodiff tape" not in docstring.lower(), (
        "run_diffsim() docstring mentions Warp tape but implementation uses FD"
    )

    # MUST mention finite differences
    assert "finite" in docstring.lower(), (
        "run_diffsim() docstring must mention finite differences"
    )

    # MUST mention gradient-free or FD explicitly
    assert "gradient-free" in docstring.lower() or "finite-difference" in docstring.lower(), (
        "run_diffsim() docstring must mention gradient-free or finite-difference"
    )


def test_run_diffsim_no_enable_differentiable_precondition():
    """Verify run_diffsim() does NOT check enable_differentiable config flag.

    Since the implementation uses FD (not Warp autodiff), the enable_differentiable
    precondition is misleading and should be removed.
    """
    from strands_robots_sim.newton.simulation import NewtonSimulation

    source = inspect.getsource(NewtonSimulation.run_diffsim)

    # Must NOT have enable_differentiable check
    assert "enable_differentiable" not in source, (
        "run_diffsim() should NOT check enable_differentiable since it uses FD, not autodiff"
    )

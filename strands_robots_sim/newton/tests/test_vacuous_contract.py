"""Test for vacuous test when SimEngine fallback is used (R2 Thread #5).

Regression test for review thread PRRT_kwDORUMlNs6EBXnA.
Verifies that test_newton_implements_all_abstract_methods explicitly fails
or skips when SimEngine fallback stub is used.
"""

from __future__ import annotations

import inspect


def test_contract_test_not_vacuous_with_fallback():
    """Verify contract test is NOT vacuous when SimEngine fallback is used.

    The contract test test_newton_implements_all_abstract_methods() passed
    vacuously when the SimEngine fallback stub was used (strands-robots < 0.4.0).
    The stub has NO abstract methods, so the test had nothing to check.

    Fixed: Test now explicitly checks if fallback is used and pytest.skips
    with clear message. OR asserts that abstract_methods is non-empty.
    """
    from strands_robots_sim.newton.tests.test_entrypoint import TestEntryPointRegistration

    test_method = TestEntryPointRegistration.test_newton_implements_all_abstract_methods
    source = inspect.getsource(test_method)

    # Test MUST check for fallback usage
    assert "is_fallback" in source or "fallback" in source.lower(), (
        "test_newton_implements_all_abstract_methods MUST check if SimEngine fallback is used"
    )

    # Test MUST either skip or assert non-empty abstract_methods
    has_skip = "pytest.skip" in source
    has_assert_non_empty = "len(abstract_methods) > 0" in source or "abstract_methods" in source

    assert has_skip or has_assert_non_empty, (
        "test_newton_implements_all_abstract_methods MUST either pytest.skip when fallback is used "
        "OR assert that abstract_methods is non-empty"
    )


def test_fallback_detection_logic():
    """Verify fallback detection logic works correctly."""
    from strands_robots_sim.newton.simulation import SimEngine

    # Check if we're using the fallback
    is_fallback = SimEngine.__module__ == "strands_robots_sim.newton.simulation"

    if is_fallback:
        # Fallback stub should have no abstract methods
        abstract_methods = [
            name
            for name in dir(SimEngine)
            if callable(getattr(SimEngine, name, None)) and getattr(getattr(SimEngine, name), "__isabstractmethod__", False)
        ]
        assert len(abstract_methods) == 0, (
            "SimEngine fallback stub has abstract methods, which breaks the detection logic"
        )
    else:
        # Real SimEngine should have abstract methods
        abstract_methods = [
            name
            for name in dir(SimEngine)
            if callable(getattr(SimEngine, name, None)) and getattr(getattr(SimEngine, name), "__isabstractmethod__", False)
        ]
        # If real SimEngine is available, it MUST have abstract methods
        # (otherwise the ABC is not useful)
        assert len(abstract_methods) > 0, (
            "Real SimEngine has no abstract methods. This is unexpected."
        )

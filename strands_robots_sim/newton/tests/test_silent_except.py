"""Test for silent try/except: pass blocks replaced with logger.warning (R2 Thread #4).

Regression test for review thread PRRT_kwDORUMlNs6EBXm_.
Verifies that Newton API version shims log warnings instead of silently passing.
"""

from __future__ import annotations

import inspect
import re


def test_no_silent_try_except_pass_in_newton_api_shims():
    """Verify no silent try/except: pass blocks in Newton API version shims.

    Multiple silent try/except: pass blocks existed for Newton API version
    compatibility (lines 1641, 1661, 1687, 1707, 1727, 1755, 1775).
    These MUST log warnings via logger.warning() to aid debugging.
    """
    from strands_robots_sim.newton.simulation import NewtonSimulation

    # Get source of key methods that do Newton API calls
    methods = [
        "_build_procedural_in_builder",
        "_add_object_to_builder",
    ]

    for method_name in methods:
        method = getattr(NewtonSimulation, method_name)
        source = inspect.getsource(method)

        # Find all try/except blocks
        # Pattern: except <exception>: followed by pass
        pattern = r"except\s+\([^)]+\)(?:\s+as\s+\w+)?:\s*\n\s*pass\s*$"
        matches = list(re.finditer(pattern, source, re.MULTILINE))

        assert len(matches) == 0, (
            f"{method_name} has {len(matches)} silent try/except: pass blocks. "
            f"All Newton API shims MUST log via logger.warning(). "
            f"Matches: {[m.group() for m in matches]}"
        )

        # Verify logger.warning IS used in exception handlers
        # Pattern: except ... as e: ... logger.warning(...)
        has_logger_warning = "logger.warning" in source and "except" in source

        assert has_logger_warning, (
            f"{method_name} has exception handlers but no logger.warning() calls. "
            f"All Newton API version shims MUST log errors."
        )


def test_exception_handlers_capture_exception_object():
    """Verify exception handlers capture exception object for logging.

    Exception handlers MUST use 'as e' to capture the exception for logging.
    Example: except (TypeError, AttributeError) as e:
    """
    from strands_robots_sim.newton.simulation import NewtonSimulation

    methods = [
        "_build_procedural_in_builder",
        "_add_object_to_builder",
    ]

    for method_name in methods:
        method = getattr(NewtonSimulation, method_name)
        source = inspect.getsource(method)

        # Find exception handlers without 'as e'
        # Pattern: except (...): without 'as'
        lines = source.split("\n")
        for i, line in enumerate(lines):
            if "except" in line and ":" in line:
                # If this is an outer except (not nested in a nested try)
                if "except" in line and not line.strip().startswith("#"):
                    # Must have 'as e' or similar
                    if "as " not in line:
                        # Check if it's followed by logger.warning (needs the exception object)
                        next_lines = "\n".join(lines[i : i + 5])
                        if "logger.warning" in next_lines:
                            assert False, (
                                f"{method_name} line {i + 1}: exception handler uses logger.warning "
                                f"but doesn't capture exception with 'as e'. Line: {line.strip()}"
                            )

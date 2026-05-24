"""Regression pin for examples/libero_newton_fleet.py runnability.

Pins the fix from review feedback on PR #30 (R2 thread on broken fleet example):

    "As written, `python examples/libero_newton_fleet.py` will hard-crash before
    reaching the timed run."

The hard-crash was caused by the example calling ``sim.add_object("cube", ...)``
before ``sim.replicate(num_envs)``. The R2 commit (4d7c262) made
``NewtonSimulation.replicate()`` raise ``NotImplementedError`` when any
``add_object()`` object exists in the template -- that contract is correct
(silent-drop was a worse bug), but the fleet example still wired the broken
sequence. This file pins the example's import-graph and AST shape so a future
revert of the surgical fix is caught at lint time, not at user-run time.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

EXAMPLE_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "examples"
    / "libero_newton_fleet.py"
)


@pytest.fixture(scope="module")
def example_source() -> str:
    """Read the libero_newton_fleet.py source verbatim."""
    if not EXAMPLE_PATH.exists():
        pytest.skip(f"example file not present at expected path: {EXAMPLE_PATH}")
    return EXAMPLE_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def example_tree(example_source: str) -> ast.Module:
    """Parse the example to AST."""
    return ast.parse(example_source, filename=str(EXAMPLE_PATH))


def _find_main_function(tree: ast.Module) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            return node
    pytest.fail("examples/libero_newton_fleet.py: no top-level def main()")


def _attribute_calls(node: ast.AST, method_name: str) -> list[ast.Call]:
    """Find all ``something.method_name(...)`` calls under node."""
    return [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == method_name
    ]


def test_example_does_not_call_add_object_before_replicate(example_tree: ast.Module) -> None:
    """Pin: example must not call ``sim.add_object(...)`` anywhere in main().

    ``NewtonSimulation.replicate()`` raises ``NotImplementedError`` when any
    object added via ``add_object()`` exists in the scene (per-env object
    replication is not yet implemented; see
    ``strands_robots_sim/newton/simulation.py::replicate`` and
    ``test_replicate_drops.py::test_replicate_with_add_object_raises_not_implemented``).

    The PR #30 R2 review correctly flagged that the prior version of this
    example did exactly this, hard-crashing every user who ran it. This test
    fails on a regression that re-introduces the broken sequence.
    """
    main_fn = _find_main_function(example_tree)
    add_object_calls = _attribute_calls(main_fn, "add_object")
    assert add_object_calls == [], (
        "examples/libero_newton_fleet.py must not call sim.add_object() before "
        "replicate() -- replicate() will raise NotImplementedError. Either drop the "
        "call (current fix), or wait for per-env object replication and update this "
        "pin in lockstep."
    )


def test_example_calls_replicate_in_main(example_tree: ast.Module) -> None:
    """Pin: example must still demonstrate replicate() (the demo's whole point).

    Sanity check that the surgical fix above did not accidentally remove the
    ``replicate()`` call along with the broken ``add_object()`` line. The
    example's value is the fleet-throughput number, which requires replicate().
    """
    main_fn = _find_main_function(example_tree)
    replicate_calls = _attribute_calls(main_fn, "replicate")
    assert len(replicate_calls) >= 1, (
        "examples/libero_newton_fleet.py must call sim.replicate() -- the example's "
        "purpose is fleet-scale throughput demonstration."
    )


def test_example_compiles_cleanly(example_source: str) -> None:
    """Pin: example file is syntactically valid Python.

    Cheap byte-compile guard against accidental syntax breaks during edits to
    the docstring or the NOTE comment block. Does NOT execute the example
    (that would require a CUDA GPU and the [newton] extra).
    """
    compile(example_source, str(EXAMPLE_PATH), "exec")


def test_example_documents_add_object_omission(example_source: str) -> None:
    """Pin: example must explain why add_object() is omitted.

    Without an inline comment, a future contributor copying the so100-only
    pattern from a tutorial would naturally re-add the object call and
    rediscover the hard-crash. This pin asserts the rationale stays in the
    file so the broken-example fix is not just a silent deletion.
    """
    assert "add_object" in example_source, (
        "examples/libero_newton_fleet.py must mention add_object() in a comment "
        "explaining why it is omitted -- otherwise a future re-add silently "
        "regresses the fix."
    )
    assert "NotImplementedError" in example_source, (
        "examples/libero_newton_fleet.py must mention NotImplementedError in the "
        "explanatory comment so future readers understand the constraint."
    )

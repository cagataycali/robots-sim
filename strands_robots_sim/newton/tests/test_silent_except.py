"""AST-based pin: no silent ``except: pass`` anywhere in newton.simulation.

Closes #35.

Background
==========

The previous regex pin (R2 Thread #4, review `PRRT_kwDORUMlNs6EBXm_`) scoped
the check to two methods (`_build_procedural_in_builder`, `_add_object_to_builder`)
and used the regex::

    r"except\\s+\\([^)]+\\)(?:\\s+as\\s+\\w+)?:\\s*\\n\\s*pass\\s*$"

That regex passed vacuously on three real silent-except sites (issue #35):

1. ``except Exception:`` (un-parenthesized) -- not matched.
2. ``pass  # comment`` -- the `\\s*$` anchor rejects trailing comments, so a
   parenthesized ``except (...)`` followed by ``pass  # ...`` is missed even
   inside the methods the regex did scan.
3. The two-method scope itself missed ``get_observation`` (camera fallback)
   and ``_load_urdf_robot`` (XML-parse fallback).

This file replaces the regex with an ``ast``-based scan that walks every
``ExceptHandler`` in ``simulation.py``, classifies the handler body, and
asserts the body either contains a non-trivial statement or calls one of
``logger.{warning,error,exception}``. It is robust to comments, whitespace,
multi-line ``except`` clauses, un-parenthesized exception types, and
nested handlers.

Carve-out
---------

``__del__`` is exempt from the scan. Calling ``logger.{warning,error,...}``
from a finalizer is itself the anti-pattern documented in the PR #31 thread
on ``IsaacSimulation`` (interpreter shutdown can replace ``logging``,
``threading``, and module globals with ``None`` before ``__del__`` runs).
The ``__del__`` body intentionally swallows everything; the explicit
``cleanup()`` / ``__exit__`` paths are the supported teardown contract.
The exemption is enforced by name (``FunctionDef.name == "__del__"``),
covers exactly one site today (``simulation.py``'s ``NewtonSimulation.__del__``),
and any new ``__del__`` would inherit it deliberately.
"""

from __future__ import annotations

import ast
import inspect
from typing import List, Tuple


def _enclosing_function_name(tree: ast.AST, target: ast.AST) -> str | None:
    """Return the nearest enclosing FunctionDef name for ``target``, or None."""
    enclosing: List[ast.FunctionDef | ast.AsyncFunctionDef] = []
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child._parent = parent  # type: ignore[attr-defined]
    cur = target
    while True:
        parent = getattr(cur, "_parent", None)
        if parent is None:
            return None
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return parent.name
        cur = parent


def _handler_body_is_silent(body: List[ast.stmt]) -> bool:
    """True iff body is logically ``pass`` only (allowing string-literal docstrings).

    Recognises:
    - ``pass`` (the surface anti-pattern)
    - bare string-literal docstrings (e.g. ``"intentional no-op"``)
    Anything else (including ``return``, ``continue``, function calls, assignments)
    counts as non-silent and the handler passes the pin.
    """
    for stmt in body:
        if isinstance(stmt, ast.Pass):
            continue
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(
            stmt.value.value, str
        ):
            continue
        return False
    return True


def _handler_body_logs(body: List[ast.stmt]) -> bool:
    """True iff body calls one of ``logger.{warning, error, exception}``."""
    LOG_METHODS = {"warning", "error", "exception"}
    for stmt in ast.walk(ast.Module(body=body, type_ignores=[])):
        if (
            isinstance(stmt, ast.Call)
            and isinstance(stmt.func, ast.Attribute)
            and stmt.func.attr in LOG_METHODS
        ):
            return True
    return False


def _silent_except_offenders() -> List[Tuple[int, str]]:
    """Return ``(lineno, enclosing_function_name)`` for every silent ExceptHandler
    in ``newton.simulation`` whose enclosing function is not exempt.
    """
    from strands_robots_sim.newton import simulation as mod

    src = inspect.getsource(mod)
    tree = ast.parse(src)

    offenders: List[Tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        if not _handler_body_is_silent(node.body):
            continue
        if _handler_body_logs(node.body):
            continue
        enclosing = _enclosing_function_name(tree, node) or "<module>"
        # Carve-out: __del__ finalizer (see module docstring).
        if enclosing == "__del__":
            continue
        offenders.append((node.lineno, enclosing))
    return offenders


def test_no_silent_except_pass_in_newton_simulation() -> None:
    """No ExceptHandler in newton.simulation may swallow exceptions silently.

    A silent handler is one whose body is ``pass`` (with optional string
    literals) AND whose body does not call ``logger.{warning, error, exception}``.

    Carve-out: ``__del__`` is exempt because logging during interpreter
    finalization is itself unsafe (modules may be torn down).

    Pre-fix verification (issue #35 acceptance criterion #2): on the
    pre-fix ``feat/newton-sim`` HEAD, this test fails listing three
    offenders (``get_observation`` line ~848, ``_build_procedural_in_builder``
    line ~1631, ``_load_urdf_robot`` line ~1741). After the surgical fix
    landed alongside this pin, the offender list is empty.
    """
    offenders = _silent_except_offenders()
    assert offenders == [], (
        "Silent except: pass detected in newton.simulation. Each ExceptHandler "
        "must either contain a non-trivial statement or call one of "
        "logger.warning / logger.error / logger.exception. "
        f"Offenders (line, function): {offenders}"
    )


def test_ast_pin_classifier_recognises_pass_only_handler() -> None:
    """Self-test: the ``pass``-only classifier flags a synthetic offender.

    Guards against a future refactor of the helpers that would let the
    primary pin pass vacuously again (the regression mode that motivated
    issue #35 in the first place).
    """
    src = """
def f():
    try:
        x()
    except Exception:
        pass
"""
    tree = ast.parse(src)
    handlers = [n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler)]
    assert len(handlers) == 1
    assert _handler_body_is_silent(handlers[0].body) is True
    assert _handler_body_logs(handlers[0].body) is False


def test_ast_pin_classifier_recognises_logger_warning_handler() -> None:
    """Self-test: a handler that logs is not flagged."""
    src = """
def f():
    try:
        x()
    except Exception as e:
        logger.warning("x failed: %s", e)
"""
    tree = ast.parse(src)
    handlers = [n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler)]
    assert _handler_body_is_silent(handlers[0].body) is False
    assert _handler_body_logs(handlers[0].body) is True


def test_ast_pin_classifier_handles_pass_with_trailing_comment() -> None:
    """Self-test: the AST view treats ``pass  # comment`` as ``pass`` (comments
    are not in the AST), pinning the exact regex defect that motivated #35."""
    src = """
def f():
    try:
        x()
    except (TypeError, AttributeError):
        pass  # version shim, no-op on older Newton
"""
    tree = ast.parse(src)
    handlers = [n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler)]
    assert _handler_body_is_silent(handlers[0].body) is True
    assert _handler_body_logs(handlers[0].body) is False


def test_ast_pin_classifier_handles_unparenthesized_exception_type() -> None:
    """Self-test: ``except Exception:`` (no parens) is the second regex defect."""
    src = """
def f():
    try:
        x()
    except Exception:
        pass
"""
    tree = ast.parse(src)
    handlers = [n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler)]
    assert _handler_body_is_silent(handlers[0].body) is True


def test_dunder_del_is_exempt_in_helper_classification() -> None:
    """Self-test: the ``__del__`` carve-out is enforced by enclosing-name
    lookup, so a synthetic silent except inside ``__del__`` would be
    skipped by ``_silent_except_offenders``. Documents the carve-out
    contract for future readers.
    """
    src = """
class C:
    def __del__(self):
        try:
            x()
        except Exception:
            pass
"""
    tree = ast.parse(src)
    handlers = [n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler)]
    assert len(handlers) == 1
    assert _handler_body_is_silent(handlers[0].body) is True
    # _silent_except_offenders walks the actual newton.simulation module, not
    # this synthetic source, so we exercise the enclosing-name lookup directly.
    name = _enclosing_function_name(tree, handlers[0])
    assert name == "__del__", (
        f"Carve-out depends on FunctionDef.name lookup; got {name!r}"
    )

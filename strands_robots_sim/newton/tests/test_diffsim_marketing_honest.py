"""Pin tests asserting diff-sim marketing surface is FD-grad honest.

Background
----------
PR #30 R2 review thread asked for the diff-sim docs / docstrings / examples to
be honest about the current implementation: ``NewtonSimulation.run_diffsim`` is
finite-difference-backed, NOT Warp autodiff tape (the originally advertised R13
feature). The fix to ``run_diffsim`` itself (commit 9cf2998) made the docstring
honest; this test pins the broader marketing surface so a future copy-paste
that re-introduces "uses Warp's autodiff tape" claims fails fast.

These pins are non-blocking on actual implementation changes -- they simply
require any "Warp autodiff tape" claim that lands in these files to be paired
with an FD-grad disclaimer in the same paragraph (so a later autodiff-tape
implementation that lifts the disclaimer is the natural fix path).
"""
from __future__ import annotations

from pathlib import Path

import pytest


def _diffsim_module() -> Path:
    return Path(__file__).resolve().parents[1] / "diffsim.py"


def _example_file() -> Path:
    return Path(__file__).resolve().parents[3] / "examples" / "newton_diffsim_toy.py"


def _docs_file() -> Path:
    return Path(__file__).resolve().parents[3] / "docs" / "backends" / "newton.md"


def test_diffsim_module_docstring_does_not_advertise_only_autodiff() -> None:
    """diffsim.py module docstring must not claim it wraps autodiff tape only.

    The optimizer loop is gradient-method-agnostic; the caller chooses whether
    to compute grads via autodiff or finite differences.
    """
    text = _diffsim_module().read_text()
    docstring_end = text.find('"""', 3)
    assert docstring_end > 0, "diffsim.py: module docstring not found"
    docstring = text[:docstring_end]

    forbidden = "Provides high-level wrappers around Warp's autodiff tape"
    assert forbidden not in docstring, (
        f"diffsim.py module docstring still advertises autodiff-only wrappers; "
        f"found: {forbidden!r}. The loop is gradient-method-agnostic; caller "
        f"supplies forward_fn / backward_fn."
    )


def test_diffsim_module_docstring_acknowledges_fd_grad_today() -> None:
    """Module docstring must explicitly say the current run_diffsim is FD-grad."""
    text = _diffsim_module().read_text()
    docstring_end = text.find('"""', 3)
    docstring = text[:docstring_end]

    required_phrases = ["finite-difference", "deferred"]
    missing = [p for p in required_phrases if p not in docstring]
    assert not missing, (
        f"diffsim.py module docstring missing FD-grad disclosure phrases: "
        f"{missing}. Required so reviewers / agents reading the module can tell "
        f"the current implementation from the future autodiff-tape integration."
    )


def test_example_header_does_not_claim_autodiff_tape() -> None:
    """examples/newton_diffsim_toy.py header must not claim it uses Warp autodiff.

    The forward_fn in the example is a closed-form numpy expression, not a sim
    rollout. Using "Warp's autodiff tape" here misrepresents both the helper
    and the example.
    """
    if not _example_file().exists():
        pytest.skip("newton_diffsim_toy.py not present in this checkout")
    text = _example_file().read_text()
    header_end = text.find('"""', 3)
    assert header_end > 0, "newton_diffsim_toy.py: header docstring not found"
    header = text[:header_end]

    forbidden = "Demonstrates trajectory optimization using Warp's autodiff tape."
    assert forbidden not in header, (
        f"newton_diffsim_toy.py header still claims Warp autodiff tape; the "
        f"example's forward_fn is a closed-form numpy expression and "
        f"NewtonSimulation.run_diffsim uses finite differences. Update the "
        f"header to match. Found: {forbidden!r}"
    )


def test_example_header_acknowledges_fd_grad() -> None:
    """Header must explicitly disclose FD-grad + closed-form forward_fn."""
    if not _example_file().exists():
        pytest.skip("newton_diffsim_toy.py not present in this checkout")
    text = _example_file().read_text()
    header_end = text.find('"""', 3)
    header = text[:header_end]

    required_phrases = ["finite-difference", "closed-form"]
    missing = [p for p in required_phrases if p not in header]
    assert not missing, (
        f"newton_diffsim_toy.py header missing FD-grad disclosure phrases: "
        f"{missing}. A reader running this example must be able to tell from "
        f"the header that it does not exercise true autodiff physics today."
    )


def test_docs_section_not_titled_only_differentiable() -> None:
    """docs/backends/newton.md diff-sim section must reflect FD-grad reality.

    The previous title "Differentiable Simulation" + body "Enable Warp's
    autodiff tape" misled readers about today's implementation.
    """
    if not _docs_file().exists():
        pytest.skip("docs/backends/newton.md not present in this checkout")
    text = _docs_file().read_text()

    forbidden_pair = (
        "## Differentiable Simulation\n\nEnable Warp's autodiff tape for trajectory optimization:"
    )
    assert forbidden_pair not in text, (
        "docs/backends/newton.md still advertises 'Enable Warp's autodiff tape' "
        "directly under the diff-sim section header. Update to reflect the "
        "FD-grad current implementation; mention autodiff tape only as "
        "deferred future work."
    )


def test_docs_diffsim_section_acknowledges_fd_grad() -> None:
    """The diff-sim section in docs must mention FD-grad + deferral."""
    if not _docs_file().exists():
        pytest.skip("docs/backends/newton.md not present in this checkout")
    text = _docs_file().read_text()

    # Find a header that mentions either gradient-based optimization or differentiable
    # and assert the FD-grad disclaimer is present in the same doc.
    section_anchors = [
        "## Gradient-Based Simulation Optimization",
        "## Differentiable Simulation",
    ]
    assert any(a in text for a in section_anchors), (
        f"docs/backends/newton.md missing diff-sim section header. Looked for "
        f"any of: {section_anchors}"
    )
    assert "finite-difference" in text and "deferred" in text, (
        "docs/backends/newton.md must mention 'finite-difference' and "
        "'deferred' so readers can tell the current diff-sim implementation "
        "(FD-grad) from the future autodiff-tape integration."
    )

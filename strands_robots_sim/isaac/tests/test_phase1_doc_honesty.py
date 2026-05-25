"""Documentation honesty pin: G1 DOF count + Phase 1 doc banner.

Background
==========

R2 review on PR #31 surfaced two documentation-honesty drifts in the Isaac
Phase 1 skeleton:

1. Procedural G1 docstring / inline comment claimed "29 DOF" but the actual
   ``g1()`` joint set is 21 (1 torso + 6 left leg + 6 right leg + 4 left arm
   + 4 right arm). ``__init__.py`` and ``docs/backends/isaac.md`` both
   already advertise 21-DOF; only ``procedural.py`` was stale.

2. ``docs/backends/isaac.md`` Quick Start did not disclose that the Phase 1
   skeleton silently no-ops the data plane -- ``add_robot`` on the
   procedural branch, ``_load_usd_robot`` / ``_load_urdf_robot``,
   ``add_object``, ``add_camera``, ``replicate`` all return
   ``status: "success"`` without instantiating the underlying USD prim or
   articulation handle. A user following the doc on a real Isaac Sim install
   gets cheerful success strings and an empty observation.

Both drifts are documentation issues, not behavioural bugs -- the data-plane
wiring lands in Phase 2 and later. This pin guards against the comment /
docstring / banner drifting back out of sync with the code under future
refactors. It does NOT pin the kinematic-tree topology defect on the G1
joint graph (duplicate ``(parent_body, child_body)`` edges); that is a
Phase 2 fix that needs intermediate massless link bodies and is
documented in ``procedural.py`` line 165 as deferred.
"""

from __future__ import annotations

import re
from pathlib import Path

_PROCEDURAL_PY = Path(__file__).resolve().parent.parent / "procedural.py"
_ISAAC_DOCS = Path(__file__).resolve().parent.parent.parent.parent / "docs" / "backends" / "isaac.md"


class TestG1DOFCount:
    """Pin: G1 doc-string / inline-comment DOF count must match the joint set."""

    def test_g1_actual_joint_count_is_21(self) -> None:
        """The shipped G1 procedural definition has exactly 21 joints."""
        from strands_robots_sim.isaac.procedural import get_procedural_robot

        robot = get_procedural_robot("unitree_g1")
        assert robot.num_joints == 21, (
            f"unitree_g1 has {robot.num_joints} joints; if this changes, the "
            f"DOF count in procedural.py docstrings/comments and "
            f"docs/backends/isaac.md must be updated together."
        )

    def test_g1_module_docstring_advertises_21_not_29(self) -> None:
        """Module docstring must not claim 29-DOF (stale -- actual is 21)."""
        text = _PROCEDURAL_PY.read_text(encoding="utf-8")
        # Look only at the module-level docstring (everything before the first
        # `from __future__` import, which is the canonical top-of-module
        # marker for this file).
        head = text.split("from __future__", 1)[0]
        assert "29-DOF" not in head and "29 DOF" not in head, (
            "procedural.py module docstring still claims 29-DOF for G1; the "
            "actual joint count is 21 (verified by test_g1_actual_joint_count_is_21)."
        )
        assert "21-DOF" in head or "21 DOF" in head, (
            "procedural.py module docstring no longer mentions the actual "
            "21-DOF count for G1; documentation must stay in sync with code."
        )

    def test_g1_builder_docstring_advertises_21_not_29(self) -> None:
        """``_build_unitree_g1`` docstring must not claim 29-DOF."""
        text = _PROCEDURAL_PY.read_text(encoding="utf-8")
        # Find the def / docstring window for _build_unitree_g1 specifically.
        match = re.search(
            r"def\s+_build_unitree_g1\b.*?(?=\ndef |\Z)",
            text,
            flags=re.DOTALL,
        )
        assert match is not None, "could not locate _build_unitree_g1 in procedural.py"
        body = match.group(0)
        assert "29-DOF" not in body and "29 DOF" not in body, (
            "_build_unitree_g1 still mentions 29-DOF; the actual joint count "
            "is 21 (verified by test_g1_actual_joint_count_is_21)."
        )
        assert "21-DOF" in body or "21 DOF" in body, "_build_unitree_g1 no longer documents its actual 21-DOF count."


class TestIsaacDocsPhase1Banner:
    """Pin: ``docs/backends/isaac.md`` must disclose Phase 1 data-plane no-ops."""

    def test_isaac_docs_file_exists(self) -> None:
        assert _ISAAC_DOCS.is_file(), f"missing Isaac doc page at {_ISAAC_DOCS}"

    def test_phase1_banner_present_before_installation(self) -> None:
        """Banner must appear before the Installation / Quick Start sections.

        Reviewer (R2 on PR #31, ``simulation.py:627`` thread): "At minimum,
        please add a note to ``docs/backends/isaac.md`` Quick Start that the
        Phase-1 skeleton silently no-ops the data plane."
        """
        text = _ISAAC_DOCS.read_text(encoding="utf-8")

        # The banner must appear AND must appear before the Installation
        # section header (so the disclosure precedes any procedural docs the
        # user would otherwise execute).
        banner_marker = "Phase 1 status"
        install_marker = "## Installation"

        assert banner_marker in text, (
            "docs/backends/isaac.md missing the Phase 1 status disclosure "
            "banner. R2 reviewer asked for it explicitly because the doc's "
            "Quick Start otherwise executes a code path that silently no-ops "
            "on a real Isaac Sim install."
        )
        assert install_marker in text, (
            "docs/backends/isaac.md missing the Installation section -- " "doc structure has changed; pin needs review."
        )
        assert text.find(banner_marker) < text.find(install_marker), (
            "Phase 1 banner must precede the Installation section so the "
            "user sees the disclosure before following the install / quick-"
            "start steps."
        )

    def test_phase1_banner_names_the_silent_methods(self) -> None:
        """Banner must enumerate the Phase-1 silent-success methods.

        Without naming the methods, a future maintainer who reads only the
        banner won't know which API surfaces are affected, and the disclosure
        becomes a vague hedge.
        """
        text = _ISAAC_DOCS.read_text(encoding="utf-8")
        # Slice the banner block (`> **Phase 1 status...**` paragraph).
        banner_start = text.find("Phase 1 status")
        # Banner is one paragraph; cut at the next `## ` heading.
        banner_end = text.find("##", banner_start)
        assert banner_end > banner_start, "could not locate end of banner block"
        banner = text[banner_start:banner_end]

        for needed in ("add_robot", "replicate", "get_observation"):
            assert needed in banner, (
                f"Phase 1 banner does not mention `{needed}`; the disclosure "
                f"must enumerate the silent-success methods so users know "
                f"which call sites are affected."
            )

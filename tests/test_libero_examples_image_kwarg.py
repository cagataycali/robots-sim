"""Static regression guard for robots-sim#118.

``strands-robots>=0.4.0`` removed the ``gr00t_inference(image_name=...)``
keyword argument: the GR00T container image is operator-configured via the
``STRANDS_GR00T_IMAGE`` environment variable (validated against
``STRANDS_GR00T_IMAGE_ALLOW``) and resolved internally by the tool. The four
LIBERO examples used to pass ``image_name=args.image``, which raised
``TypeError: gr00t_inference() got an unexpected keyword argument 'image_name'``
on every ``--policy groot`` run, on both the MuJoCo and Isaac backends.

These checks are intentionally static (stdlib ``ast`` only): the hatch test
env has ``skip-install = true`` and does not install ``strands-robots``, and
the real path needs Docker + an NVIDIA GPU. Parsing the example source pins
the regression without importing the heavy dependency or touching hardware.
"""

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLES_DIR = _REPO_ROOT / "examples" / "libero"

# The four scripts that orchestrate the GR00T inference container via
# ``gr00t_inference(action="lifecycle", ...)``.
_EXAMPLE_FILES = [
    "run_mujoco.py",
    "run_mujoco_agent.py",
    "run_isaac.py",
    "run_isaac_agent.py",
]


def _parse(name: str) -> ast.Module:
    path = _EXAMPLES_DIR / name
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _gr00t_inference_calls(tree: ast.Module) -> list[ast.Call]:
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "gr00t_inference":
                calls.append(node)
    return calls


@pytest.mark.parametrize("name", _EXAMPLE_FILES)
def test_no_image_name_kwarg(name: str) -> None:
    """No gr00t_inference call may pass the removed ``image_name`` kwarg."""
    tree = _parse(name)
    calls = _gr00t_inference_calls(tree)
    assert calls, f"{name}: expected at least one gr00t_inference(...) call"
    offenders = [kw.arg for call in calls for kw in call.keywords if kw.arg == "image_name"]
    assert not offenders, (
        f"{name}: gr00t_inference(image_name=...) is rejected by "
        f"strands-robots>=0.4.0; route --image through STRANDS_GR00T_IMAGE instead"
    )


@pytest.mark.parametrize("name", _EXAMPLE_FILES)
def test_routes_image_to_env(name: str) -> None:
    """The --image flag must be routed to os.environ['STRANDS_GR00T_IMAGE']."""
    tree = _parse(name)
    found = False
    for node in ast.walk(tree):
        # Match: os.environ["STRANDS_GR00T_IMAGE"] = ...
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Attribute)
                    and target.value.attr == "environ"
                    and isinstance(target.value.value, ast.Name)
                    and target.value.value.id == "os"
                    and isinstance(target.slice, ast.Constant)
                    and target.slice.value == "STRANDS_GR00T_IMAGE"
                ):
                    found = True
    assert found, (
        f"{name}: expected os.environ['STRANDS_GR00T_IMAGE'] = args.image "
        f"to route the --image flag through the 0.4.0 image mechanism"
    )

"""Procedural robot builders for Newton simulation.

When a user calls ``sim.add_robot("so100")`` without an explicit URDF path,
we build the robot procedurally using Newton's ModelBuilder API. This avoids
asset-path dependencies and enables instant testing.

Supported procedural robots:
    - ``so100``: SO-100 6-DOF arm (SO-ARM100 from HuggingFace/lerobot)
    - ``unitree_g1``: Unitree G1 humanoid (simplified 23-DOF)
    - ``panda``: Franka Panda 7-DOF (classic benchmark arm)

For robots not in this registry, ``add_robot()`` falls back to URDF loading.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class JointDef:
    """Definition of a single joint for procedural building."""

    name: str
    parent_body: int
    child_body: int
    joint_type: str = "revolute"  # revolute, prismatic, fixed
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    limit_lower: float = -math.pi
    limit_upper: float = math.pi
    damping: float = 0.5
    stiffness: float = 0.0
    armature: float = 0.01


@dataclass
class BodyDef:
    """Definition of a single rigid body for procedural building."""

    name: str
    mass: float = 0.1
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)  # quaternion xyzw
    shape: str = "capsule"  # box, sphere, capsule, cylinder
    shape_size: tuple[float, ...] = (0.02, 0.05)  # depends on shape type
    color: tuple[float, float, float] = (0.5, 0.5, 0.5)


@dataclass
class ProceduralRobot:
    """Complete procedural robot definition."""

    name: str
    bodies: list[BodyDef] = field(default_factory=list)
    joints: list[JointDef] = field(default_factory=list)
    base_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    base_orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    is_floating_base: bool = False

    @property
    def num_joints(self) -> int:
        """Number of actuated joints (excludes fixed joints)."""
        return sum(1 for j in self.joints if j.joint_type != "fixed")

    @property
    def joint_names(self) -> list[str]:
        """Ordered list of actuated joint names."""
        return [j.name for j in self.joints if j.joint_type != "fixed"]


def build_so100() -> ProceduralRobot:
    """Build SO-100 6-DOF arm procedurally.

    The SO-100 (SO-ARM100) is a low-cost 6-DOF desktop manipulator with:
    - 5 revolute joints for arm positioning
    - 1 revolute joint for gripper

    Kinematic chain: base → shoulder → upper_arm → forearm → wrist → hand → gripper
    """
    bodies = [
        BodyDef(name="base", mass=1.0, position=(0.0, 0.0, 0.0), shape="cylinder", shape_size=(0.04, 0.02)),
        BodyDef(name="shoulder", mass=0.3, position=(0.0, 0.0, 0.05), shape="capsule", shape_size=(0.02, 0.03)),
        BodyDef(name="upper_arm", mass=0.2, position=(0.0, 0.0, 0.12), shape="capsule", shape_size=(0.015, 0.06)),
        BodyDef(name="forearm", mass=0.15, position=(0.0, 0.0, 0.22), shape="capsule", shape_size=(0.012, 0.05)),
        BodyDef(name="wrist", mass=0.1, position=(0.0, 0.0, 0.30), shape="capsule", shape_size=(0.01, 0.03)),
        BodyDef(name="hand", mass=0.08, position=(0.0, 0.0, 0.35), shape="box", shape_size=(0.03, 0.02, 0.02)),
        BodyDef(name="gripper", mass=0.05, position=(0.0, 0.0, 0.38), shape="box", shape_size=(0.025, 0.015, 0.01)),
    ]

    joints = [
        JointDef(
            name="shoulder_pan",
            parent_body=0,
            child_body=1,
            axis=(0.0, 0.0, 1.0),
            limit_lower=-math.pi,
            limit_upper=math.pi,
            damping=1.0,
        ),
        JointDef(
            name="shoulder_lift",
            parent_body=1,
            child_body=2,
            axis=(0.0, 1.0, 0.0),
            limit_lower=-math.pi / 2,
            limit_upper=math.pi / 2,
            damping=0.8,
        ),
        JointDef(
            name="elbow_flex",
            parent_body=2,
            child_body=3,
            axis=(0.0, 1.0, 0.0),
            limit_lower=-math.pi * 0.75,
            limit_upper=math.pi * 0.75,
            damping=0.6,
        ),
        JointDef(
            name="wrist_flex",
            parent_body=3,
            child_body=4,
            axis=(0.0, 1.0, 0.0),
            limit_lower=-math.pi / 2,
            limit_upper=math.pi / 2,
            damping=0.4,
        ),
        JointDef(
            name="wrist_roll",
            parent_body=4,
            child_body=5,
            axis=(0.0, 0.0, 1.0),
            limit_lower=-math.pi,
            limit_upper=math.pi,
            damping=0.3,
        ),
        JointDef(
            name="gripper",
            parent_body=5,
            child_body=6,
            joint_type="revolute",
            axis=(1.0, 0.0, 0.0),
            limit_lower=0.0,
            limit_upper=math.pi / 4,
            damping=0.2,
        ),
    ]

    return ProceduralRobot(name="so100", bodies=bodies, joints=joints)


def build_panda() -> ProceduralRobot:
    """Build Franka Panda 7-DOF arm procedurally.

    Classic 7-DOF benchmark manipulator used widely in robotics research.
    """
    bodies = [
        BodyDef(name="base", mass=2.0, position=(0.0, 0.0, 0.0), shape="cylinder", shape_size=(0.05, 0.03)),
        BodyDef(name="link1", mass=2.34, position=(0.0, 0.0, 0.1525), shape="capsule", shape_size=(0.03, 0.07)),
        BodyDef(name="link2", mass=2.36, position=(0.0, 0.0, 0.305), shape="capsule", shape_size=(0.03, 0.07)),
        BodyDef(name="link3", mass=2.38, position=(0.0, 0.0, 0.42), shape="capsule", shape_size=(0.025, 0.06)),
        BodyDef(name="link4", mass=2.43, position=(0.0, 0.0, 0.535), shape="capsule", shape_size=(0.025, 0.06)),
        BodyDef(name="link5", mass=3.5, position=(0.0, 0.0, 0.65), shape="capsule", shape_size=(0.025, 0.06)),
        BodyDef(name="link6", mass=1.47, position=(0.0, 0.0, 0.75), shape="capsule", shape_size=(0.02, 0.04)),
        BodyDef(name="link7", mass=0.8, position=(0.0, 0.0, 0.84), shape="capsule", shape_size=(0.02, 0.03)),
        BodyDef(name="hand", mass=0.73, position=(0.0, 0.0, 0.89), shape="box", shape_size=(0.04, 0.04, 0.02)),
    ]

    joints = [
        JointDef(name="joint1", parent_body=0, child_body=1, axis=(0, 0, 1), limit_lower=-2.8973, limit_upper=2.8973),
        JointDef(name="joint2", parent_body=1, child_body=2, axis=(0, 1, 0), limit_lower=-1.7628, limit_upper=1.7628),
        JointDef(name="joint3", parent_body=2, child_body=3, axis=(0, 0, 1), limit_lower=-2.8973, limit_upper=2.8973),
        JointDef(name="joint4", parent_body=3, child_body=4, axis=(0, 1, 0), limit_lower=-3.0718, limit_upper=-0.0698),
        JointDef(name="joint5", parent_body=4, child_body=5, axis=(0, 0, 1), limit_lower=-2.8973, limit_upper=2.8973),
        JointDef(name="joint6", parent_body=5, child_body=6, axis=(0, 1, 0), limit_lower=-0.0175, limit_upper=3.7525),
        JointDef(name="joint7", parent_body=6, child_body=7, axis=(0, 0, 1), limit_lower=-2.8973, limit_upper=2.8973),
        JointDef(
            name="finger_joint",
            parent_body=7,
            child_body=8,
            joint_type="prismatic",
            axis=(0, 1, 0),
            limit_lower=0.0,
            limit_upper=0.04,
        ),
    ]

    return ProceduralRobot(name="panda", bodies=bodies, joints=joints)


def build_unitree_g1() -> ProceduralRobot:
    """Build simplified Unitree G1 humanoid procedurally.

    23-DOF simplified humanoid for locomotion / whole-body manipulation tasks.
    Floating base (6 DOF) + 17 actuated joints.
    """
    bodies = [
        BodyDef(name="pelvis", mass=8.0, position=(0.0, 0.0, 0.85), shape="box", shape_size=(0.15, 0.1, 0.1)),
        BodyDef(name="torso", mass=12.0, position=(0.0, 0.0, 1.1), shape="box", shape_size=(0.2, 0.12, 0.2)),
        BodyDef(name="head", mass=3.0, position=(0.0, 0.0, 1.4), shape="sphere", shape_size=(0.08,)),
        # Left leg
        BodyDef(name="l_hip", mass=2.0, position=(-0.08, 0.0, 0.8), shape="sphere", shape_size=(0.04,)),
        BodyDef(name="l_thigh", mass=4.0, position=(-0.08, 0.0, 0.55), shape="capsule", shape_size=(0.035, 0.15)),
        BodyDef(name="l_shin", mass=2.5, position=(-0.08, 0.0, 0.3), shape="capsule", shape_size=(0.03, 0.15)),
        BodyDef(name="l_foot", mass=1.0, position=(-0.08, 0.0, 0.05), shape="box", shape_size=(0.1, 0.06, 0.02)),
        # Right leg
        BodyDef(name="r_hip", mass=2.0, position=(0.08, 0.0, 0.8), shape="sphere", shape_size=(0.04,)),
        BodyDef(name="r_thigh", mass=4.0, position=(0.08, 0.0, 0.55), shape="capsule", shape_size=(0.035, 0.15)),
        BodyDef(name="r_shin", mass=2.5, position=(0.08, 0.0, 0.3), shape="capsule", shape_size=(0.03, 0.15)),
        BodyDef(name="r_foot", mass=1.0, position=(0.08, 0.0, 0.05), shape="box", shape_size=(0.1, 0.06, 0.02)),
        # Left arm
        BodyDef(name="l_shoulder", mass=1.5, position=(-0.2, 0.0, 1.2), shape="sphere", shape_size=(0.03,)),
        BodyDef(name="l_upper_arm", mass=1.8, position=(-0.2, 0.0, 1.0), shape="capsule", shape_size=(0.025, 0.1)),
        BodyDef(name="l_forearm", mass=1.2, position=(-0.2, 0.0, 0.8), shape="capsule", shape_size=(0.02, 0.1)),
        # Right arm
        BodyDef(name="r_shoulder", mass=1.5, position=(0.2, 0.0, 1.2), shape="sphere", shape_size=(0.03,)),
        BodyDef(name="r_upper_arm", mass=1.8, position=(0.2, 0.0, 1.0), shape="capsule", shape_size=(0.025, 0.1)),
        BodyDef(name="r_forearm", mass=1.2, position=(0.2, 0.0, 0.8), shape="capsule", shape_size=(0.02, 0.1)),
    ]

    joints = [
        # Torso
        JointDef(name="torso_yaw", parent_body=0, child_body=1, axis=(0, 0, 1), limit_lower=-1.0, limit_upper=1.0),
        JointDef(
            name="neck_pitch",
            parent_body=1,
            child_body=2,
            axis=(0, 1, 0),
            limit_lower=-0.5,
            limit_upper=0.5,
            damping=0.1,
        ),
        # Left leg (3 DOF simplified)
        JointDef(name="l_hip_yaw", parent_body=0, child_body=3, axis=(0, 0, 1), limit_lower=-0.5, limit_upper=0.5),
        JointDef(name="l_hip_pitch", parent_body=3, child_body=4, axis=(0, 1, 0), limit_lower=-1.5, limit_upper=1.5),
        JointDef(name="l_knee", parent_body=4, child_body=5, axis=(0, 1, 0), limit_lower=-2.5, limit_upper=0.0),
        JointDef(name="l_ankle", parent_body=5, child_body=6, axis=(0, 1, 0), limit_lower=-0.8, limit_upper=0.8),
        # Right leg (3 DOF simplified)
        JointDef(name="r_hip_yaw", parent_body=0, child_body=7, axis=(0, 0, 1), limit_lower=-0.5, limit_upper=0.5),
        JointDef(name="r_hip_pitch", parent_body=7, child_body=8, axis=(0, 1, 0), limit_lower=-1.5, limit_upper=1.5),
        JointDef(name="r_knee", parent_body=8, child_body=9, axis=(0, 1, 0), limit_lower=-2.5, limit_upper=0.0),
        JointDef(name="r_ankle", parent_body=9, child_body=10, axis=(0, 1, 0), limit_lower=-0.8, limit_upper=0.8),
        # Left arm (3 DOF simplified)
        JointDef(
            name="l_shoulder_pitch", parent_body=1, child_body=11, axis=(0, 1, 0), limit_lower=-3.14, limit_upper=1.0
        ),
        JointDef(
            name="l_shoulder_roll",
            parent_body=11,
            child_body=12,
            axis=(1, 0, 0),
            limit_lower=-1.5,
            limit_upper=1.5,
        ),
        JointDef(name="l_elbow", parent_body=12, child_body=13, axis=(0, 1, 0), limit_lower=-2.5, limit_upper=0.0),
        # Right arm (3 DOF simplified)
        JointDef(
            name="r_shoulder_pitch", parent_body=1, child_body=14, axis=(0, 1, 0), limit_lower=-3.14, limit_upper=1.0
        ),
        JointDef(
            name="r_shoulder_roll",
            parent_body=14,
            child_body=15,
            axis=(1, 0, 0),
            limit_lower=-1.5,
            limit_upper=1.5,
        ),
        JointDef(name="r_elbow", parent_body=15, child_body=16, axis=(0, 1, 0), limit_lower=-2.5, limit_upper=0.0),
    ]

    return ProceduralRobot(
        name="unitree_g1",
        bodies=bodies,
        joints=joints,
        base_position=(0.0, 0.0, 0.85),
        is_floating_base=True,
    )


# Registry of procedural robot builders
_PROCEDURAL_ROBOTS: dict[str, Callable[[], ProceduralRobot]] = {
    "so100": build_so100,
    "so_arm100": build_so100,
    "so-100": build_so100,
    "panda": build_panda,
    "franka": build_panda,
    "franka_panda": build_panda,
    "unitree_g1": build_unitree_g1,
    "g1": build_unitree_g1,
}


def get_procedural_robot(name: str) -> ProceduralRobot | None:
    """Look up a procedural robot by name.

    Parameters
    ----------
    name : str
        Robot name or alias.

    Returns
    -------
    ProceduralRobot or None
        The procedural definition, or None if not found.
    """
    builder = _PROCEDURAL_ROBOTS.get(name.lower())
    if builder is None:
        return None
    return builder()


def list_procedural_robots() -> list[str]:
    """Return list of canonical procedural robot names (no aliases)."""
    seen = set()
    result = []
    for name, builder in _PROCEDURAL_ROBOTS.items():
        robot = builder()
        if robot.name not in seen:
            seen.add(robot.name)
            result.append(robot.name)
    return sorted(result)

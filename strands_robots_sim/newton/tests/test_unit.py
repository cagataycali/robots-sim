"""Unit tests for Newton backend (no GPU required).

Run with: pytest strands_robots_sim/newton/tests/test_unit.py -v
"""

# Tests are defined in __init__.py to keep things in one file during initial development.
# Import and re-export for pytest discovery.

from strands_robots_sim.newton.tests import (
    TestDiffSim,
    TestNewtonConfig,
    TestNewtonSimulation,
    TestNewtonSimulationIntegration,
    TestProceduralRobots,
    TestSolvers,
)

__all__ = [
    "TestNewtonConfig",
    "TestSolvers",
    "TestProceduralRobots",
    "TestDiffSim",
    "TestNewtonSimulation",
    "TestNewtonSimulationIntegration",
]

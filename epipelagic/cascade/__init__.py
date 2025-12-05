"""
Cascade solvers for turbulent energy transfer dynamics.

This module provides:
- ShellCascade: N-shell cascade model with configurable interactions
- CascadeSolver: Time integration and steady-state computation
- Phase diagram tools: Parameter space exploration
"""

from epipelagic.cascade.shell_model import ShellCascade
from epipelagic.cascade.solver import CascadeSolver
from epipelagic.cascade.phase_diagram import compute_phase_diagram

__all__ = [
    "ShellCascade",
    "CascadeSolver",
    "compute_phase_diagram",
]

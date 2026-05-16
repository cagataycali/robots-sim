"""Differentiable simulation helpers for Newton/Warp.

Provides high-level wrappers around Warp's autodiff tape for:
1. Trajectory optimization (``run_diffsim``)
2. System identification
3. Differentiable rendering (future)

These helpers are used by :meth:`NewtonSimulation.run_diffsim` and
exposed for advanced users who need fine-grained control over the
optimization loop.

Requires ``config.enable_differentiable = True`` to work.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DiffSimResult:
    """Result of a differentiable simulation optimization run.

    Parameters
    ----------
    converged : bool
        Whether the optimization converged (loss below threshold).
    iterations : int
        Number of optimization iterations completed.
    final_loss : float
        Final loss value.
    loss_history : list[float]
        Loss at each iteration.
    optimized_params : dict[str, Any]
        Final optimized parameter values.
    wall_time : float
        Wall-clock time in seconds.
    grad_norms : list[float]
        Gradient norm at each iteration (for debugging).
    """

    converged: bool = False
    iterations: int = 0
    final_loss: float = float("inf")
    loss_history: list[float] = field(default_factory=list)
    optimized_params: dict[str, Any] = field(default_factory=dict)
    wall_time: float = 0.0
    grad_norms: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "converged": self.converged,
            "iterations": self.iterations,
            "final_loss": self.final_loss,
            "loss_history": self.loss_history,
            "optimized_params": {
                k: v.tolist() if hasattr(v, "tolist") else v for k, v in self.optimized_params.items()
            },
            "wall_time": self.wall_time,
            "grad_norms": self.grad_norms,
        }


@dataclass
class DiffSimConfig:
    """Configuration for differentiable simulation optimization.

    Parameters
    ----------
    num_steps : int
        Number of simulation steps per forward pass.
    lr : float
        Learning rate for the optimizer. Default 0.02.
    iterations : int
        Maximum optimization iterations. Default 200.
    convergence_threshold : float
        Stop when loss drops below this. Default 1e-6.
    grad_clip : float
        Maximum gradient norm (for stability). Default 10.0.
    optimizer : str
        Optimizer type: "adam" or "sgd". Default "adam".
    verbose : bool
        Print progress every N iterations. Default False.
    print_interval : int
        Iterations between progress prints. Default 10.
    """

    num_steps: int = 100
    lr: float = 0.02
    iterations: int = 200
    convergence_threshold: float = 1e-6
    grad_clip: float = 10.0
    optimizer: str = "adam"
    verbose: bool = False
    print_interval: int = 10


def _numpy_sgd_step(params: dict[str, np.ndarray], grads: dict[str, np.ndarray], lr: float) -> None:
    """Simple SGD step on numpy arrays (in-place)."""
    for key in params:
        if key in grads:
            params[key] = params[key] - lr * grads[key]


def _numpy_adam_step(
    params: dict[str, np.ndarray],
    grads: dict[str, np.ndarray],
    m: dict[str, np.ndarray],
    v: dict[str, np.ndarray],
    lr: float,
    t: int,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> None:
    """Adam optimizer step on numpy arrays (in-place)."""
    for key in params:
        if key not in grads:
            continue
        g = grads[key]
        m[key] = beta1 * m.get(key, np.zeros_like(g)) + (1 - beta1) * g
        v[key] = beta2 * v.get(key, np.zeros_like(g)) + (1 - beta2) * g * g
        m_hat = m[key] / (1 - beta1**t)
        v_hat = v[key] / (1 - beta2**t)
        params[key] = params[key] - lr * m_hat / (np.sqrt(v_hat) + eps)


def run_diffsim_loop(
    forward_fn: Callable[[dict[str, Any]], float],
    backward_fn: Callable[[dict[str, Any]], dict[str, np.ndarray]],
    initial_params: dict[str, np.ndarray],
    config: DiffSimConfig,
) -> DiffSimResult:
    """Run a differentiable simulation optimization loop.

    This is the CPU-side orchestration loop. The actual forward and backward
    passes are delegated to Warp kernel execution inside ``forward_fn`` and
    ``backward_fn``.

    Parameters
    ----------
    forward_fn : callable
        Takes params dict, returns scalar loss.
    backward_fn : callable
        Takes params dict, returns grads dict (same keys as params).
    initial_params : dict
        Initial parameter values as numpy arrays.
    config : DiffSimConfig
        Optimization configuration.

    Returns
    -------
    DiffSimResult
        Optimization results including final params and loss history.
    """
    t0 = time.time()
    params = {k: v.copy() for k, v in initial_params.items()}
    result = DiffSimResult()

    # Adam state
    m: dict[str, np.ndarray] = {}
    v: dict[str, np.ndarray] = {}

    for iteration in range(1, config.iterations + 1):
        # Forward pass
        loss = forward_fn(params)
        result.loss_history.append(float(loss))

        # Check convergence
        if loss < config.convergence_threshold:
            result.converged = True
            result.iterations = iteration
            result.final_loss = float(loss)
            break

        # Backward pass
        grads = backward_fn(params)

        # Compute gradient norm
        grad_norm = 0.0
        for g in grads.values():
            grad_norm += float(np.sum(g * g))
        grad_norm = float(np.sqrt(grad_norm))
        result.grad_norms.append(grad_norm)

        # Gradient clipping
        if grad_norm > config.grad_clip:
            scale = config.grad_clip / (grad_norm + 1e-8)
            grads = {k: v * scale for k, v in grads.items()}

        # Optimizer step
        if config.optimizer == "adam":
            _numpy_adam_step(params, grads, m, v, config.lr, iteration)
        else:
            _numpy_sgd_step(params, grads, config.lr)

        # Logging
        if config.verbose and iteration % config.print_interval == 0:
            logger.info(
                "DiffSim iter %d/%d: loss=%.6f grad_norm=%.4f",
                iteration,
                config.iterations,
                loss,
                grad_norm,
            )

    else:
        # Did not converge within max iterations
        result.iterations = config.iterations
        result.final_loss = float(result.loss_history[-1]) if result.loss_history else float("inf")

    result.optimized_params = params
    result.wall_time = time.time() - t0
    return result


def compute_finite_difference_gradients(
    forward_fn: Callable[[dict[str, Any]], float],
    params: dict[str, np.ndarray],
    epsilon: float = 1e-4,
) -> dict[str, np.ndarray]:
    """Compute gradients via finite differences (fallback when Warp tape unavailable).

    Parameters
    ----------
    forward_fn : callable
        Forward function (params → scalar loss).
    params : dict
        Current parameter values.
    epsilon : float
        Finite difference step size.

    Returns
    -------
    dict[str, np.ndarray]
        Gradient for each parameter.
    """
    grads = {}
    base_loss = forward_fn(params)

    for key, value in params.items():
        flat_value = value.flatten()
        grad_flat = np.zeros_like(flat_value)
        for i in range(len(flat_value)):
            perturbed = dict(params)
            perturbed_val = flat_value.copy()
            perturbed_val[i] += epsilon
            perturbed[key] = perturbed_val.reshape(value.shape)
            perturbed_loss = forward_fn(perturbed)
            grad_flat[i] = (perturbed_loss - base_loss) / epsilon
        grads[key] = grad_flat.reshape(value.shape)

    return grads

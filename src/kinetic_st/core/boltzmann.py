"""Mesoscopic kinetic equation solver via operator splitting.

Integrates df/dt = jump + (1/tau)*Q(f,f) on a spatial graph, where Q is
the collision integral realized by CollisionKernel.
"""

import numpy as np
from scipy import sparse
from typing import Optional, Dict, Callable
from dataclasses import dataclass, field

from kinetic_st.core.collision import CollisionKernel


@dataclass
class KineticConfig:
    """Solver configuration."""
    tau: float = 1.0
    jump_rate: float = 0.1
    dt: float = 0.05
    n_steps: int = 200
    drift_field: Optional[Callable] = None
    n_pairs_per_spot: int = 4
    record_every: int = 10
    seed: int = 0


@dataclass
class KineticTrajectory:
    """Recorded trajectory."""
    states: np.ndarray              # (n_spots, d) final states
    history: np.ndarray             # (T, n_spots, d)
    times: np.ndarray               # (T,)
    info: Dict = field(default_factory=dict)


class BoltzmannSolver:
    """Integrate the cell-state kinetic equation via operator splitting.

    Steps per time step: streaming (spatial jump) -> collision -> drift.
    """

    def __init__(self, kernel: CollisionKernel, config: Optional[KineticConfig] = None):
        self.kernel = kernel
        self.config = config or KineticConfig()
        self.trajectory_ = None

    def solve(self, states0, adjacency, Y_lr=None, verbose=False) -> KineticTrajectory:
        """Integrate forward in time."""
        cfg = self.config
        rng = np.random.RandomState(cfg.seed)
        states = np.array(states0, dtype=np.float64, copy=True)
        n, d = states.shape

        A = adjacency.tocsr().astype(np.float64)
        deg = np.maximum(np.asarray(A.sum(axis=1)).flatten(), 1.0)
        P = (sparse.diags(1.0 / deg) @ A).tocsr()

        history = [states.copy()]
        times = [0.0]
        for step in range(cfg.n_steps):
            t = (step + 1) * cfg.dt
            # Streaming
            states += cfg.dt * cfg.jump_rate * (P @ states - states)
            # Collision
            collided = self.kernel.apply_collisions(
                states, adjacency, Y_lr=Y_lr,
                n_pairs_per_spot=cfg.n_pairs_per_spot, rng=rng)
            states += (cfg.dt / cfg.tau) * (collided - states)
            # Drift
            if cfg.drift_field is not None:
                states += cfg.dt * cfg.drift_field(states, t)
            if (step + 1) % cfg.record_every == 0:
                history.append(states.copy())
                times.append(t)
                if verbose and (step + 1) % (cfg.record_every * 10) == 0:
                    print(f"  [Boltzmann] t={t:.3f} step={step+1}/{cfg.n_steps}")

        self.trajectory_ = KineticTrajectory(
            states=states, history=np.array(history), times=np.array(times),
            info={"n_steps": cfg.n_steps, "dt": cfg.dt, "n_spots": n, "dim": d})
        return self.trajectory_


def solve_kinetic(states0, adjacency, kernel, config=None, Y_lr=None) -> KineticTrajectory:
    """Functional API for BoltzmannSolver.solve."""
    return BoltzmannSolver(kernel, config).solve(states0, adjacency, Y_lr=Y_lr)

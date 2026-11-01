"""Time-slicing for bounded-cost long-time kinetic integration.

Partitions [0, T] into K slices, integrates locally on each, and applies
moment renormalization at boundaries to discard negligible cross-slice
correlations.
"""

import numpy as np
from scipy import sparse
from typing import Optional, Dict, Callable, List
from dataclasses import dataclass, field

from kinetic_st.core.collision import CollisionKernel
from kinetic_st.core.boltzmann import KineticConfig, KineticTrajectory, BoltzmannSolver


@dataclass
class TimeSliceConfig:
    """Time-slicing configuration."""
    slice_length: float = 1.0
    n_slices: int = 20
    dt: float = 0.05
    record_every: int = 5
    renormalize: str = "moment"  # or "none"
    n_pairs_per_spot: int = 4
    jump_rate: float = 0.1
    tau: float = 1.0
    drift_field: Optional[Callable] = None
    seed: int = 0


@dataclass
class SlicedTrajectory:
    """Result of time-sliced integration."""
    states: np.ndarray
    history: np.ndarray
    times: np.ndarray
    slice_boundaries: np.ndarray
    slice_info: List[Dict] = field(default_factory=list)
    info: Dict = field(default_factory=dict)


class TimeSlicer:
    """Long-time integration via temporal slicing with renormalization."""

    def __init__(self, kernel: CollisionKernel, config: Optional[TimeSliceConfig] = None):
        self.kernel = kernel
        self.config = config or TimeSliceConfig()
        self.trajectory_ = None

    def integrate(self, states0, adjacency, Y_lr=None, verbose=False) -> SlicedTrajectory:
        cfg = self.config
        states = np.array(states0, dtype=np.float64, copy=True)
        n, d = states.shape

        n_steps_slice = max(1, int(round(cfg.slice_length / cfg.dt)))
        slice_config = KineticConfig(
            tau=cfg.tau, jump_rate=cfg.jump_rate, dt=cfg.dt,
            n_steps=n_steps_slice, drift_field=cfg.drift_field,
            n_pairs_per_spot=cfg.n_pairs_per_spot,
            record_every=cfg.record_every, seed=cfg.seed)
        solver = BoltzmannSolver(self.kernel, slice_config)

        boundaries = [0.0]
        all_history = [states.copy()]
        all_times = [0.0]
        slice_info = []
        current = states

        for k in range(cfg.n_slices):
            traj = solver.solve(current, adjacency, Y_lr=Y_lr, verbose=False)
            all_history.append(traj.history[1:])
            all_times.append(traj.times[1:] + boundaries[-1])
            boundaries.append(boundaries[-1] + cfg.slice_length)
            current = traj.states.copy()

            if cfg.renormalize == "moment":
                current = _moment_renorm(current, adjacency)

            slice_info.append({"slice": k, "t_start": boundaries[-2],
                               "t_end": boundaries[-1], "n_steps": n_steps_slice})
            if verbose:
                print(f"  [TimeSlicer] slice {k+1}/{cfg.n_slices} "
                      f"t=[{boundaries[-2]:.2f},{boundaries[-1]:.2f}]")

        self.trajectory_ = SlicedTrajectory(
            states=current,
            history=np.concatenate([all_history[0][None]] + all_history[1:], axis=0),
            times=np.concatenate([[all_times[0]]] + all_times[1:]),
            slice_boundaries=np.array(boundaries), slice_info=slice_info,
            info={"n_slices": cfg.n_slices, "slice_length": cfg.slice_length,
                  "renormalize": cfg.renormalize, "n_spots": n, "dim": d})
        return self.trajectory_


def _moment_renorm(states, adjacency, n_neighbors=8):
    """Moment renormalization at slice boundaries.

    Replaces each state by the local-neighborhood mean, rescaled to preserve
    global variance. Discards fine-grained cross-cell correlations.
    """
    A = adjacency.tocsr()
    n, d = states.shape
    global_mean = states.mean(axis=0)
    global_std = states.std(axis=0) + 1e-8

    local_mean = states.copy()
    for i in range(n):
        idx = A.indices[A.indptr[i]:A.indptr[i + 1]][:n_neighbors]
        nbrs = np.concatenate([[i], idx])
        local_mean[i] = states[nbrs].mean(axis=0)

    smoothed_std = local_mean.std(axis=0) + 1e-8
    return global_mean + (local_mean - local_mean.mean(axis=0)) * (global_std / smoothed_std)


def reconstruct_from_snapshots(snapshots, snapshot_times, adjacency, kernel,
                                config=None) -> SlicedTrajectory:
    """Reconstruct a long-time trajectory from sparse temporal snapshots.

    Fills dynamics between consecutive snapshots by running the time-sliced
    solver initialized at each snapshot.
    """
    m = len(snapshot_times)
    cfg = config or TimeSliceConfig()

    boundaries = [float(snapshot_times[0])]
    all_history = [snapshots[0].copy()]
    all_times = [float(snapshot_times[0])]
    slice_info = []
    current = snapshots[0].copy()

    for k in range(m - 1):
        dt_span = float(snapshot_times[k + 1] - snapshot_times[k])
        if dt_span <= 0:
            continue
        n_slices_gap = max(1, int(np.ceil(dt_span / cfg.slice_length)))
        local_cfg = TimeSliceConfig(
            slice_length=dt_span / n_slices_gap, n_slices=n_slices_gap,
            dt=cfg.dt, record_every=cfg.record_every,
            renormalize=cfg.renormalize, n_pairs_per_spot=cfg.n_pairs_per_spot,
            jump_rate=cfg.jump_rate, tau=cfg.tau,
            drift_field=cfg.drift_field, seed=cfg.seed + k)
        traj = TimeSlicer(kernel, local_cfg).integrate(current, adjacency, verbose=False)
        all_history.append(traj.history[1:])
        all_times.append(traj.times[1:])
        boundaries.extend(traj.slice_boundaries[1:].tolist())
        current = snapshots[k + 1].copy()
        slice_info.append({"segment": k, "t_start": float(snapshot_times[k]),
                           "t_end": float(snapshot_times[k + 1]),
                           "n_slices": n_slices_gap})

    return SlicedTrajectory(
        states=snapshots[-1].copy(),
        history=np.concatenate([all_history[0][None]] + all_history[1:], axis=0),
        times=np.concatenate([[all_times[0]]] + all_times[1:]),
        slice_boundaries=np.array(boundaries), slice_info=slice_info,
        info={"reconstructed": True, "n_snapshots": m})

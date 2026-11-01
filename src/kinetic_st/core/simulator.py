"""Tissue-level expression field simulator with kinetic ground truth.

Forward-simulates the kinetic equation to produce synthetic ST data.
Composes spatial layout, attractors, collision kernel, kinetic integration,
decoding to gene space, and Poisson + dropout measurement.
"""

import numpy as np
from scipy import sparse
from typing import Optional
from dataclasses import dataclass, field

from kinetic_st.core.collision import CollisionKernel, LigandReceptorRule
from kinetic_st.core.boltzmann import BoltzmannSolver, KineticConfig
from kinetic_st.core.stochastic_pde import StochasticKineticSolver, StochasticConfig
from kinetic_st.core.time_slicing import TimeSlicer, TimeSliceConfig
from kinetic_st.core.fluid_limit import FluidLimiter
from kinetic_st.core.graph_utils import coords_to_adjacency


@dataclass
class SimConfig:
    """Simulator configuration."""
    n_types: int = 6
    n_genes: int = 2000
    n_spots: int = 2500
    latent_dim: int = 12
    grid_side: int = 50
    alpha: float = 0.1
    beta: float = 0.25
    gamma: float = 0.08
    tau: float = 1.0
    jump_rate: float = 0.1
    diffusion: float = 0.03
    dt: float = 0.05
    n_steps: int = 200
    library_mean: float = 7.5
    dropout: float = 0.3
    use_time_slicing: bool = False
    seed: int = 42


@dataclass
class SimResult:
    """Simulated data with kinetic ground truth."""
    Y: np.ndarray
    coords: np.ndarray
    gene_names: np.ndarray
    labels: np.ndarray
    latent_states: np.ndarray
    attractors: np.ndarray
    adjacency: sparse.spmatrix
    trajectory: object
    fields: list = field(default_factory=list)
    embedder: object = None
    config: SimConfig = None


class TissueSimulator:
    """Generate synthetic ST data with a kinetic ground truth."""

    def __init__(self, config: Optional[SimConfig] = None):
        self.config = config or SimConfig()
        self.result_ = None

    def run(self, verbose: bool = False) -> SimResult:
        cfg = self.config
        rng = np.random.RandomState(cfg.seed)
        d = cfg.latent_dim

        # Spatial layout
        side = int(np.sqrt(cfg.n_spots))
        xx, yy = np.meshgrid(np.arange(side), np.arange(side))
        coords = np.column_stack([xx.ravel(), yy.ravel()])[:cfg.n_spots]
        if len(coords) < cfg.n_spots:
            coords = np.vstack([coords, rng.uniform(0, side, (cfg.n_spots - len(coords), 2))])
        n_spots = len(coords)
        A = coords_to_adjacency(coords, method="knn", k=6)

        # Attractors + spatially-biased initial assignment
        attractors = rng.standard_normal((cfg.n_types, d)) * 2.0
        from sklearn.cluster import KMeans
        spatial_labels = KMeans(n_clusters=cfg.n_types, random_state=cfg.seed, n_init=5).fit(coords).labels_
        states0 = attractors[spatial_labels] + rng.standard_normal((n_spots, d)) * 0.3

        # Decoder: attractor marker profiles + latent contribution
        marker_profiles = rng.lognormal(mean=2.0, sigma=0.5, size=(cfg.n_types, cfg.n_genes))
        decoder_W = rng.standard_normal((d, cfg.n_genes)) * 0.1

        def decode(states):
            d2 = ((states[:, None, :] - attractors[None, :, :]) ** 2).sum(axis=2)
            w = np.exp(-0.5 * d2 / 0.5)
            w /= w.sum(axis=1, keepdims=True) + 1e-8
            return (w @ marker_profiles) * np.exp(0.1 * states @ decoder_W)

        # Collision kernel + kinetic integration
        rule = LigandReceptorRule(alpha=cfg.alpha, beta=cfg.beta,
                                   gamma=cfg.gamma, attractors=attractors)
        kernel = CollisionKernel(rule, sigma_scale=1.0)

        if cfg.use_time_slicing:
            traj = TimeSlicer(kernel, TimeSliceConfig(
                slice_length=1.0, n_slices=max(1, cfg.n_steps // 40),
                dt=cfg.dt, jump_rate=cfg.jump_rate, tau=cfg.tau, seed=cfg.seed
            )).integrate(states0, A, verbose=verbose)
        else:
            traj = StochasticKineticSolver(kernel, StochasticConfig(
                tau=cfg.tau, jump_rate=cfg.jump_rate, dt=cfg.dt,
                n_steps=cfg.n_steps, diffusion=cfg.diffusion, seed=cfg.seed
            )).solve(states0, A, verbose=verbose)

        # Decode to gene expression and sample counts
        expression = decode(traj.states)
        lib_sizes = rng.lognormal(mean=cfg.library_mean, sigma=0.4, size=n_spots)
        rates = np.maximum(expression / expression.sum(axis=1, keepdims=True) * lib_sizes[:, None], 1e-3)
        Y = rng.poisson(rates).astype(np.float64)
        Y[rng.random(Y.shape) < cfg.dropout] = 0

        labels = np.argmin(((traj.states[:, None, :] - attractors[None, :, :]) ** 2).sum(axis=2), axis=1)
        fields = FluidLimiter(n_neighbors=10).compute(traj, A)

        self.result_ = SimResult(
            Y=Y, coords=coords, gene_names=np.array([f"Gene_{i}" for i in range(cfg.n_genes)]),
            labels=labels, latent_states=traj.states, attractors=attractors,
            adjacency=A, trajectory=traj, fields=fields, config=cfg)
        if verbose:
            print(f"  [TissueSimulator] {n_spots} spots, {cfg.n_genes} genes, {cfg.n_types} types")
        return self.result_


def simulate_tissue(config: Optional[SimConfig] = None) -> SimResult:
    """Functional API for TissueSimulator.run."""
    return TissueSimulator(config).run()

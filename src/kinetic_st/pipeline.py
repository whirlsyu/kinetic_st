"""High-level KineticST pipeline: micro -> meso -> macro.

Orchestrates latent embedding, collision kernel, kinetic integration,
moment closure, invariant measure estimation, and domain identification.
"""

import numpy as np
from scipy import sparse
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass

from kinetic_st.core.latent import LatentEmbedding
from kinetic_st.core.collision import CollisionKernel, LigandReceptorRule
from kinetic_st.core.boltzmann import BoltzmannSolver, KineticConfig
from kinetic_st.core.stochastic_pde import StochasticKineticSolver, StochasticConfig, InvariantMeasureEstimator
from kinetic_st.core.time_slicing import TimeSlicer, TimeSliceConfig
from kinetic_st.core.fluid_limit import FluidLimiter, TissueField
from kinetic_st.core.domains import DomainIdentifier, DomainResult
from kinetic_st.core.wave_turbulence import WaveKineticAnalyzer, WaveKineticResult
from kinetic_st.core.graph_utils import coords_to_adjacency


@dataclass
class KineticSTConfig:
    """Pipeline configuration."""
    latent_dim: int = 12
    alpha: float = 0.1
    beta: float = 0.2
    gamma: float = 0.05
    sigma_scale: float = 1.0
    solver: str = "stochastic"  # "boltzmann", "stochastic", or "sliced"
    tau: float = 1.0
    jump_rate: float = 0.1
    dt: float = 0.05
    n_steps: int = 200
    diffusion: float = 0.03
    slice_length: float = 1.0
    n_slices: int = 20
    spatial_method: str = "knn"
    k_neighbors: int = 6
    n_domains: Union[int, str] = "auto"
    run_wave_analysis: bool = True
    verbose: bool = False
    seed: int = 0


class KineticST:
    """End-to-end kinetic-theory pipeline for spatial transcriptomics.

    Parameters
    ----------
    config : KineticSTConfig, optional

    Attributes
    ----------
    embedder_, kernel_, trajectory_, fields_, invariant_, domains_, wave_
    """

    def __init__(self, config: Optional[KineticSTConfig] = None):
        self.config = config or KineticSTConfig()
        self._fitted = False

    def fit(self, Y, coords: np.ndarray,
            attractors: Optional[np.ndarray] = None,
            Y_lr: Optional[np.ndarray] = None) -> "KineticST":
        """Run the full kinetic pipeline.

        Parameters
        ----------
        Y : (n_spots, n_genes) expression matrix
        coords : (n_spots, 2) spatial coordinates
        attractors : (n_types, latent_dim) cell-type centroids; None = KMeans estimate
        Y_lr : ligand-receptor expression for cross-section
        """
        cfg = self.config
        n_spots, n_genes = Y.shape
        if cfg.verbose:
            print(f"KineticST: {n_spots} spots x {n_genes} genes")
            print(f"  solver={cfg.solver} latent_dim={cfg.latent_dim} n_steps={cfg.n_steps}")

        # Latent embedding
        if cfg.verbose:
            print("[1/6] Latent embedding...")
        self.embedder_ = LatentEmbedding(n_components=cfg.latent_dim, method="pca")
        Z = self.embedder_.fit_transform(Y)
        self.latent_states_ = Z

        # Spatial graph
        if cfg.verbose:
            print("[2/6] Spatial graph...")
        self.adjacency_ = coords_to_adjacency(coords, method=cfg.spatial_method, k=cfg.k_neighbors)

        # Collision kernel
        if cfg.verbose:
            print("[3/6] Collision kernel...")
        if attractors is None:
            from sklearn.cluster import KMeans
            k = max(2, min(15, n_spots // 50))
            attractors = KMeans(n_clusters=k, random_state=cfg.seed, n_init=3).fit(Z).cluster_centers_
        self.attractors_ = attractors
        rule = LigandReceptorRule(alpha=cfg.alpha, beta=cfg.beta, gamma=cfg.gamma,
                                   attractors=attractors)
        self.kernel_ = CollisionKernel(rule, sigma_scale=cfg.sigma_scale)

        # Kinetic integration
        if cfg.verbose:
            print(f"[4/6] Kinetic integration ({cfg.solver})...")
        if cfg.solver == "boltzmann":
            self.trajectory_ = BoltzmannSolver(
                self.kernel_, KineticConfig(
                    tau=cfg.tau, jump_rate=cfg.jump_rate, dt=cfg.dt,
                    n_steps=cfg.n_steps, seed=cfg.seed)
            ).solve(Z, self.adjacency_, Y_lr=Y_lr, verbose=cfg.verbose)
        elif cfg.solver == "stochastic":
            self.trajectory_ = StochasticKineticSolver(
                self.kernel_, StochasticConfig(
                    tau=cfg.tau, jump_rate=cfg.jump_rate, dt=cfg.dt,
                    n_steps=cfg.n_steps, diffusion=cfg.diffusion, seed=cfg.seed)
            ).solve(Z, self.adjacency_, Y_lr=Y_lr, verbose=cfg.verbose)
        elif cfg.solver == "sliced":
            self.trajectory_ = TimeSlicer(
                self.kernel_, TimeSliceConfig(
                    slice_length=cfg.slice_length, n_slices=cfg.n_slices,
                    dt=cfg.dt, jump_rate=cfg.jump_rate, tau=cfg.tau, seed=cfg.seed)
            ).integrate(Z, self.adjacency_, Y_lr=Y_lr, verbose=cfg.verbose)
        else:
            raise ValueError(f"Unknown solver: {cfg.solver}")

        # Tissue fields + invariant measure
        if cfg.verbose:
            print("[5/6] Tissue fields & invariant measure...")
        self.fields_ = FluidLimiter(n_neighbors=cfg.k_neighbors).compute(
            self.trajectory_, self.adjacency_)
        self.relaxation_ = FluidLimiter(n_neighbors=cfg.k_neighbors).relaxation_indicator(self.trajectory_)
        self.invariant_ = InvariantMeasureEstimator(
            representation="gmm", n_components=max(2, len(attractors)), seed=cfg.seed
        ).fit(self.trajectory_)

        # Spatial domains
        if cfg.verbose:
            print("[6/6] Spatial domains...")
        self.domains_ = DomainIdentifier(
            n_neighbors=cfg.k_neighbors, n_domains=cfg.n_domains,
            representation="gmm", n_components=max(2, len(attractors)), seed=cfg.seed
        ).fit(self.trajectory_, self.adjacency_, verbose=cfg.verbose)

        # Wave analysis
        self.wave_ = self._run_wave_analysis(coords) if cfg.run_wave_analysis else None

        self._fitted = True
        if cfg.verbose:
            print("KineticST: done.")
        return self

    def _run_wave_analysis(self, coords):
        analyzer = WaveKineticAnalyzer(n_k_bins=15)
        try:
            return analyzer.analyze(self.fields_, coords, component=0,
                                    verbose=self.config.verbose)
        except Exception as e:
            if self.config.verbose:
                print(f"  [wave analysis skipped: {e}]")
            return None

    def get_expression_field(self, time_idx: int = -1) -> np.ndarray:
        """Macroscopic expression field at a recorded time index."""
        if not self._fitted:
            raise RuntimeError("Not fitted.")
        return self.fields_[time_idx].mean_state

    def get_domain_labels(self) -> np.ndarray:
        """Spatial domain labels."""
        if not self._fitted:
            raise RuntimeError("Not fitted.")
        return self.domains_.labels

    def decode_to_genes(self, states: Optional[np.ndarray] = None) -> np.ndarray:
        """Decode latent states back to gene-expression space."""
        if not self._fitted:
            raise RuntimeError("Not fitted.")
        return self.embedder_.inverse_transform(states or self.trajectory_.states)

    def summary(self) -> Dict[str, Any]:
        """Model summary."""
        if not self._fitted:
            return {"fitted": False}
        s = {"fitted": True, "n_spots": self.latent_states_.shape[0],
             "n_genes": self.embedder_.mean_.shape[0],
             "latent_dim": self.config.latent_dim, "solver": self.config.solver,
             "n_attractors": len(self.attractors_),
             "n_domains": self.domains_.n_domains,
             "n_recorded_times": len(self.fields_),
             "mean_relaxation": float(np.mean(self.relaxation_))}
        if self.wave_ is not None:
            s["wave_cascade"] = self.wave_.cascade_direction
            s["wave_slope"] = self.wave_.spectral_slope
        return s

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        return f"KineticST(solver='{self.config.solver}', latent_dim={self.config.latent_dim}, status={status})"

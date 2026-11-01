"""KineticST: kinetic-theory framework for spatial transcriptomics.

Bridges microscopic cell-cell interactions to macroscopic tissue organisation
via a Boltzmann-like collision operator on latent cell states, integrated on
a spatial graph with stochastic dynamics and invariant-measure analysis.
"""

__version__ = "0.1.0"
__author__ = "KineticST Project Team"

from kinetic_st.core.latent import LatentEmbedding, fit_latent
from kinetic_st.core.collision import CollisionKernel, LigandReceptorRule
from kinetic_st.core.boltzmann import BoltzmannSolver, solve_kinetic
from kinetic_st.core.fluid_limit import FluidLimiter, compute_moments
from kinetic_st.core.stochastic_pde import StochasticKineticSolver, InvariantMeasureEstimator
from kinetic_st.core.time_slicing import TimeSlicer
from kinetic_st.core.domains import DomainIdentifier
from kinetic_st.core.wave_turbulence import WaveKineticAnalyzer
from kinetic_st.core.simulator import TissueSimulator
from kinetic_st.core.markers import KineticMarkerFinder, find_kinetic_markers, markers_from_model
from kinetic_st.data.simulation import generate_kinetic_simulation
from kinetic_st.pipeline import KineticST, KineticSTConfig

__all__ = [
    "KineticST", "KineticSTConfig",
    "LatentEmbedding", "fit_latent",
    "CollisionKernel", "LigandReceptorRule",
    "BoltzmannSolver", "solve_kinetic",
    "FluidLimiter", "compute_moments",
    "StochasticKineticSolver", "InvariantMeasureEstimator",
    "TimeSlicer", "DomainIdentifier", "WaveKineticAnalyzer",
    "TissueSimulator", "KineticMarkerFinder", "find_kinetic_markers",
    "markers_from_model", "generate_kinetic_simulation",
]

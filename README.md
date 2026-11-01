# KineticST

Kinetic-theory framework for multiscale modelling of spatial transcriptomics.

Bridges microscopic cell-cell interactions to macroscopic tissue organisation
via a Boltzmann-like collision operator on latent cell states, integrated on
a spatial graph with stochastic dynamics and invariant-measure analysis.

## Installation

```bash
pip install kinetic-st
```

For development (editable install):

```bash
pip install -e .
```

## Quick start

```python
import numpy as np
from kinetic_st import KineticST, KineticSTConfig, generate_kinetic_simulation

# Synthetic data
sim = generate_kinetic_simulation(n_types=6, n_spots=1600, seed=42)

# Fit
model = KineticST(KineticSTConfig(n_steps=150, latent_dim=10, verbose=True))
model.fit(sim.Y, sim.coords)

# Results
labels = model.get_domain_labels()
field = model.get_expression_field()
print(model.summary())
```

## Pipeline stages

1. **Latent embedding** — PCA on log-CPM expression to a low-dimensional cell-state space
2. **Spatial graph** — kNN adjacency from coordinates
3. **Collision kernel** — ligand-receptor rule with irreversible attractor bias
4. **Kinetic integration** — deterministic (Boltzmann), stochastic (Euler-Maruyama), or time-sliced solver
5. **Moment closure** — macroscopic tissue expression fields via local KDE moments
6. **Invariant measure** — non-equilibrium steady-state estimation (KDE or GMM)
7. **Spatial domains** — segmentation via local invariant-measure divergence
8. **Supervised markers** — one-vs-rest z-scores on the dynamics-smoothed field

## License

MIT

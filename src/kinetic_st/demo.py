"""Demo: end-to-end KineticST pipeline on synthetic ST data."""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from kinetic_st import KineticST, KineticSTConfig, generate_kinetic_simulation


def plot_domains(coords, labels, title, save_path, cmap_name="tab10"):
    """Plot spatial layout colored by labels."""
    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = plt.colormaps.get_cmap(cmap_name)
    uniq = np.unique(labels)
    n = len(uniq)
    for k in uniq:
        mask = labels == k
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[cmap(k / max(n - 1, 1))], s=10, marker="s",
                   label=f"{int(k)}", alpha=0.8, edgecolors="none")
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=13)
    ax.legend(loc="upper right", fontsize=6, ncol=3, markerscale=1.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_field(coords, field, title, save_path):
    """Plot a continuous tissue field."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=field, s=12,
                    marker="s", cmap="viridis", edgecolors="none")
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=13)
    plt.colorbar(sc, ax=ax, shrink=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    output_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 60)
    print("KineticST Demo")
    print("=" * 60)

    print("\n[1] Generating synthetic data...")
    sim = generate_kinetic_simulation(
        n_types=6, n_genes=1500, n_spots=1600,
        latent_dim=10, n_steps=150, seed=42, verbose=True)
    Y, coords, true_labels = sim.Y, sim.coords, sim.labels

    plot_domains(coords, true_labels, "Ground truth",
                 os.path.join(output_dir, "demo_01_ground_truth.png"))

    print("\n[2] Running KineticST pipeline...")
    model = KineticST(KineticSTConfig(
        latent_dim=10, solver="stochastic", n_steps=150,
        diffusion=0.03, alpha=0.1, beta=0.2, gamma=0.05,
        n_domains=sim.config.n_types, run_wave_analysis=True,
        verbose=True, seed=0))
    model.fit(Y, coords)

    print("\n[3] Evaluating domains...")
    pred_labels = model.get_domain_labels()
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    print(f"    ARI = {ari:.4f}  NMI = {nmi:.4f}")

    plot_domains(coords, pred_labels, "KineticST domains",
                 os.path.join(output_dir, "demo_02_recovered_domains.png"))

    print("\n[4] Tissue field...")
    plot_field(coords, model.get_expression_field()[:, 0],
               "Expression field (latent dim 0)",
               os.path.join(output_dir, "demo_03_expression_field.png"))

    print("\n[5] Wave analysis...")
    if model.wave_ is not None:
        print(f"    Cascade: {model.wave_.cascade_direction}, Slope: {model.wave_.spectral_slope:.4f}")
        k = model.wave_.k_range
        n_mean = np.mean([s.spectrum for s in model.wave_.spectra], axis=0)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.loglog(k[1:], n_mean[1:], "o-", markersize=4)
        ax.set_xlabel("wave number |k|")
        ax.set_ylabel("wave action n(k)")
        ax.set_title("Wave-kinetic spectrum")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "demo_04_wave_spectrum.png"),
                    dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print("  Saved: demo_04_wave_spectrum.png")

    print("\n" + "=" * 60)
    s = model.summary()
    for k, v in s.items():
        print(f"  {k:20s}: {v}")
    print(f"\n  ARI={ari:.4f}, NMI={nmi:.4f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate the camera-ready Mip-NeRF 360 speed/LPIPS teaser."""

from pathlib import Path

import matplotlib.pyplot as plt


METHODS = {
    "3DGS": (31.93, 0.221, 2.63),
    "3DGS-MCMC": (32.41, 0.186, 3.23),
    "Mini-Splatting": (28.69, 0.217, 0.53),
    "Speedy-splat": (24.38, 0.295, 0.30),
    "Taming-3DGS": (16.36, 0.261, 0.68),
    "DashGaussian": (17.35, 0.218, 2.40),
    "FastGS-big": (14.58, 0.216, 1.15),
    "GloSplat-A": (22.00, 0.139, 3.00),
    "VGGT-X": (73.71, 0.177, 3.00),
    "SalientGS": (11.79, 0.148, 1.50),
}

COLORS = {
    "3DGS": "#9aa0a6",
    "3DGS-MCMC": "#4c9ed9",
    "Mini-Splatting": "#9b59b6",
    "Speedy-splat": "#2fb9a3",
    "Taming-3DGS": "#ed8b2f",
    "DashGaussian": "#36b47e",
    "FastGS-big": "#f2a92f",
    "GloSplat-A": "#4c9bc0",
    "VGGT-X": "#6f7d8c",
    "SalientGS": "#ef6255",
}

OFFSETS = {
    "3DGS": (-8, 10),
    "3DGS-MCMC": (8, -10),
    "Mini-Splatting": (-10, -12),
    "Speedy-splat": (8, 8),
    "Taming-3DGS": (8, 8),
    "DashGaussian": (8, 8),
    "FastGS-big": (-32, -12),
    "GloSplat-A": (8, -12),
    "SalientGS": (8, 10),
}


def main() -> None:
    output = Path(__file__).with_name("teaser_lpips_speed.pdf")
    fig, ax = plt.subplots(figsize=(5.2, 3.65))
    for name, (time_min, lpips, num_gs) in METHODS.items():
        if name == "VGGT-X":
            continue
        size = 55 + 90 * num_gs
        edge = "black" if name == "SalientGS" else "white"
        width = 1.8 if name == "SalientGS" else 0.8
        ax.scatter(time_min, lpips, s=size, color=COLORS[name], alpha=0.95,
                   edgecolor=edge, linewidth=width, zorder=3)
        dx, dy = OFFSETS[name]
        ax.annotate(name + (" (Ours)" if name == "SalientGS" else ""),
                    (time_min, lpips), xytext=(dx, dy), textcoords="offset points",
                    fontsize=7.3, color=COLORS[name],
                    fontweight="bold" if name == "SalientGS" else "normal")

    # VGGT-X is far outside the useful 10--40 minute range; retain it explicitly.
    ax.annotate("VGGT-X: 73.7 min, 0.177 (off-axis)", xy=(40.2, 0.177),
                xytext=(30.0, 0.166), fontsize=6.7, color=COLORS["VGGT-X"],
                arrowprops={"arrowstyle": "-|>", "color": COLORS["VGGT-X"], "lw": 1.0})

    for n in (0.5, 1.5, 3.0):
        ax.scatter([], [], s=55 + 90 * n, color="#b8b8b8", alpha=0.8,
                   edgecolor="white", label=f"{n:.1f}M")
    ax.legend(title="# Gaussians", loc="upper right", frameon=True,
              fontsize=6.8, title_fontsize=7.0, borderpad=0.6, labelspacing=0.8)
    ax.set_xlim(9.5, 41.5)
    ax.set_ylim(0.115, 0.315)
    ax.set_xlabel("End-to-End Time (minutes)", fontweight="bold", fontsize=9)
    ax.set_ylabel("LPIPS (lower is better)", fontweight="bold", fontsize=9)
    ax.set_title("Quality vs. Speed vs. Model Size on Mip-NeRF 360",
                 fontsize=10, fontweight="bold")
    ax.grid(True, linestyle="--", linewidth=0.55, alpha=0.45)
    ax.tick_params(labelsize=7.5)
    ax.annotate("Better quality", xy=(10.8, 0.27), xytext=(10.8, 0.302),
                fontsize=7, ha="center", arrowprops={"arrowstyle": "->", "lw": 1})
    ax.annotate("Faster", xy=(34.5, 0.126), xytext=(39.5, 0.126),
                fontsize=7, va="center", arrowprops={"arrowstyle": "->", "lw": 1})
    fig.tight_layout(pad=0.6)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

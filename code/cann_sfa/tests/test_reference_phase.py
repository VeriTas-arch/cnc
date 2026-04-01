"""Standalone reproduction of ref/oscillatory/fig_2C.m."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cann_sfa_triple_regimes import RegimeConfig, simulate_regime

REFERENCE_FIG2C = RegimeConfig(
    name="fig2c_reference",
    num=512,
    tau=3.0,
    tau_v=144.0,
    g=5.0,
    k=5.0,
    a=0.4,
    j0=1.0 / 5.0,
    m=80.0 / 48.0,
    dt=3.0 / 10.0,
    alpha=0.14,
    v_ext=4.36 / 3.0 * 1e-3,
    target_pos0=-(np.pi / 2.0 + np.pi / 8.0),
    total_duration=27000.0,
    analysis_start=4500.0,
    display_start=4500.0,
)


def classify_reference_state(result) -> tuple[int, float]:
    """Match ref/oscillatory/omega_simulate.m."""

    config = result["config"]
    start_idx = int(round(config.analysis_start / config.dt))
    relative = np.asarray(result["relative"][start_idx:], dtype=np.float64)
    if len(relative) == 0:
        return 0, 0.0

    data = relative - np.mean(relative)
    span = float(np.max(data) - np.min(data))

    if span > 1.0:
        return 4, 0.0

    if span > 0.1:
        fL = np.fft.fft(data)
        fL_shift = np.fft.fftshift(fL)
        power = np.abs(fL_shift) ** 2 / len(data)
        fs = 1000.0 / config.dt
        fshift = (-len(data) / 2.0 + np.arange(len(data))) * (fs / len(data))
        max_idx = int(np.argmax(power))
        return 3, float(abs(fshift[max_idx]))

    if np.sum(relative) < 0.0:
        return 1, 0.0
    return 2, 0.0


def run_reference_point(alpha: float, m: float):
    config = replace(REFERENCE_FIG2C, alpha=alpha, m=m)
    result = simulate_regime(config)
    state, freq_hz = classify_reference_state(result)
    return result, state, freq_hz


def scan_reference_fig2c(alpha_values, m_values):
    state = np.zeros((len(alpha_values), len(m_values)), dtype=np.int8)
    omega = np.zeros_like(state, dtype=np.float64)
    total = len(alpha_values) * len(m_values)
    counter = 0
    for i, alpha in enumerate(alpha_values):
        for j, m in enumerate(m_values):
            counter += 1
            print(f"[fig2c {counter}/{total}] alpha={alpha:.4f}, m={m:.4f}", flush=True)
            _, state[i, j], omega[i, j] = run_reference_point(alpha, m)
    return state, omega


def plot_reference_fig2c(alpha_values, m_values, state, save_path: Path, show: bool = False):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    image = ax.imshow(
        state,
        origin="lower",
        aspect="auto",
        extent=[m_values[0], m_values[-1], alpha_values[0], alpha_values[-1]],
        interpolation="nearest",
        cmap="viridis",
        vmin=1,
        vmax=4,
    )
    cbar = fig.colorbar(image, ax=ax, ticks=[1, 2, 3, 4])
    cbar.ax.set_yticklabels(["Delayed", "Anticipative", "Oscillatory", "Traveling wave"])
    ax.set_xlabel("feedback inh. stre. m")
    ax.set_ylabel("input stre. alpha")
    ax.set_title("Reference oscillatory Fig. 2C phase diagram")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    if show:
        plt.show()
    else:
        plt.close(fig)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Standalone reproduction of ref/oscillatory/fig_2C.m.")
    parser.add_argument("--save-dir", type=Path, default=Path("code/cann_sfa/outputs/fig2c_reference"))
    parser.add_argument("--alpha-min", type=float, default=0.14)
    parser.add_argument("--alpha-max", type=float, default=0.35)
    parser.add_argument("--alpha-points", type=int, default=8)
    parser.add_argument("--m-min", type=float, default=80.0)
    parser.add_argument("--m-max", type=float, default=200.0)
    parser.add_argument("--m-points", type=int, default=8)
    parser.add_argument("--mbar-scale", type=float, default=48.0)
    parser.add_argument("--show", action="store_true")
    return parser


def main():
    args = build_arg_parser().parse_args()
    alpha_values = np.linspace(args.alpha_min, args.alpha_max, args.alpha_points)
    m_values = np.linspace(args.m_min, args.m_max, args.m_points) / args.mbar_scale
    print(
        f"Scanning reference Fig. 2C to {args.save_dir} | "
        f"alpha_points={len(alpha_values)}, m_points={len(m_values)}"
    )
    state, omega = scan_reference_fig2c(alpha_values, m_values)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    plot_reference_fig2c(alpha_values, m_values, state, args.save_dir / "fig2c_reference_phase.png", show=args.show)
    np.savez(
        args.save_dir / "fig2c_reference_phase.npz",
        alpha=alpha_values,
        m=m_values,
        state=state,
        omega=omega,
    )
    print(f"saved reference Fig. 2C outputs to {args.save_dir}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from cann_sfa_triple_regimes import REGIMES, RegimeConfig, simulate_regime

CLASS_TO_ID = {
    "smooth": 0,
    "oscillatory": 1,
    "travelling_wave": 2,
}

ID_TO_CLASS = {value: key for key, value in CLASS_TO_ID.items()}
CLASS_TO_LABEL = {
    "smooth": "Smooth Tracking",
    "oscillatory": "Oscillatory Tracking",
    "travelling_wave": "Traveling Wave",
}
CLASS_COLORS = ["#5AA469", "#F6C445", "#D1495B"]


def compute_eta(config: RegimeConfig) -> float:
    return float(2.0 * np.sqrt(np.pi) * config.a * config.k * config.alpha / config.j0)


def compute_boundary_distance(result: dict) -> float:
    predicted_vint = float(result["predicted_vint"])
    if predicted_vint <= 0.0:
        return float("-inf")
    return float(1.0 - result["config"].v_ext / predicted_vint)


def build_scan_config(base_config: RegimeConfig, mbar: float, vbar: float, alpha: float) -> RegimeConfig:
    return replace(
        base_config,
        m=mbar * base_config.tau / base_config.tau_v,
        v_ext=vbar * base_config.a / base_config.tau_v,
        alpha=alpha,
        noise_sigma=0.0,
    )


def classify_tracking_state(result: dict) -> tuple[str, dict]:
    config = result["config"]
    start_idx = int(round(config.analysis_start / config.dt))
    relative = np.asarray(result["relative"][start_idx:], dtype=np.float64)
    ts = np.asarray(result["ts"][start_idx:], dtype=np.float64)

    if len(relative) < 4 or len(ts) < 4:
        metrics = {
            "relative_amplitude": 0.0,
            "slip_distance": 0.0,
            "slip_speed": 0.0,
            "omega_measured": float(result["measured_omega"]),
        }
        return "smooth", metrics

    centered = relative - relative.mean()
    duration = max(float(ts[-1] - ts[0]), config.dt)
    amplitude = float(np.ptp(centered))
    slip_distance = float(abs(relative[-1] - relative[0]))
    slip_speed = float(slip_distance / duration)
    omega_measured = float(result["measured_omega"])

    if slip_distance > max(4.0 * config.a, 0.8) and slip_speed > 0.005:
        label = "travelling_wave"
    elif amplitude > max(0.5 * config.a, 0.12) and omega_measured > 0.02:
        label = "oscillatory"
    else:
        label = "smooth"

    metrics = {
        "relative_amplitude": amplitude,
        "slip_distance": slip_distance,
        "slip_speed": slip_speed,
        "omega_measured": omega_measured,
    }
    return label, metrics


def scan_phase_diagram(
    base_config: RegimeConfig,
    mbar_values: np.ndarray,
    vbar_values: np.ndarray,
    alpha_values: np.ndarray,
) -> tuple[list[dict], np.ndarray]:
    records: list[dict] = []
    class_grid = np.full(
        (len(alpha_values), len(mbar_values), len(vbar_values)),
        fill_value=-1,
        dtype=np.int8,
    )

    total = len(alpha_values) * len(mbar_values) * len(vbar_values)
    counter = 0
    for alpha_idx, alpha in enumerate(alpha_values):
        for mbar_idx, mbar in enumerate(mbar_values):
            for vbar_idx, vbar in enumerate(vbar_values):
                counter += 1
                print(
                    f"[scan {counter}/{total}] alpha={alpha:.3f}, mbar={mbar:.3f}, vbar={vbar:.3f}",
                    flush=True,
                )
                config = build_scan_config(base_config, mbar=mbar, vbar=vbar, alpha=alpha)
                result = simulate_regime(config)
                state, metrics = classify_tracking_state(result)
                class_grid[alpha_idx, mbar_idx, vbar_idx] = CLASS_TO_ID[state]

                record = {
                    "state": state,
                    "state_id": CLASS_TO_ID[state],
                    "mbar": float(mbar),
                    "vbar": float(vbar),
                    "alpha": float(alpha),
                    "eta": compute_eta(config),
                    "boundary_distance": compute_boundary_distance(result),
                    "predicted_vint": float(result["predicted_vint"]),
                    "predicted_omega": float(result["predicted_omega"]),
                    "measured_omega": float(result["measured_omega"]),
                    "drift_speed": float(result["drift_speed"]),
                    "target_speed": float(result["target_speed_measured"]),
                    "near_width": float(result["near_width"]),
                    "far_width": float(result["far_width"]),
                    "cue_amplitude": float(config.cue_amplitude),
                    "cue_duration": float(config.cue_duration),
                    "base_regime": base_config.name,
                }
                record.update(metrics)
                records.append(record)
    return records, class_grid


def plot_mbar_vbar_slices(
    class_grid: np.ndarray,
    mbar_values: np.ndarray,
    vbar_values: np.ndarray,
    alpha_values: np.ndarray,
    base_config: RegimeConfig,
    save_path: Path,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    ncols = len(alpha_values)
    fig, axes = plt.subplots(1, ncols, figsize=(5.2 * ncols, 4.6), squeeze=False)
    cmap = plt.matplotlib.colors.ListedColormap(CLASS_COLORS)
    norm = plt.matplotlib.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    extent = [vbar_values[0], vbar_values[-1], mbar_values[0], mbar_values[-1]]
    for idx, alpha in enumerate(alpha_values):
        ax = axes[0, idx]
        image = ax.imshow(
            class_grid[idx],
            origin="lower",
            aspect="auto",
            extent=extent,
            interpolation="nearest",
            cmap=cmap,
            norm=norm,
        )
        ax.set_title(
            f"alpha={alpha:.3f}, eta={2.0 * np.sqrt(np.pi) * base_config.a * base_config.k * alpha / base_config.j0:.3f}"
        )
        ax.set_xlabel("vbar = v_ext tau_v / a")
        if idx == 0:
            ax.set_ylabel("mbar = m tau_v / tau")

    cbar = fig.colorbar(image, ax=axes.ravel().tolist(), ticks=[0, 1, 2], shrink=0.95)
    cbar.ax.set_yticklabels(
        [CLASS_TO_LABEL[ID_TO_CLASS[idx]] for idx in [0, 1, 2]]
    )
    fig.suptitle(
        f"No-noise tracking phase diagram | base protocol={base_config.name}, cue={base_config.cue_amplitude:.2f}/{base_config.cue_duration:.1f} ms"
    )
    fig.subplots_adjust(left=0.07, right=0.92, bottom=0.12, top=0.84, wspace=0.25)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_d_eta_projection(records: list[dict], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for state in ["smooth", "oscillatory", "travelling_wave"]:
        xs = [item["boundary_distance"] for item in records if item["state"] == state and np.isfinite(item["boundary_distance"])]
        ys = [item["eta"] for item in records if item["state"] == state and np.isfinite(item["boundary_distance"])]
        ax.scatter(
            xs,
            ys,
            s=36,
            alpha=0.8,
            label=CLASS_TO_LABEL[state],
            color=CLASS_COLORS[CLASS_TO_ID[state]],
            edgecolors="none",
        )
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("D = 1 - v_ext / v_int")
    ax.set_ylabel("eta = 2 sqrt(pi) a k alpha / J0")
    ax.set_title("No-noise classification projected to boundary-distance plane")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def save_records(records: list[dict], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    numeric_keys = [
        "state_id",
        "mbar",
        "vbar",
        "alpha",
        "eta",
        "boundary_distance",
        "predicted_vint",
        "predicted_omega",
        "measured_omega",
        "drift_speed",
        "target_speed",
        "near_width",
        "far_width",
        "cue_amplitude",
        "cue_duration",
        "relative_amplitude",
        "slip_distance",
        "slip_speed",
        "omega_measured",
    ]
    arrays = {key: np.asarray([item[key] for item in records], dtype=np.float64) for key in numeric_keys}
    arrays["state"] = np.asarray([item["state"] for item in records], dtype=object)
    arrays["base_regime"] = np.asarray([item["base_regime"] for item in records], dtype=object)
    np.savez(save_path, **arrays)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="No-noise phase diagram for the fixed CANN1D+SFA tracking model."
    )
    parser.add_argument("--base-regime", choices=tuple(REGIMES.keys()), default="travelling_wave")
    parser.add_argument("--save-dir", type=Path, default=Path("code/cann_sfa/outputs/phase_diagram"))
    parser.add_argument("--mbar-values", type=float, nargs="+", default=[2.5, 10.0, 16.7, 30.0, 45.0])
    parser.add_argument("--vbar-values", type=float, nargs="+", default=[0.15, 0.20, 0.35, 0.67, 1.00])
    parser.add_argument("--alpha-values", type=float, nargs="+", default=[0.03, 0.10, 0.20])
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    base_config = replace(REGIMES[args.base_regime])
    mbar_values = np.asarray(sorted(args.mbar_values), dtype=np.float64)
    vbar_values = np.asarray(sorted(args.vbar_values), dtype=np.float64)
    alpha_values = np.asarray(sorted(args.alpha_values), dtype=np.float64)

    print(
        f"Scanning no-noise phase diagram | base={base_config.name}, "
        f"grid={len(alpha_values)} x {len(mbar_values)} x {len(vbar_values)}"
    )

    records, class_grid = scan_phase_diagram(
        base_config=base_config,
        mbar_values=mbar_values,
        vbar_values=vbar_values,
        alpha_values=alpha_values,
    )

    args.save_dir.mkdir(parents=True, exist_ok=True)
    plot_mbar_vbar_slices(
        class_grid=class_grid,
        mbar_values=mbar_values,
        vbar_values=vbar_values,
        alpha_values=alpha_values,
        base_config=base_config,
        save_path=args.save_dir / f"{base_config.name}_mbar_vbar_phase.png",
    )
    plot_d_eta_projection(
        records=records,
        save_path=args.save_dir / f"{base_config.name}_D_eta_projection.png",
    )
    save_records(records, args.save_dir / f"{base_config.name}_phase_data.npz")
    print(
        f"saved phase diagram to {args.save_dir} | "
        f"points={len(records)}, classes={np.unique(class_grid).tolist()}"
    )


if __name__ == "__main__":
    main()

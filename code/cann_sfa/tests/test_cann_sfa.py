import argparse
from dataclasses import dataclass, replace
from pathlib import Path

import brainpy as bp
import brainpy.math as bm
import brainstate
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


@dataclass(frozen=True)
class TrackingConfig:
    num: int = 512
    tau: float = 1.0
    tau_v: float = 50.0
    k: float = 1.0
    a: float = 0.3
    A: float = 0.2
    J0: float = 1.0
    m: float = 0.05
    dt: float = 0.1
    warmup: float = 100.0
    move_duration: float = 1000.0
    hold_duration: float = 400.0
    v_ext: float = 0.004
    analysis_offset: float = 200.0
    display_start: float = 0.0
    z_min: float = -np.pi
    z_max: float = np.pi

    @property
    def total_duration(self) -> float:
        return self.warmup + self.move_duration + self.hold_duration

    @property
    def mbar(self) -> float:
        return self.m * self.tau_v / self.tau

    @property
    def vbar(self) -> float:
        return self.v_ext * self.tau_v / self.a


TRACKING_PRESETS = {
    "smooth": TrackingConfig(m=0.05),
    "oscillatory": TrackingConfig(m=0.30),
    "travelling_wave": TrackingConfig(
        tau=1.0,
        tau_v=50.0,
        k=1.0,
        a=0.3,
        A=0.2,
        J0=1.0,
        m=0.8,
        dt=0.1,
        warmup=0.0,
        move_duration=2000.0,
        hold_duration=0.0,
        analysis_offset=1000.0,
        display_start=1000.0,
        v_ext=0.004,
    ),
}

REF_TYPE_LABELS = {
    1: "delayed",
    2: "anticipative",
    3: "oscillatory",
    4: "travelling_wave",
}

MACRO_REGIME_LABELS = {
    1: "smooth",
    2: "smooth",
    3: "oscillatory",
    4: "travelling_wave",
}


class CANN1D_SFA(bp.dyn.NeuDyn):
    def __init__(
        self,
        num,
        m=0.1,
        tau=1.0,
        tau_v=10.0,
        k=0.1,
        a=0.5,
        A=10.0,
        J0=4.0,
        z_min=-bm.pi,
        z_max=bm.pi,
        **kwargs,
    ):
        super().__init__(size=num, **kwargs)

        self.tau = tau
        self.tau_v = tau_v
        self.k = k
        self.a = a
        self.A = A
        self.J0 = J0
        self.m = m

        self.z_min = z_min
        self.z_max = z_max
        self.z_range = z_max - z_min
        self.x = bm.arange(num) * self.z_range / num + z_min

        self.u = bm.Variable(bm.zeros(num))
        self.v = bm.Variable(bm.zeros(num))
        self.input = bm.Variable(bm.zeros(num))
        self.center = bm.Variable(0.0)

        self.conn_mat = self.make_conn(self.x)
        self.phase_kernel = bm.exp(1j * self.x)
        self.integral = bp.odeint(self.derivative)

    @property
    def derivative(self):
        def du(u, t, v, Irec, Iext):
            return (-u + Irec + Iext - v) / self.tau

        def dv(v, t, u):
            return (-v + self.m * u) / self.tau_v

        return bp.JointEq([du, dv])

    def dist(self, d):
        d = bm.remainder(d, self.z_range)
        d = bm.where(d > 0.5 * self.z_range, d - self.z_range, d)
        return d

    def make_conn(self, x):
        d = self.dist(x - x[:, None])
        return (
            self.J0
            * bm.exp(-0.5 * bm.square(d / self.a))
            / (bm.sqrt(2 * bm.pi) * self.a)
        )

    def get_stimulus_by_pos(self, pos):
        return self.A * bm.exp(-0.25 * bm.square(self.dist(self.x - pos) / self.a))

    def update(self, x=None):
        u2 = bm.square(self.u)
        r = u2 / (1.0 + self.k * bm.sum(u2))
        Irec = bm.dot(self.conn_mat, r)
        u, v = self.integral(self.u, self.v, bp.share["t"], Irec, self.input)
        self.u[:] = bm.maximum(u, 0.0)
        self.v[:] = v
        self.input[:] = 0.0
        self.center.value = bm.angle(bm.sum(self.u * self.phase_kernel))


def wrap_angle(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


def get_display_slice(config: TrackingConfig):
    start_idx = int(round(config.display_start / config.dt))
    return slice(start_idx, None)


def build_tracking_stimulus(config: TrackingConfig):
    dt = config.dt
    num1 = int(round(config.warmup / dt))
    num2 = int(round(config.move_duration / dt))
    num3 = int(round(config.hold_duration / dt))

    position = np.zeros(num1 + num2 + num3, dtype=float)
    final_pos = config.v_ext * config.move_duration
    position[num1 : num1 + num2] = np.linspace(0.0, final_pos, num2, endpoint=False)
    position[num1 + num2 :] = final_pos
    return position


def simulate_tracking(config: TrackingConfig, monitor_activity: bool = True):
    brainstate.environ.set(dt=config.dt)
    bm.set_dt(config.dt)

    cann = CANN1D_SFA(
        num=config.num,
        tau=config.tau,
        tau_v=config.tau_v,
        k=config.k,
        a=config.a,
        A=config.A,
        J0=config.J0,
        m=config.m,
        z_min=config.z_min,
        z_max=config.z_max,
    )

    stimulus_pos = build_tracking_stimulus(config)
    stimulus_pos_2d = bm.asarray(stimulus_pos).reshape((-1, 1))
    Iext = cann.get_stimulus_by_pos(stimulus_pos_2d)

    monitors = ["center", "v"]
    if monitor_activity:
        monitors.append("u")

    runner = bp.DSRunner(
        cann,
        inputs=("input", Iext, "iter"),
        monitors=monitors,
        progress_bar=not monitor_activity,
    )
    runner.predict(config.total_duration)

    center_wrapped = np.asarray(runner.mon.center)
    ts = np.asarray(runner.mon.ts)
    common_len = min(len(ts), len(center_wrapped), len(stimulus_pos))
    ts = ts[:common_len]
    center_wrapped = center_wrapped[:common_len]
    stimulus_pos = stimulus_pos[:common_len]
    center_unwrapped = np.unwrap(center_wrapped)
    stimulus_wrapped = wrap_angle(stimulus_pos)
    error = wrap_angle(center_wrapped - stimulus_wrapped)

    result = {
        "config": config,
        "ts": ts,
        "x": np.asarray(cann.x),
        "stimulus_pos": stimulus_pos,
        "stimulus_wrapped": stimulus_wrapped,
        "center_wrapped": center_wrapped,
        "center_unwrapped": center_unwrapped,
        "error": error,
        "adaptation": np.asarray(runner.mon.v)[:common_len],
    }
    if monitor_activity:
        result["u"] = np.asarray(runner.mon.u)[:common_len]
        result["input"] = np.asarray(Iext)[:common_len]
    return result


def summarize_tracking(result):
    config = result["config"]
    dt = config.dt
    num1 = int(round(config.warmup / dt))
    num2 = int(round(config.move_duration / dt))
    analysis_start = num1 + int(round(config.analysis_offset / dt))
    track_slice = slice(analysis_start, num1 + num2)
    hold_slice = slice(num1 + num2, None)

    track_error = result["error"][track_slice]
    hold_center = result["center_unwrapped"][hold_slice]

    centered_error = track_error - track_error.mean()
    spectrum = np.abs(np.fft.rfft(centered_error))
    freqs = np.fft.rfftfreq(centered_error.size, d=dt)
    peak_idx = 0 if spectrum.size <= 1 else int(np.argmax(spectrum[1:]) + 1)
    peak_freq = 0.0 if peak_idx == 0 else float(freqs[peak_idx])

    hold_drift = 0.0
    if hold_center.size > 1:
        hold_drift = float((hold_center[-1] - hold_center[0]) / config.hold_duration)

    ref_class = classify_with_ref_criteria(result)

    summary = {
        "mean_lag": float(track_error.mean()),
        "error_std": float(track_error.std()),
        "error_min": float(track_error.min()),
        "error_max": float(track_error.max()),
        "peak_freq": peak_freq,
        "peak_amp": 0.0 if peak_idx == 0 else float(spectrum[peak_idx]),
        "hold_drift": hold_drift,
        "mbar": float(config.mbar),
        "vbar": float(config.vbar),
        "ref_type": ref_class["type_id"],
        "ref_label": ref_class["type_label"],
        "ref_macro_regime": ref_class["macro_regime"],
        "ref_omega_hz": ref_class["omega_hz"],
        "ref_data_range": ref_class["data_range"],
        "ref_mean_offset": ref_class["mean_offset"],
    }

    summary["regime"] = summary["ref_macro_regime"]

    return summary


def classify_with_ref_criteria(result):
    config = result["config"]
    checkpoint = int(round(config.analysis_offset / config.dt))
    relative_position = result["center_unwrapped"] - result["stimulus_pos"]
    data = relative_position[checkpoint:] - relative_position[checkpoint:].mean()
    data_range = float(data.max() - data.min())

    type_id = 0
    omega_hz = 0.0

    if data_range > 1.0:
        type_id = 4
    elif data_range > 0.1:
        type_id = 3
        fL = np.fft.fft(data)
        fL_shift = np.fft.fftshift(fL)
        n = len(data)
        power = np.abs(fL_shift) ** 2 / n
        fs = 1.0 / config.dt * 1e3
        fshift = np.arange(-n / 2, n / 2) * (fs / n)
        omega_hz = float(abs(fshift[int(np.argmax(power))]))
    elif relative_position[checkpoint:].sum() < 0:
        type_id = 1
    else:
        type_id = 2

    return {
        "type_id": type_id,
        "type_label": REF_TYPE_LABELS[type_id],
        "macro_regime": MACRO_REGIME_LABELS[type_id],
        "omega_hz": omega_hz,
        "data_range": data_range,
        "mean_offset": float(relative_position[checkpoint:].mean()),
    }


def plot_tracking(result, summary, title, save_path: Path | None = None, show: bool = False):
    display_slice = get_display_slice(result["config"])
    ts = result["ts"][display_slice]
    config = result["config"]

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(ts, result["stimulus_pos"][display_slice], label="stimulus", linewidth=2)
    axes[0].plot(
        ts,
        result["center_unwrapped"][display_slice],
        label="bump center",
        linewidth=1.5,
    )
    axes[0].set_ylabel("position")
    axes[0].legend(loc="upper left")
    axes[0].set_title(
        f"{title} | m={config.m:.2f} (mbar={config.mbar:.1f}), "
        f"v_ext={config.v_ext:.4f} (vbar={config.vbar:.2f})"
    )

    axes[1].plot(ts, result["error"][display_slice], color="tab:red", linewidth=1.2)
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("wrapped error")
    axes[1].set_title(
        f"classified as {summary['ref_label']} ({summary['regime']}), "
        f"lag={summary['mean_lag']:.4f}, std={summary['error_std']:.4f}, "
        f"hold drift={summary['hold_drift']:.5f}, omega={summary['ref_omega_hz']:.3f} Hz"
    )

    if "u" in result:
        extent = [ts[0], ts[-1], result["x"][0], result["x"][-1]]
        axes[2].imshow(
            result["u"][display_slice].T,
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap="magma",
        )
        axes[2].plot(
            ts,
            result["stimulus_wrapped"][display_slice],
            color="cyan",
            linewidth=1.0,
        )
        axes[2].set_ylabel("network position")
        axes[2].set_title("activity heatmap")
    else:
        axes[2].plot(
            ts,
            result["adaptation"][display_slice].mean(axis=1),
            color="tab:green",
        )
        axes[2].set_ylabel("mean adaptation")

    axes[2].set_xlabel("time (ms)")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


def animate_tracking(
    result,
    summary,
    title,
    save_path: Path | None = None,
    show: bool = False,
    frame_stride: int = 40,
    fps: int = 20,
):
    if "u" not in result or "input" not in result:
        raise ValueError("Animation requires activity and input traces.")

    display_slice = get_display_slice(result["config"])
    ts = result["ts"][display_slice]
    x = result["x"]
    stimulus = result["input"][display_slice]
    activity = result["u"][display_slice]
    center = result["center_unwrapped"][display_slice]
    stim_center = result["stimulus_pos"][display_slice]
    error = result["error"][display_slice]
    config = result["config"]

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)

    max_y = float(max(activity.max(), stimulus.max()) * 1.1)
    frames = list(range(0, len(ts), frame_stride))
    if frames[-1] != len(ts) - 1:
        frames.append(len(ts) - 1)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

    profile_ax = axes[0]
    profile_ax.set_xlim(float(x[0]), float(x[-1]))
    profile_ax.set_ylim(0.0, max_y)
    profile_ax.set_ylabel("activity")
    profile_ax.set_title(
        f"{title} | regime={summary['ref_label']} | m={config.m:.2f}, "
        f"mbar={config.mbar:.1f}, vbar={config.vbar:.2f}"
    )
    stimulus_line, = profile_ax.plot([], [], linewidth=2.0, label="stimulus")
    bump_line, = profile_ax.plot([], [], linewidth=2.0, linestyle="--", label="bump")
    profile_ax.legend(loc="upper right")

    track_ax = axes[1]
    track_ax.set_xlim(float(ts[0]), float(ts[-1]))
    ymin = float(min(stim_center.min(), center.min()))
    ymax = float(max(stim_center.max(), center.max()))
    margin = 0.05 * max(1e-6, ymax - ymin)
    track_ax.set_ylim(ymin - margin, ymax + margin)
    track_ax.set_xlabel("time (ms)")
    track_ax.set_ylabel("position")
    stim_hist_line, = track_ax.plot([], [], linewidth=2.0, label="stimulus center")
    bump_hist_line, = track_ax.plot([], [], linewidth=1.8, label="bump center")
    cursor_line = track_ax.axvline(float(ts[0]), color="black", linestyle=":", linewidth=1.2)
    status_text = track_ax.text(
        0.02,
        0.96,
        "",
        transform=track_ax.transAxes,
        va="top",
        ha="left",
    )
    track_ax.legend(loc="upper left")

    def init():
        stimulus_line.set_data([], [])
        bump_line.set_data([], [])
        stim_hist_line.set_data([], [])
        bump_hist_line.set_data([], [])
        cursor_line.set_xdata([float(ts[0]), float(ts[0])])
        status_text.set_text("")
        return (
            stimulus_line,
            bump_line,
            stim_hist_line,
            bump_hist_line,
            cursor_line,
            status_text,
        )

    def update(frame_id):
        idx = frames[frame_id]
        stimulus_line.set_data(x, stimulus[idx])
        bump_line.set_data(x, activity[idx])
        stim_hist_line.set_data(ts[: idx + 1], stim_center[: idx + 1])
        bump_hist_line.set_data(ts[: idx + 1], center[: idx + 1])
        cursor_line.set_xdata([float(ts[idx]), float(ts[idx])])
        status_text.set_text(
            f"t={ts[idx]:.1f} ms\n"
            f"stim={stim_center[idx]:.3f}\n"
            f"bump={center[idx]:.3f}\n"
            f"err={error[idx]:.3f}"
        )
        return (
            stimulus_line,
            bump_line,
            stim_hist_line,
            bump_hist_line,
            cursor_line,
            status_text,
        )

    animation = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(frames),
        interval=1000 / fps,
        blit=False,
        repeat=True,
    )
    fig.tight_layout()

    if save_path is not None:
        animation.save(save_path, writer=PillowWriter(fps=fps))
    if show:
        plt.show()
    else:
        plt.close(fig)


def run_regime(
    name: str,
    config: TrackingConfig,
    save_dir: Path | None,
    show: bool,
    save_animation: bool,
    animation_stride: int,
    animation_fps: int,
):
    result = simulate_tracking(config, monitor_activity=True)
    summary = summarize_tracking(result)

    print(
        f"[{name}] "
        f"expected={name}, classified={summary['ref_label']} ({summary['regime']}), "
        f"m={config.m:.2f}, mbar={summary['mbar']:.1f}, "
        f"v_ext={config.v_ext:.4f}, vbar={summary['vbar']:.2f}, "
        f"lag={summary['mean_lag']:.5f}, std={summary['error_std']:.5f}, "
        f"hold_drift={summary['hold_drift']:.5f}, omega={summary['ref_omega_hz']:.5f}"
    )

    if save_dir is not None:
        plot_tracking(
            result,
            summary,
            title=name,
            save_path=save_dir / f"{name}.png",
            show=show,
        )
        if save_animation:
            animate_tracking(
                result,
                summary,
                title=name,
                save_path=save_dir / f"{name}.gif",
                show=False,
                frame_stride=animation_stride,
                fps=animation_fps,
            )
    elif show:
        plot_tracking(result, summary, title=name, save_path=None, show=True)
        if save_animation:
            animate_tracking(
                result,
                summary,
                title=name,
                save_path=None,
                show=True,
                frame_stride=animation_stride,
                fps=animation_fps,
            )

    return summary


def scan_m_values(m_values, base_config: TrackingConfig):
    rows = []
    for m in m_values:
        config = replace(base_config, m=float(m))
        result = simulate_tracking(config, monitor_activity=False)
        summary = summarize_tracking(result)
        rows.append(summary | {"m": float(m)})
        print(
            f"m={m:.3f} -> regime={summary['ref_label']} ({summary['regime']}), "
            f"lag={summary['mean_lag']:.5f}, std={summary['error_std']:.5f}, "
            f"hold_drift={summary['hold_drift']:.5f}, omega={summary['ref_omega_hz']:.5f}"
        )
    return rows


def scan_phase_diagram(
    mbar_values,
    vbar_values,
    base_config: TrackingConfig,
):
    type_grid = np.zeros((len(vbar_values), len(mbar_values)), dtype=int)
    omega_grid = np.zeros_like(type_grid, dtype=float)
    lag_grid = np.zeros_like(type_grid, dtype=float)
    all_rows = []

    for i, vbar in enumerate(vbar_values):
        print(f"phase row {i + 1}/{len(vbar_values)}: vbar={vbar:.3f}")
        for j, mbar in enumerate(mbar_values):
            config = replace(
                base_config,
                m=float(mbar * base_config.tau / base_config.tau_v),
                v_ext=float(vbar * base_config.a / base_config.tau_v),
            )
            result = simulate_tracking(config, monitor_activity=False)
            summary = summarize_tracking(result)
            type_grid[i, j] = summary["ref_type"]
            omega_grid[i, j] = summary["ref_omega_hz"]
            lag_grid[i, j] = summary["mean_lag"]
            all_rows.append(
                {
                    "mbar": float(mbar),
                    "vbar": float(vbar),
                    "ref_type": int(summary["ref_type"]),
                    "ref_label": summary["ref_label"],
                    "macro_regime": summary["regime"],
                    "omega_hz": float(summary["ref_omega_hz"]),
                    "data_range": float(summary["ref_data_range"]),
                    "lag": float(summary["mean_lag"]),
                    "error_std": float(summary["error_std"]),
                }
            )
    return {
        "mbar_values": np.asarray(mbar_values, dtype=float),
        "vbar_values": np.asarray(vbar_values, dtype=float),
        "type_grid": type_grid,
        "omega_grid": omega_grid,
        "lag_grid": lag_grid,
        "rows": all_rows,
    }


def select_representative_points(phase_data):
    representatives = {}
    smooth_points = [
        row for row in phase_data["rows"] if row["ref_type"] in (1, 2)
    ]
    if smooth_points:
        mbar_center = float(np.mean([row["mbar"] for row in smooth_points]))
        vbar_center = float(np.mean([row["vbar"] for row in smooth_points]))
        smooth_points.sort(
            key=lambda row: (row["mbar"] - mbar_center) ** 2
            + (row["vbar"] - vbar_center) ** 2
        )
        representatives["smooth"] = smooth_points[0]

    oscillatory_points = [
        row for row in phase_data["rows"] if row["ref_type"] == 3
    ]
    if oscillatory_points:
        oscillatory_points.sort(
            key=lambda row: (row["data_range"], row["omega_hz"], row["error_std"]),
            reverse=True,
        )
        representatives["oscillatory"] = oscillatory_points[0]

    travelling_wave_points = [
        row for row in phase_data["rows"] if row["ref_type"] == 4
    ]
    if travelling_wave_points:
        travelling_wave_points.sort(
            key=lambda row: (row["data_range"], row["error_std"]),
            reverse=True,
        )
        representatives["travelling_wave"] = travelling_wave_points[0]

    return representatives


def plot_phase_diagram(
    phase_data,
    representatives,
    save_path: Path | None = None,
    show: bool = False,
):
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)

    type_grid = phase_data["type_grid"]
    mbar_values = phase_data["mbar_values"]
    vbar_values = phase_data["vbar_values"]

    color_grid = np.zeros_like(type_grid, dtype=float)
    color_grid[type_grid == 1] = 1
    color_grid[type_grid == 2] = 2
    color_grid[type_grid == 3] = 3
    color_grid[type_grid == 4] = 4

    cmap = plt.matplotlib.colors.ListedColormap(
        ["#d7e3fc", "#a9def9", "#ffb703", "#d62828"]
    )
    norm = plt.matplotlib.colors.BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)

    fig, ax = plt.subplots(figsize=(9, 6))
    extent = [
        float(mbar_values[0]),
        float(mbar_values[-1]),
        float(vbar_values[0]),
        float(vbar_values[-1]),
    ]
    image = ax.imshow(
        color_grid,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap=cmap,
        norm=norm,
    )
    cbar = fig.colorbar(image, ax=ax, ticks=[1, 2, 3, 4])
    cbar.ax.set_yticklabels(["delayed", "anticipative", "oscillatory", "travelling wave"])

    for macro_regime, point in representatives.items():
        ax.scatter(
            point["mbar"],
            point["vbar"],
            s=120,
            marker="o",
            facecolors="none",
            edgecolors="black",
            linewidths=2,
        )
        ax.text(
            point["mbar"],
            point["vbar"],
            f" {macro_regime}",
            va="bottom",
            ha="left",
            fontsize=10,
            color="black",
        )

    ax.set_xlabel("mbar")
    ax.set_ylabel("vbar")
    ax.set_title("Phase Diagram from omega_simulate-style criteria")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


def run_phase_diagram(
    base_config: TrackingConfig,
    mbar_values,
    vbar_values,
    save_dir: Path | None,
    show: bool,
    save_animation: bool,
    animation_stride: int,
    animation_fps: int,
):
    if save_dir is None:
        save_dir = Path("code/cann_sfa/outputs/phase_diagram")
    save_dir.mkdir(parents=True, exist_ok=True)

    phase_data = scan_phase_diagram(mbar_values, vbar_values, base_config)
    representatives = select_representative_points(phase_data)
    plot_phase_diagram(
        phase_data,
        representatives,
        save_path=save_dir / "mbar_vbar_phase.png",
        show=show,
    )

    np.savez(
        save_dir / "mbar_vbar_phase.npz",
        mbar_values=phase_data["mbar_values"],
        vbar_values=phase_data["vbar_values"],
        type_grid=phase_data["type_grid"],
        omega_grid=phase_data["omega_grid"],
        lag_grid=phase_data["lag_grid"],
    )

    for macro_regime, point in representatives.items():
        config = replace(
            base_config,
            m=float(point["mbar"] * base_config.tau / base_config.tau_v),
            v_ext=float(point["vbar"] * base_config.a / base_config.tau_v),
        )
        print(
            f"representative {macro_regime}: "
            f"mbar={point['mbar']:.3f}, vbar={point['vbar']:.3f}, "
            f"type={point['ref_label']}"
        )
        run_regime(
            macro_regime,
            config,
            save_dir,
            show=False,
            save_animation=save_animation,
            animation_stride=animation_stride,
            animation_fps=animation_fps,
        )

    return phase_data, representatives


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce smooth tracking, oscillatory tracking, and travelling-wave "
            "tracking in a 1D CANN with spike-frequency adaptation."
        )
    )
    parser.add_argument(
        "--regime",
        choices=tuple(TRACKING_PRESETS.keys()),
        default="smooth",
        help="Run one preset regime.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all three preset regimes sequentially.",
    )
    parser.add_argument(
        "--m",
        type=float,
        default=None,
        help="Override the preset adaptation strength.",
    )
    parser.add_argument(
        "--v-ext",
        type=float,
        default=None,
        help="Override the preset external speed in absolute units.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Optional directory to save the generated figures.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively.",
    )
    parser.add_argument(
        "--no-animation",
        action="store_true",
        help="Disable GIF export.",
    )
    parser.add_argument(
        "--animation-stride",
        type=int,
        default=40,
        help="Take one animation frame every N simulation steps.",
    )
    parser.add_argument(
        "--animation-fps",
        type=int,
        default=20,
        help="Frames per second for GIF export.",
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Scan multiple m values and print the inferred regime table.",
    )
    parser.add_argument(
        "--phase-diagram",
        action="store_true",
        help="Scan a coarse mbar-vbar phase diagram using omega_simulate-style criteria.",
    )
    parser.add_argument(
        "--m-values",
        type=float,
        nargs="*",
        default=[0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.80],
        help="m values used by --scan.",
    )
    parser.add_argument(
        "--mbar-min",
        type=float,
        default=0.0,
        help="Minimum mbar value for --phase-diagram.",
    )
    parser.add_argument(
        "--mbar-max",
        type=float,
        default=50.0,
        help="Maximum mbar value for --phase-diagram.",
    )
    parser.add_argument(
        "--mbar-points",
        type=int,
        default=10,
        help="Number of mbar samples for --phase-diagram.",
    )
    parser.add_argument(
        "--vbar-min",
        type=float,
        default=0.2,
        help="Minimum vbar value for --phase-diagram.",
    )
    parser.add_argument(
        "--vbar-max",
        type=float,
        default=1.4,
        help="Maximum vbar value for --phase-diagram.",
    )
    parser.add_argument(
        "--vbar-points",
        type=int,
        default=8,
        help="Number of vbar samples for --phase-diagram.",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    base_config = TRACKING_PRESETS[args.regime]
    if args.m is not None:
        base_config = replace(base_config, m=args.m)
    if args.v_ext is not None:
        base_config = replace(base_config, v_ext=args.v_ext)

    print("Tracking presets:")
    for name, config in TRACKING_PRESETS.items():
        print(
            f"  {name}: m={config.m:.2f}, mbar={config.mbar:.1f}, "
            f"v_ext={config.v_ext:.4f}, vbar={config.vbar:.2f}"
        )

    if args.scan:
        scan_m_values(args.m_values, base_config)
        return

    if args.phase_diagram:
        mbar_values = np.linspace(args.mbar_min, args.mbar_max, args.mbar_points)
        vbar_values = np.linspace(args.vbar_min, args.vbar_max, args.vbar_points)
        run_phase_diagram(
            base_config,
            mbar_values=mbar_values,
            vbar_values=vbar_values,
            save_dir=args.save_dir,
            show=args.show,
            save_animation=not args.no_animation,
            animation_stride=args.animation_stride,
            animation_fps=args.animation_fps,
        )
        return

    if args.all:
        for name, config in TRACKING_PRESETS.items():
            run_regime(
                name,
                config,
                args.save_dir,
                args.show,
                save_animation=not args.no_animation,
                animation_stride=args.animation_stride,
                animation_fps=args.animation_fps,
            )
        return

    run_regime(
        args.regime,
        base_config,
        args.save_dir,
        args.show,
        save_animation=not args.no_animation,
        animation_stride=args.animation_stride,
        animation_fps=args.animation_fps,
    )


if __name__ == "__main__":
    main()

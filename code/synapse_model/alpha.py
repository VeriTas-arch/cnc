import matplotlib.pyplot as plt
import numpy as np

from synapse_utils import apply_brainpy_delay, create_post_hh


class Alpha:
    def __init__(
        self,
        type,
        g_max=5.0,
        tau=5.0,
        delay_step=2,
        E=0.0,
        V_rest=-65.0,
    ):
        self.g_max = g_max
        self.tau = tau
        self.E = E
        self.V_rest = V_rest
        self.delay_step = delay_step
        self.type = type  # CUBA / COBA

        # Use a dense connection matrix to avoid numba-dependent event operators.
        self.g = 0.0
        self.h = 0.0

    def update(self, pre_spike, post_V, dt):
        post_sp = float(pre_spike) * self.g_max

        decay = np.exp(-dt / self.tau)
        self.h *= decay
        self.h += post_sp
        self.g = self.g * decay + self.h * self.tau * (1.0 - decay)

        if self.type == "CUBA":
            current = self.g * (self.E - self.V_rest)
        elif self.type == "COBA":
            current = self.g * (self.E - post_V)
        else:
            raise ValueError("type should be 'CUBA' or 'COBA'")
        return self.g, current


def make_spike_train(sp_times, run_duration, dt):
    ts = np.arange(0.0, run_duration, dt)
    spikes = np.zeros_like(ts)
    for t in sp_times:
        idx = int(round(t / dt))
        if 0 <= idx < len(spikes):
            spikes[idx] = 1.0
    return ts, spikes


def run_syn(
    syn_model, type, title, run_duration=200.0, sp_times=(25, 50, 75, 100, 150), dt=0.1
):
    ts, pre_spike = make_spike_train(sp_times, run_duration, dt)
    syn = syn_model(type=type)
    delayed_spike = apply_brainpy_delay(pre_spike, syn.delay_step)

    post = create_post_hh(-50.68)
    post_V = np.zeros_like(ts)
    g = np.zeros_like(ts)
    post_input = np.zeros_like(ts)
    for i, t in enumerate(ts):
        g[i], post_input[i] = syn.update(delayed_spike[i], post.V[0], dt)
        post.update(post_input[i], t, dt)
        post_V[i] = post.V[0]

    fig, axes = plt.subplots(4, 1, figsize=(6.0, 3.5), sharex=True)

    ax = axes[0]
    ax.plot(ts, pre_spike, label="pre.spike")
    ax.legend(loc="upper right")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    ax.plot(ts, g, label="g", color="#d62728")
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[2]
    ax.plot(ts, post_input, label="PSC", color="#d62728")
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[3]
    ax.plot(ts, post_V, label="post.V")
    ax.legend(loc="upper right")
    ax.set_xlabel("Time [ms]")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_syn(
        Alpha,
        type="CUBA",
        sp_times=[25, 50, 75, 100, 160],
        title="Alpha Synapse Model (Current-Based)",
    )
    run_syn(
        Alpha,
        type="COBA",
        sp_times=[25, 50, 75, 100, 160],
        title="Alpha Synapse Model (Conductance-Based)",
    )

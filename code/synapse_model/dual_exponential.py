import matplotlib.pyplot as plt
import numpy as np

from synapse_utils import apply_brainpy_delay, create_post_hh


class DualExponential:
    def __init__(
        self,
        type,
        g_max=5.0,
        tau_decay=20.0,
        tau_rise=2.0,
        delay_step=2,
        E=0.0,
        V_rest=-65.0,
    ):
        self.tau_decay = tau_decay
        self.tau_rise = tau_rise
        self.g_max = g_max
        self.delay_step = delay_step
        self.E = E
        self.V_rest = V_rest
        self.type = type  # CUBA / COBA

        # 使用连接矩阵聚合事件，避免 CPU 下 pre2post 事件算子对 numba 的依赖
        self.g = 0.0
        self.h = 0.0

    def update(self, pre_spike, post_V, dt):
        # 根据连接矩阵计算各个突触后神经元收到的信号强度
        post_sp = float(pre_spike) * self.g_max

        # g 和 h 的更新包括常规积分和突触前脉冲带来的跃变
        rise_decay = np.exp(-dt / self.tau_rise)
        decay_decay = np.exp(-dt / self.tau_decay)
        self.h *= rise_decay
        self.h += post_sp
        self.g = self.g * decay_decay + self.h * self.tau_decay * (1.0 - decay_decay)

        # 根据不同模式计算突触后电流
        if self.type == "CUBA":
            current = self.g * (self.E - self.V_rest)  # E - V_rest
        elif self.type == "COBA":
            current = self.g * (self.E - post_V)  # E - V_post
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
    # 定义突触前神经元、突触后神经元和突触连接，并构建神经网络
    ts, pre_spike = make_spike_train(sp_times, run_duration, dt)
    syn = syn_model(type=type)
    delayed_spike = apply_brainpy_delay(pre_spike, syn.delay_step)

    post = create_post_hh(-70.68)
    post_V = np.zeros_like(ts)
    g = np.zeros_like(ts)
    post_input = np.zeros_like(ts)
    for i, t in enumerate(ts):
        g[i], post_input[i] = syn.update(delayed_spike[i], post.V[0], dt)
        post.update(post_input[i], t, dt)
        post_V[i] = post.V[0]

    # 可视化
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
    ax.set_xlabel("t (ms)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_syn(
        DualExponential, type="CUBA", title="DualExponential Synapse Model (Current-Based)"
    )
    run_syn(
        DualExponential,
        type="COBA",
        title="DualExponential Synapse Model (Conductance-Based)",
    )

import matplotlib.pyplot as plt
import numpy as np
from synapse_utils import apply_brainpy_delay, create_post_hh


class Exponential:
    def __init__(self, type, g_max=0.02, tau=12.0, delay_step=2, E=0.0, V_rest=-65.0):
        self.tau = tau
        self.g_max = g_max
        self.delay_step = delay_step
        self.E = E
        self.V_rest = V_rest
        self.type = type  # CUBA / COBA
        self.g = 0.0

    def update(self, pre_spike, post_V, dt):
        # 根据连接矩阵计算各个突触后神经元收到的信号强度
        post_sp = float(pre_spike) * self.g_max
        # 突触的电导 g 的更新包括常规积分和突触前脉冲带来的跃变
        self.g *= np.exp(-dt / self.tau)
        self.g += post_sp
        # 计算突触后电流
        if self.type == "CUBA":
            current = self.g * (self.E - self.V_rest)  # E - V_rest
        elif self.type == "COBA":
            current = self.g * (self.E - post_V)  # E - V_post
        else:
            raise ValueError("type should be 'CUBA' or 'COBA'")
        return self.g, current

        """
        注：我们在此使用的 CUBA 模式其实并非是原始意义上的 current-based，因为计算时仍然
        引入了电压项 (E - V_rest)，但我们仍然将其称为 CUBA 模式以与 COBA 模式进行区分。

        可以参考 <https://brainpy.readthedocs.io/apis/brainpy.dyn.outs.html> 中
        关于 CUBA 和 COBA 的说明。
        """


def make_spike_train(sp_times, run_duration, dt):
    ts = np.arange(0.0, run_duration, dt)
    spikes = np.zeros_like(ts)
    for t in sp_times:
        idx = int(round(t / dt))
        if 0 <= idx < len(spikes):
            spikes[idx] = 1.0
    return ts, spikes


def run_syn(syn_model, type, title, run_duration=200.0, sp_times=(10, 20, 30), dt=0.1):
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

    fig, axes = plt.subplots(4, 1, figsize=(6.0, 3.5), sharex=True)
    axes[0].plot(ts, pre_spike, label="pre.spike")
    axes[0].legend(loc="upper right")
    axes[0].set_title(title)
    axes[1].plot(ts, g, label="g", color="#d62728")
    axes[1].legend(loc="upper right")
    axes[2].plot(ts, post_input, label="PSC", color="#d62728")
    axes[2].legend(loc="upper right")
    axes[3].plot(ts, post_V, label="post.V")
    axes[3].legend(loc="upper right")
    axes[3].set_xlabel("t (ms)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_syn(
        Exponential,
        type="CUBA",
        sp_times=[25, 50, 75, 100, 160],
        title="Exponential Synapse Model (Current-Based)",
    )
    run_syn(
        Exponential,
        type="COBA",
        sp_times=[25, 50, 75, 100, 160],
        title="Exponential Synapse Model (Conductance-Based)",
    )

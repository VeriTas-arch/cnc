import matplotlib.pyplot as plt
import numpy as np
from synapse_utils import apply_brainpy_delay, create_post_hh


class Delta:
    def __init__(self, g_max=1.0, delay_step=0, E=0.0):
        self.g_max = g_max
        self.delay_step = delay_step  # 控制突触前神经元产生的 spike delay 时间
        self.E = E
        self.g = 0.0

    def update(self, pre_spike):
        # 根据连接矩阵计算各个突触后神经元收到的信号强度
        self.g = float(pre_spike) * self.g_max
        return self.g


def make_spike_train(sp_times, run_duration, dt):
    ts = np.arange(0.0, run_duration, dt)
    spikes = np.zeros_like(ts)
    for t in sp_times:
        idx = int(round(t / dt))
        if 0 <= idx < len(spikes):
            spikes[idx] = 1.0
    return ts, spikes


def simulate_post_voltage(currents, ts, dt):
    # 突触后神经元
    post = create_post_hh(-70.68)
    V = np.zeros_like(currents)
    for i, t in enumerate(ts):
        post.update(currents[i], t, dt)
        V[i] = post.V[0]
    return V


def run_syn(run_duration=200.0, dt=0.1):
    """
    假如要让 5 个突触前神经元在 20ms 时刻同时产生一个 spike，可以定义如下：

    neu1 = SpikeTimeGroup(
        5,  # 神经元数量
        times=[20, 20, 20, 20, 20],  # 每个 spike 的时间点
        indices=[0, 1, 2, 3, 4],  # 每个 spike 对应的神经元索引，0-4 分别对应 5 个神经元
    )

    另一个例子：

    neu1 = SpikeTimeGroup(
        2, times=[20, 30, 60, 70, 100, 100, 140, 180], indices=[0, 1, 0, 1, 0, 0, 1, 0]
    )
    """
    # 突触前神经元，第一个参数为神经元数量
    ts, pre_spike = make_spike_train([20, 60, 100, 140, 180], run_duration, dt)
    syn = Delta(g_max=2.0)  # All2All 意味着每个突触前神经元都连接到每个突触后神经元

    delayed_spike = apply_brainpy_delay(pre_spike, syn.delay_step)

    g = np.array([syn.update(sp) for sp in delayed_spike])
    post_V = simulate_post_voltage(g, ts, dt)

    # 可视化
    fig, axes = plt.subplots(3, 1, figsize=(6.0, 4.5), sharex=True)
    axes[0].plot(ts, pre_spike, label="pre.spike")
    axes[0].legend(loc="upper right")
    axes[0].set_title("Delta Synapse Model (Current-Based)")
    axes[1].plot(ts, g, label="g", color="#d62728")
    axes[1].legend(loc="upper right")
    axes[2].plot(ts, post_V, label="post.V")
    axes[2].legend(loc="upper right")
    axes[2].set_xlabel("t (ms)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_syn()

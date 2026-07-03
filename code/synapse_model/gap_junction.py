import matplotlib.pyplot as plt
import numpy as np
from synapse_utils import create_post_hh


class GapJunction:
    def __init__(self, g=0.2):
        self.g = g
        self.current = np.zeros(2)

        # 获取每个连接的突触前神经元 pre_ids 和突触后神经元 post_ids
        self.pre_ids = np.array([0, 1])
        self.post_ids = np.array([1, 0])

    def update(self, V):
        # 计算突触后电流，从外向内为正方向
        # 计算方式：电导 g 乘以突触前神经元电位与突触后神经元电位之差（pre - post）
        inputs = self.g * (V[self.pre_ids] - V[self.post_ids])

        # 从 synapse 到 post 的计算：post id 相同的电流加和到一起
        self.current[:] = 0.0
        np.add.at(self.current, self.post_ids, inputs)
        return self.current.copy()


def run_syn(syn_model, title, run_duration=100.0, Iext=7.5, dt=0.01):
    # 定义神经元组和突触连接，并构建神经网络
    ts = np.arange(0.0, run_duration, dt)
    neu = create_post_hh(-70.68, size=2)
    V = np.zeros((len(ts), 2))
    current = np.zeros_like(V)
    syn = syn_model()  # include_self=False: 自己和自己没有连接

    # 运行模拟
    external = np.array([Iext, 0.0])
    for i, t in enumerate(ts):
        current[i] = syn.update(neu.V)
        total_input = external + current[i]
        neu.update(total_input, t, dt)
        V[i] = neu.V

    # 可视化
    fig, gs = plt.subplots(2, 1, figsize=(6, 4.5), sharex=True)
    plt.sca(gs[0])
    plt.plot(ts, V[:, 0], label="neu0-V")
    plt.plot(ts, V[:, 1], label="neu1-V", linestyle="--")
    plt.legend(loc="upper right")
    plt.title(title)

    plt.sca(gs[1])
    plt.plot(ts, current[:, 0], label="neu0-current", color="#48d688")
    plt.plot(ts, current[:, 1], label="neu1-current", color="#d64888", linestyle="--")
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_syn(GapJunction, Iext=7.5, title="Gap Junction Model")
    run_syn(GapJunction, Iext=5.0, title="Gap Junction Model")

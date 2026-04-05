import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt


class GapJunction(bp.dyn.SynConn):
    def __init__(self, pre, post, conn, g=0.2):
        super().__init__(pre=pre, post=post, conn=conn)
        self.g = g
        self.current = bm.Variable(bm.zeros(self.post.num))

        # 获取每个连接的突触前神经元 pre_ids 和突触后神经元 post_ids
        self.pre_ids, self.post_ids = self.conn.require("pre_ids", "post_ids")

    def update(self):
        # 计算突触后电流，从外向内为正方向
        # 计算方式：电导 g 乘以突触前神经元电位与突触后神经元电位之差（pre - post）
        inputs = self.g * (self.pre.V[self.pre_ids] - self.post.V[self.post_ids])

        # 从 synapse 到 post 的计算：post id 相同的电流加和到一起
        self.current.value = bm.syn2post_sum(inputs, self.post_ids, self.post.num)
        self.post.input += self.current


def run_syn(syn_model, title, run_duration=100.0, Iext=7.5):
    # 定义神经元组和突触连接，并构建神经网络
    neu = bp.neurons.HH(2, V_initializer=bp.initialize.Constant(-70.68))
    syn = syn_model(
        neu, neu, conn=bp.connect.All2All(include_self=False)
    )  # include_self=False: 自己和自己没有连接
    net = bp.DynSysGroup(syn=syn, neu=neu)

    # 运行模拟
    runner = bp.DSRunner(
        net,
        inputs=[("neu.input", bm.array([Iext, 0.0]))],
        monitors=["neu.V", "syn.current"],
    )
    runner.predict(run_duration)

    # 可视化
    fig, gs = plt.subplots(2, 1, figsize=(6, 4.5))
    plt.sca(gs[0])
    plt.plot(runner.mon.ts, runner.mon["neu.V"][:, 0], label="neu0-V")
    plt.plot(runner.mon.ts, runner.mon["neu.V"][:, 1], label="neu1-V", linestyle="--")
    plt.legend(loc="upper right")
    plt.title(title)

    plt.sca(gs[1])
    plt.plot(
        runner.mon.ts,
        runner.mon["syn.current"][:, 0],
        label="neu0-current",
        color="#48d688",
    )
    plt.plot(
        runner.mon.ts,
        runner.mon["syn.current"][:, 1],
        label="neu1-current",
        color="#d64888",
        linestyle="--",
    )
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


run_syn(GapJunction, Iext=7.5, title="Gap Junction Model")
run_syn(GapJunction, Iext=5.0, title="Gap Junction Model")

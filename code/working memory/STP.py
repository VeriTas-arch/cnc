import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt


class STP(bp.synapses.TwoEndConn):
    def __init__(
        self,
        pre,
        post,
        conn,
        g_max=0.1,
        U=0.15,
        tau_f=1500.0,
        tau_d=200.0,
        tau=8.0,
        E=1.0,
        delay_step=2,
        **kwargs
    ):
        super().__init__(pre=pre, post=post, conn=conn, **kwargs)
        # 初始化参数
        self.tau_d = tau_d
        self.tau_f = tau_f
        self.tau = tau
        self.U = U
        self.g_max = g_max
        self.E = E
        self.delay_step = delay_step
        # 获取每个连接的突触前神经元 pre_ids 和突触后神经元 post_ids
        self.pre_ids, self.post_ids = self.conn.require("pre_ids", "post_ids")

        # 初始化变量
        num = len(self.pre_ids)
        self.x = bm.Variable(bm.ones(num))
        self.u = bm.Variable(bm.zeros(num))
        self.g = bm.Variable(bm.zeros(num))

        self.delay = bm.LengthDelay(self.g, delay_step)  # 定义一个处理 g 的延迟器
        self.integral = bp.odeint(method="exp_auto", f=self.derivative)

    @property
    def derivative(self):
        def du(u, t):
            return -u / self.tau_f

        def dx(x, t):
            return (1 - x) / self.tau_d

        def dg(g, t):
            return -g / self.tau

        return bp.JointEq(du, dx, dg)

    def update(self):
        # 将 g 的计算延迟 delay_step 的时间步长
        delayed_g = self.delay(self.delay_step)

        # 计算突触后电流
        post_g = bm.syn2post_sum(delayed_g, self.post_ids, self.post.num)
        self.post.input += post_g * (self.E - self.post.V_rest)

        # 更新各个变量
        syn_sps = bm.pre2syn(self.pre.spike, self.pre_ids)
        t = bp.share["t"]
        dt = bp.share["dt"]
        u, x, g = self.integral(self.u, self.x, self.g, t, dt)
        u = bm.where(syn_sps, u + self.U * (1 - self.u), u)
        x = bm.where(syn_sps, x - u * self.x, x)
        g = bm.where(syn_sps, g + self.g_max * u * self.x, g)
        self.u.value = u
        self.x.value = x
        self.g.value = g

        # 更新延迟器
        self.delay.update(self.g)


def run_STP(title=None, **kwargs):
    # 定义突触前神经元、突触后神经元和突触连接，并构建神经网络
    neu1 = bp.neurons.LIF(1)
    neu2 = bp.neurons.LIF(1)
    syn = STP(neu1, neu2, bp.connect.All2All(), **kwargs)
    net = bp.Network(pre=neu1, syn=syn, post=neu2)

    # 分段电流
    inputs, dur = bp.inputs.section_input(
        values=[22.0, 0.0, 22.0, 0.0],
        durations=[200.0, 200.0, 25.0, 75.0],
        return_length=True,
    )
    # 运行模拟
    runner = bp.DSRunner(
        net,
        inputs=[("pre.input", inputs, "iter")],
        monitors=["syn.u", "syn.x", "syn.g"],
    )
    runner.predict(dur)

    # 可视化
    fig, gs = plt.subplots(2, 1, figsize=(6, 4.5))

    plt.sca(gs[0])
    plt.plot(runner.mon.ts, runner.mon["syn.x"][:, 0], label="x")
    plt.plot(runner.mon.ts, runner.mon["syn.u"][:, 0], label="u")
    plt.legend(loc="center right")
    if title:
        plt.title(title)

    plt.sca(gs[1])
    plt.plot(runner.mon.ts, runner.mon["syn.g"][:, 0], label="g", color="#d62728")
    plt.legend(loc="center right")

    plt.xlabel("t (ms)")
    plt.tight_layout()
    plt.show()


run_STP(title="STF", U=0.1, tau_d=15, tau_f=200)
run_STP(title="STD", U=0.4, tau_d=200, tau_f=15)

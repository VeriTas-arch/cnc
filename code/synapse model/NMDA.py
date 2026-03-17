import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt


class NMDA(bp.dyn.SynConn):
    def __init__(
        self,
        pre,
        post,
        conn,
        g_max=0.02,
        E=0.0,
        c_Mg=1.2,
        alpha1=2.0,
        beta1=0.01,
        alpha2=0.2,
        beta2=0.5,
        delay_step=2,
        T=1.0,
        T_duration=1.0,
    ):
        super().__init__(pre=pre, post=post, conn=conn)
        self.g_max = g_max
        self.E = E
        self.c_Mg = c_Mg
        self.alpha1 = alpha1
        self.beta1 = beta1
        self.alpha2 = alpha2
        self.beta2 = beta2
        self.T = T
        self.T_duration = T_duration
        self.delay_step = delay_step

        self.pre2post = self.conn.require("pre2post")

        self.x = bm.Variable(bm.zeros(self.post.num))
        self.s = bm.Variable(bm.zeros(self.post.num))
        self.g = bm.Variable(bm.zeros(self.post.num))
        self.b = bm.Variable(bm.zeros(self.post.num))
        self.delay = bm.LengthDelay(self.pre.spike, delay_step)

        # 初始化为一个非常大的负数，使得初始时刻 [T] 的值为 0
        # 可以思考，如果我们将其初始化为 0，会对模拟结果产生什么样的影响？
        self.spike_arrival_time = bm.Variable(bm.ones(self.post.num) * -1e7)

        self.integral = bp.odeint(f=bp.JointEq(self.ds, self.dx), method="exp_auto")

    def ds(self, s, t, x):
        return self.alpha1 * x * (1 - s) - self.beta1 * s

    def dx(self, x, t, T):
        return self.alpha2 * T * (1 - x) - self.beta2 * x

    def update(self):
        ## step 1：更新突触前神经元产生 spike
        t = bp.share["t"]
        dt = bp.share["dt"]

        delayed_pre_spike = self.delay(
            self.delay_step
        )  # 取出延迟了 delay_step 时间步长的突触前脉冲信号
        self.delay.update(self.pre.spike)

        ## step 2：计算 [T] 的影响
        self.spike_arrival_time.value = bm.where(
            delayed_pre_spike, t, self.spike_arrival_time
        )  # 计算神经递质到达突触后膜的时间
        T = (
            (t - self.spike_arrival_time) < self.T_duration
        ) * self.T  # 计算突触后膜附近神经递质的浓度

        ## step 3：更新 s，g，b 的数值
        self.s.value, self.x.value = self.integral(self.s, self.x, t, T, dt)
        self.g.value = self.g_max * self.s
        self.b.value = 1 / (1 + bm.exp(-0.062 * self.post.V) * self.c_Mg / 3.57)

        ## step 4：计算突触后细胞的输入电流（conductance-based）
        self.post.input += self.g * self.b * (self.E - self.post.V)


def run_syn(syn_model, title, run_duration=200.0, sp_times=(25, 50, 75, 100, 160)):
    # 定义突触前神经元、突触后神经元和突触连接，并构建神经网络
    neu1 = bp.neurons.SpikeTimeGroup(1, times=sp_times, indices=[0] * len(sp_times))
    neu2 = bp.neurons.HH(1, V_initializer=bp.initialize.Constant(-70.68))
    syn1 = syn_model(neu1, neu2, conn=bp.connect.All2All())
    net = bp.DynSysGroup(pre=neu1, syn=syn1, post=neu2)
    currents, _ = bp.inputs.section_input(
        values=[0.0, 8, 0.0], durations=[130, 1, 69], return_length=True
    )

    # 运行模拟
    runner = bp.DSRunner(
        net,
        monitors=["pre.spike", "post.V", "syn.b", "syn.g", "post.input"],
        inputs=["post.input", currents, "iter"],
    )
    runner.predict(run_duration)

    # 可视化
    fig, gs = bp.visualize.get_figure(9, 1, 0.5, 6.0)
    ax = fig.add_subplot(gs[0, 0])
    plt.plot(runner.mon.ts, runner.mon["pre.spike"], label="pre.spike")
    plt.legend(loc="upper right")
    plt.title(title)
    plt.xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = fig.add_subplot(gs[3:5, 0])
    plt.plot(runner.mon.ts, runner.mon["syn.g"], label="g", color="#d62728")
    plt.legend(loc="upper right")
    plt.xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = fig.add_subplot(gs[7:9, 0])
    plt.plot(runner.mon.ts, runner.mon["post.input"], label="PSC", color="y")
    plt.legend(loc="upper right")
    plt.xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = fig.add_subplot(gs[1:3, 0])
    plt.plot(runner.mon.ts, runner.mon["post.V"], label="post.V")
    plt.legend(loc="upper right")
    plt.xlabel("Time [ms]")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax = fig.add_subplot(gs[5:7, 0])
    plt.plot(runner.mon.ts, runner.mon["syn.b"], label="b", color="blue")
    plt.legend(loc="upper right")
    plt.xlabel("Time [ms]")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.show()


run_syn(NMDA, title="NMDA Synapse Model")

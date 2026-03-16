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
        method="exp_auto",
    ):
        super(NMDA, self).__init__(pre=pre, post=post, conn=conn)
        # 初始化参数
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
        # 获取关于连接的信息
        self.pre2post = self.conn.require("pre2post")
        # 初始化变量
        self.x = bm.Variable(bm.zeros(self.post.num))
        self.s = bm.Variable(bm.zeros(self.post.num))
        self.g = bm.Variable(bm.zeros(self.post.num))
        self.b = bm.Variable(bm.zeros(self.post.num))
        self.delay = bm.LengthDelay(self.pre.spike, delay_step)  # 定义一个延迟处理器
        self.spike_arrival_time = bm.Variable(bm.zeros(self.post.num))
        # 定义积分函数
        self.integral = bp.odeint(method=method, f=bp.JointEq(self.ds, self.dx))

    def ds(self, s, t, x):
        return self.alpha1 * x * (1 - s) - self.beta1 * s

    def dx(self, x, t, T):
        return self.alpha2 * T * (1 - x) - self.beta2 * x

    def update(self):
        t = bp.share["t"]
        dt = bp.share["dt"]
        # 取出延迟了delay_step时间步长的突触前脉冲信号
        delayed_pre_spike = self.delay(self.delay_step)
        self.delay.update(self.pre.spike)
        # 计算神经递质到达突触后膜的时间
        self.spike_arrival_time.value = bm.where(
            delayed_pre_spike, t, self.spike_arrival_time
        )
        # 计算突触后膜附近神经递质的浓度
        T = ((t - self.spike_arrival_time) < self.T_duration) * self.T
        # 更新x，s和g
        self.s.value, self.x.value = self.integral(self.s, self.x, t, T, dt)
        self.g.value = self.g_max * self.s
        # 更新b
        self.b.value = 1 / (1 + bm.exp(-0.062 * self.post.V) * self.c_Mg / 3.57)
        # 电导模式下计算突触后电流大小
        self.post.input += self.g * self.b * (self.E - self.post.V)


def run_syn(
    syn_model, title, run_duration=200.0, sp_times=(25, 50, 75, 100, 150), **kwargs
):
    # 定义突触前神经元、突触后神经元和突触连接，并构建神经网络
    neu1 = bp.neurons.SpikeTimeGroup(1, times=sp_times, indices=[0] * len(sp_times))
    neu2 = bp.neurons.HH(1, V_initializer=bp.init.Constant(-70.68))
    syn1 = syn_model(neu1, neu2, conn=bp.connect.All2All(), **kwargs)
    net = bp.Network(pre=neu1, syn=syn1, post=neu2)
    currents, length = bp.inputs.section_input(
        values=[0.0, 10, 0.0], durations=[130, 2, 65], return_length=True
    )
    # 运行模拟
    runner = bp.DSRunner(
        net,
        monitors=["pre.spike", "post.V", "syn.b", "syn.g", "post.input"],
        inputs=["post.input", currents, "iter"],
    )
    runner.run(run_duration)
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


run_syn(NMDA, title="Delta Synapse Model (Current-Based)")

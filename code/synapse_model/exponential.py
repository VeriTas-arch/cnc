import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt


class Exponential(bp.dyn.SynConn):
    def __init__(
        self,
        pre,
        post,
        conn,
        type,
        g_max=0.02,
        tau=12.0,
        delay_step=2,
        E=0.0,
        V_rest=-65.0,
    ):
        super().__init__(pre=pre, post=post, conn=conn)
        self.tau = tau
        self.g_max = g_max
        self.delay_step = delay_step
        self.E = E
        self.V_rest = V_rest
        self.type = type  # CUBA / COBA

        self.g = bm.Variable(bm.zeros(self.post.num))
        self.delay = bm.LengthDelay(self.pre.spike, delay_step)

        # 使用连接矩阵聚合事件，避免 CPU 下 pre2post 事件算子对 numba 的依赖
        self.conn_mat = bm.asarray(self.conn.require("conn_mat"), dtype=bm.float_)

        self.integral = bp.odeint(f=lambda g, t: -g / self.tau, method="exp_auto")

    def update(self):
        t = bp.share["t"]
        dt = bp.share["dt"]
        # 取出延迟了 delay_step 时间步长的突触前脉冲信号
        delayed_pre_spike = self.delay(self.delay_step)
        self.delay.update(self.pre.spike)
        # 根据连接矩阵计算各个突触后神经元收到的信号强度
        pre_sp = bm.asarray(delayed_pre_spike, dtype=bm.float_)
        post_sp = bm.matmul(pre_sp, self.conn_mat) * self.g_max
        # 突触的电导 g 的更新包括常规积分和突触前脉冲带来的跃变
        self.g.value = self.integral(self.g, t, dt) + post_sp
        # 计算突触后电流
        if self.type == "CUBA":
            self.post.input += self.g * (self.E - self.V_rest)  # E - V_rest
        elif self.type == "COBA":
            self.post.input += self.g * (self.E - self.post.V)  # E - V_post

        """
        注：我们在此使用的 CUBA 模式其实并非是原始意义上的 current-based，因为计算时仍然
        引入了电压项 (E - V_rest)，但我们仍然将其称为 CUBA 模式以与 COBA 模式进行区分。

        可以参考 <https://brainpy.readthedocs.io/apis/brainpy.dyn.outs.html> 中
        关于 CUBA 和 COBA 的说明。
        """


def run_syn(syn_model, type, title, run_duration=200.0, sp_times=(10, 20, 30)):
    # 定义突触前神经元、突触后神经元和突触连接，并构建神经网络
    neu1 = bp.neurons.SpikeTimeGroup(1, times=sp_times, indices=[0] * len(sp_times))
    neu2 = bp.neurons.HH(1, V_initializer=bp.initialize.Constant(-70.68))
    syn1 = syn_model(neu1, neu2, conn=bp.connect.All2All(), type=type)
    net = bp.DynSysGroup(pre=neu1, syn=syn1, post=neu2)

    runner = bp.DSRunner(net, monitors=["pre.spike", "post.V", "syn.g", "post.input"])
    runner.predict(run_duration)

    fig, gs = bp.visualize.get_figure(7, 1, 0.5, 6.0)
    fig.add_subplot(gs[0, 0])
    plt.plot(runner.mon.ts, runner.mon["pre.spike"], label="pre.spike")
    plt.legend(loc="upper right")
    plt.title(title)
    plt.xticks([])
    fig.add_subplot(gs[1:3, 0])
    plt.plot(runner.mon.ts, runner.mon["syn.g"], label="g", color="#d62728")
    plt.legend(loc="upper right")
    plt.xticks([])
    fig.add_subplot(gs[3:5, 0])
    plt.plot(runner.mon.ts, runner.mon["post.input"], label="PSC", color="#d62728")
    plt.legend(loc="upper right")
    plt.xticks([])
    fig.add_subplot(gs[5:7, 0])
    plt.plot(runner.mon.ts, runner.mon["post.V"], label="post.V")
    plt.legend(loc="upper right")
    plt.xlabel("t (ms)")
    plt.show()


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

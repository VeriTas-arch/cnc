import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt


class Alpha(bp.dyn.SynConn):
    def __init__(
        self,
        pre,
        post,
        conn,
        type,
        g_max=5.0,
        tau=5.0,
        delay_step=2,
        E=0.0,
        V_rest=-65.0,
    ):
        super().__init__(pre=pre, post=post, conn=conn)
        self.g_max = g_max
        self.tau = tau
        self.E = E
        self.V_rest = V_rest
        self.delay_step = delay_step
        self.type = type  # CUBA / COBA

        self.pre2post = self.conn.require("pre2post")
        self.g = bm.Variable(bm.zeros(self.post.num))
        self.h = bm.Variable(bm.zeros(self.post.num))
        self.delay = bm.LengthDelay(self.pre.spike, delay_step)

        self.ini_h = bp.odeint(f=lambda h, t: -h / self.tau, method="exp_auto")
        self.int_g = bp.odeint(f=lambda g, t, h: -g / self.tau + h, method="exp_auto")

    def update(self):
        t = bp.share["t"]
        dt = bp.share["dt"]
        delayed_pre_spike = self.delay(self.delay_step)
        self.delay.update(self.pre.spike)

        post_sp = bm.pre2post_event_sum(
            delayed_pre_spike, self.pre2post, self.post.num, self.g_max
        )

        self.h.value = self.ini_h(self.h, t, dt) + post_sp
        self.g.value = self.int_g(self.g, t, self.h, dt)

        if self.type == "CUBA":
            self.post.input += self.g * (self.E - self.V_rest)
        elif self.type == "COBA":
            self.post.input += self.g * (self.E - self.post.V)


def run_syn(
    syn_model, type, title, run_duration=200.0, sp_times=(25, 50, 75, 100, 150)
):
    neu1 = bp.neurons.SpikeTimeGroup(1, times=sp_times, indices=[0] * len(sp_times))
    neu2 = bp.neurons.HH(1, V_initializer=bp.initialize.Constant(-50.68))
    syn1 = syn_model(neu1, neu2, conn=bp.connect.All2All(), type=type)
    net = bp.DynSysGroup(pre=neu1, syn=syn1, post=neu2)

    runner = bp.DSRunner(net, monitors=["pre.spike", "post.V", "syn.g", "post.input"])
    runner.predict(run_duration)

    fig, gs = bp.visualize.get_figure(7, 1, 0.5, 6.0)

    ax = fig.add_subplot(gs[0, 0])
    plt.plot(runner.mon.ts, runner.mon["pre.spike"], label="pre.spike")
    plt.legend(loc="upper right")
    plt.title(title)
    plt.xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = fig.add_subplot(gs[1:3, 0])
    plt.plot(runner.mon.ts, runner.mon["syn.g"], label="g", color="#d62728")
    plt.legend(loc="upper right")
    plt.xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = fig.add_subplot(gs[3:5, 0])
    plt.plot(runner.mon.ts, runner.mon["post.input"], label="PSC", color="#d62728")
    plt.legend(loc="upper right")
    plt.xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = fig.add_subplot(gs[5:7, 0])
    plt.plot(runner.mon.ts, runner.mon["post.V"], label="post.V")
    plt.legend(loc="upper right")
    plt.xlabel("Time [ms]")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.show()


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

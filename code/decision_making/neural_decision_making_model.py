import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt

JE = 7.0
JM = 3.0


class DecisionMakingModel(bp.dyn.NeuDyn):
    def __init__(
        self, size, In=0.6, gamma=0.1, theta=5.0, alpha=1.5, beta=3.0, tau_s=100.0
    ):
        super().__init__(size)

        # 初始化参数
        self.In = In
        self.gamma = gamma
        self.theta = theta
        self.alpha = alpha
        self.beta = beta
        self.tau_s = tau_s

        # 初始化变量
        self.s1 = bm.Variable(bm.zeros(self.num))
        self.s2 = bm.Variable(bm.zeros(self.num))
        self.r1 = bm.Variable(bm.zeros(self.num))
        self.r2 = bm.Variable(bm.zeros(self.num))
        self.I10 = bm.Variable(bm.zeros(self.num))
        self.I20 = bm.Variable(bm.zeros(self.num))
        self.I1 = bm.Variable(bm.zeros(self.num))
        self.I2 = bm.Variable(bm.zeros(self.num))

        # 噪声输入
        self.I1_noise = 0
        self.I2_noise = 0

        # 定义积分函数
        self.integral = bp.odeint(self.derivative, method="exp_auto")

    def _activation(self, x):
        return self.beta / self.gamma * bm.log1p(bm.exp((x - self.theta) / self.alpha))

    def _input_current(self, s_self, s_other, I_ext):
        return JE * s_self - JM * s_other + I_ext

    @property
    def derivative(self):
        return bp.JointEq([self.ds1, self.ds2])

    def ds1(self, s1, t, s2, I10=0.6):
        I1 = I10 + self.I1_noise
        x1 = self._input_current(s1, s2, I1)
        r1 = self._activation(x1)
        return (-s1 + (1.0 - s1) * self.gamma * r1) / self.tau_s

    def ds2(self, s2, t, s1, I20=0.6):
        I2 = I20 + self.I2_noise
        x2 = self._input_current(s2, s1, I2)
        r2 = self._activation(x2)
        return (-s2 + (1.0 - s2) * self.gamma * r2) / self.tau_s

    def update(self):
        # 更新噪声（每次迭代都重新计算 I1, I2 中的噪声）
        self.I1_noise = self.In * bm.random.randn(self.num) * bm.sqrt(bp.share["dt"])
        self.I2_noise = self.In * bm.random.randn(self.num) * bm.sqrt(bp.share["dt"])

        # 更新 s1、s2
        integral = self.integral(
            self.s1,
            self.s2,
            bp.share["t"],
            I10=self.I10,
            I20=self.I20,
            dt=bp.share["dt"],
        )
        self.s1.value, self.s2.value = integral

        # 用更新后的 s1、s2 计算 r1、r2
        I1 = self.I10 + self.I1_noise
        x1 = self._input_current(self.s1, self.s2, I1)
        self.r1.value = self._activation(x1)
        self.I1.value = I1

        I2 = self.I20 + self.I2_noise
        x2 = self._input_current(self.s2, self.s1, I2)
        self.r2.value = self._activation(x2)
        self.I2.value = I2

        # 重置外部输入
        self.I10[:] = 0.0
        self.I20[:] = 0.0


# 生成模型
dmnet = DecisionMakingModel(1, In=0.6)

# 定义各个阶段的时长
pre_stimulus_period, stimulus_period = 0.0, 10000.0

simulation_dt = 0.1

# 定义电流随时间的变化
inputs, total_period = bp.inputs.section_input(
    values=[0, [0.66, 0.6]],  # I10, I20 的取值
    durations=[pre_stimulus_period, stimulus_period],
    return_length=True,
    dt=simulation_dt,
)

# 运行数值模拟
runner = bp.DSRunner(
    dmnet,
    monitors=["s1", "s2", "r1", "r2", "I1", "I2"],
    dt=simulation_dt,
    inputs=[("I10", inputs[:, 0], "iter"), ("I20", inputs[:, 1], "iter")],
)
runner.predict(total_period)


# 以 tau_s 为时间尺度，等间隔地从 runner 保存的变量中取值
indur = int(dmnet.tau_s)
n_steps = runner.mon.ts.shape[0]
sampled = slice(0, n_steps - 1, indur)

x_ts = runner.mon.ts[sampled] / indur
y_s1 = runner.mon.s1[sampled]
y_s2 = runner.mon.s2[sampled]
y_r1 = runner.mon.r1[sampled]
y_r2 = runner.mon.r2[sampled]
y_I1 = runner.mon.I1[sampled]
y_I2 = runner.mon.I2[sampled]

# 可视化
fig, gs = plt.subplots(3, 1, figsize=(8, 8), sharex="all")
gs[0].plot(x_ts, y_s1, label="s1")
gs[0].plot(x_ts, y_s2, label="s2")
gs[0].axvline(
    pre_stimulus_period / indur, 0.0, 1.0, linestyle="dashed", color="#444444"
)
gs[0].axvline(
    (pre_stimulus_period + stimulus_period) / indur,
    0.0,
    1.0,
    linestyle="dashed",
    color="#444444",
)
gs[0].set_ylabel(r"gating variable $s$", fontsize=16)
gs[0].legend()

gs[1].plot(x_ts, y_r1, label="r1")
gs[1].plot(x_ts, y_r2, label="r2")
gs[1].axvline(
    pre_stimulus_period / indur, 0.0, 1.0, linestyle="dashed", color="#444444"
)
gs[1].axvline(
    (pre_stimulus_period + stimulus_period) / indur,
    0.0,
    1.0,
    linestyle="dashed",
    color="#444444",
)
gs[1].set_ylabel(r"firing rate $r$", fontsize=16)
gs[1].legend()

gs[2].plot(x_ts, y_I1, label="I1")
gs[2].plot(x_ts, y_I2, label="I2")
gs[2].axvline(
    pre_stimulus_period / indur, 0.0, 1.0, linestyle="dashed", color="#444444"
)
gs[2].axvline(
    (pre_stimulus_period + stimulus_period) / indur,
    0.0,
    1.0,
    linestyle="dashed",
    color="#444444",
)
gs[2].set_xlabel("time(" r"$\tau_s$)", fontsize=16)
gs[2].set_ylabel(r"$I_{ext}$", fontsize=16)
gs[2].legend()

plt.subplots_adjust(hspace=0.1)
plt.show()


## DM model 的动力学分析
bp.math.enable_x64()
# I1 = I2 的平均输入强度，修改其值以观察相平面图的变化
I0 = 0.6
model = DecisionMakingModel(1)
# 构建相平面分析器
analyzer = bp.analysis.PhasePlane2D(
    model=model,
    target_vars={"s1": [0, 1], "s2": [0, 1]},  # 规定 s1, s2 的取值范围
    pars_update={"I10": I0, "I20": I0},  # 令 I1，I2 均值相等
    resolutions={"s1": 0.001, "s2": 0.001},
)

## 画出相图
plt.figure(figsize=(4.5, 4.5))
# 画出向量场
analyzer.plot_vector_field(plot_style=dict(color="lightgrey"))
# 画出零增长等值线
analyzer.plot_nullcline(
    coords=dict(s2="s2-s1"), x_style={"fmt": "--"}, y_style={"fmt": "--"}
)
# 画出奇点
analyzer.plot_fixed_point(tol_aux=2e-10)

# 画出 s1, s2 的运动轨迹
plt.plot(
    runner.mon.s1, runner.mon.s2, color="red", label="s1,s2 trajectory (from zero)"
)
plt.legend()
plt.show()

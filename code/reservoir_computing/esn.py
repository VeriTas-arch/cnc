import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np


class LorenzEq:
    """Local copy of the Lorenz-system generator used from brainpy_datasets."""

    def __init__(
        self,
        duration,
        dt=0.001,
        sigma=10,
        beta=8 / 3,
        rho=28,
        method="rk4",
        inits=None,
        numpy_mon=False,
        t_transform=None,
        x_transform=None,
        y_transform=None,
        z_transform=None,
    ):
        self.t_transform = t_transform
        self.x_transform = x_transform
        self.y_transform = y_transform
        self.z_transform = z_transform

        def dx(x, t, y):
            return sigma * (y - x)

        def dy(y, t, x, z):
            return x * (rho - z) - y

        def dz(z, t, x, y):
            return x * y - beta * z

        integral = bp.odeint(bp.JointEq([dx, dy, dz]), method=method)

        res = _three_variable_model(
            integral,
            default_inits={"x": 8, "y": 1, "z": 1},
            duration=duration,
            dt=dt,
            inits=inits,
            numpy_mon=numpy_mon,
        )
        self.ts = res["ts"]
        self.xs = res["x"]
        self.ys = res["y"]
        self.zs = res["z"]

    def __len__(self):
        return self.ts.size

    def __getitem__(self, item):
        t, x, y, z = self.ts, self.xs[item], self.ys[item], self.zs[item]
        if self.t_transform is not None:
            t = self.t_transform(t)
        if self.x_transform is not None:
            x = self.x_transform(x)
        if self.y_transform is not None:
            y = self.y_transform(y)
        if self.z_transform is not None:
            z = self.z_transform(z)
        return t, x, y, z


def _three_variable_model(
    integrator,
    duration,
    default_inits,
    inits=None,
    args=None,
    dyn_args=None,
    dt=0.001,
    numpy_mon=False,
):
    if inits is None:
        inits = default_inits
    elif isinstance(inits, dict):
        assert "x" in inits
        assert "y" in inits
        assert "z" in inits
        inits = {
            "x": bm.asarray(inits["x"]).flatten(),
            "y": bm.asarray(inits["y"]).flatten(),
            "z": bm.asarray(inits["z"]).flatten(),
        }
        assert inits["x"].shape == inits["y"].shape == inits["z"].shape
    else:
        raise ValueError

    runner = bp.IntegratorRunner(
        integrator,
        monitors=["x", "y", "z"],
        inits=inits,
        args=args,
        dyn_args=dyn_args,
        dt=dt,
        progress_bar=False,
        numpy_mon_after_run=numpy_mon,
    )
    runner.run(duration)
    return {
        "ts": runner.mon["ts"],
        "x": runner.mon["x"],
        "y": runner.mon["y"],
        "z": runner.mon["z"],
    }


class ESN(bp.DynamicalSystem):
    def __init__(
        self,
        num_in,
        num_rec,
        num_out,
        lambda_max=0.9,
        W_in_initializer=bp.init.Uniform(-0.1, 0.1, seed=345),
        W_rec_initializer=bp.init.Normal(scale=0.1, seed=456),
        in_connectivity=0.05,
        rec_connectivity=0.05,
    ):
        super().__init__(mode=bm.BatchingMode())

        self.num_in = num_in
        self.num_rec = num_rec
        self.num_out = num_out
        self.rng = bm.random.RandomState(1)  # 随机数生成器

        # 初始化连接矩阵
        self.W_in = W_in_initializer((num_in, num_rec))
        conn_mat = self.rng.random((num_in, num_rec)) > in_connectivity
        self.W_in = bm.where(conn_mat, 0.0, self.W_in)  # 按连接概率削减连接度

        self.W = W_rec_initializer((num_rec, num_rec))
        conn_mat = self.rng.random(self.W.shape) > rec_connectivity
        self.W = bm.where(conn_mat, 0.0, self.W)  # 按连接概率削减连接度

        # 定义输出层
        self.readout = bp.dnn.Dense(
            num_rec, num_out, W_initializer=bp.init.Normal(), mode=bm.TrainingMode()
        )

        # 缩放 W，使 ESN 具有回声性质
        spectral_radius = bm.max(bm.abs(bm.linalg.eigvals(self.W)))  # 计算谱半径
        self.W *= lambda_max / spectral_radius  # 根据谱半径缩放 W

        # 初始化变量
        self.state = bm.Variable(bm.zeros((1, num_rec)), batch_axis=0)  # 神经元状态
        self.y = bm.Variable(bm.zeros((1, num_out)), batch_axis=0)  # 库网络输出

    # 重置函数：重置模型中各变量的值
    def reset_state(self, batch_size=None):
        if batch_size is None:
            self.state.value = bm.zeros(self.state.shape)
            self.y.value = bm.zeros(self.y.shape)
        else:
            self.state.value = bm.zeros((int(batch_size),) + self.state.shape[1:])
            self.y.value = bm.zeros((int(batch_size),) + self.y.shape[1:])

    def update(self, u):
        self.state.value = bm.tanh(bm.dot(u, self.W_in) + bm.dot(self.state, self.W))
        out = self.readout(self.state.value)
        self.y.value = out
        return out


def show_ESN_property():
    num_in = 10
    num_res = 500
    num_out = 30
    num_step = 500  # 模拟总步长
    num_batch = 1

    # 生成网络，运行两次模拟，两次模拟的输入相同，但网络的初始化状态不同
    def get_esn_states(lambda_max):
        model = ESN(num_in, num_res, num_out, lambda_max=lambda_max)
        model.reset_state(batch_size=num_batch)

        inputs = bm.random.randn(
            num_batch, int(num_step / num_batch), num_in
        )  # 第 0 个维度为 batch 的大小

        # 第一次运行
        model.state.value = bp.init.Uniform(-1.0, 1.0, seed=123)(
            (num_batch, num_res)
        )  # 随机初始化网络状态
        runner = bp.DSTrainer(model, monitors=["state"])
        runner.predict(inputs)
        state1 = np.concatenate(runner.mon["state"], axis=0)

        # 第二次运行
        model.state.value = bp.init.Uniform(-1.0, 1.0, seed=234)(
            (num_batch, num_res)
        )  # 再次随机初始化网络状态
        runner = bp.DSTrainer(model, monitors=["state"])
        runner.predict(inputs)
        state2 = np.concatenate(runner.mon["state"], axis=0)

        return state1, state2

    # 画出两次模拟中某一时刻网络的状态
    def plot_states(state1, state2, title):
        assert len(state1) == len(state2)
        x = np.arange(len(state1))
        plt.plot(x, state1, marker=".", markersize=4, linestyle="", label="first state")
        plt.plot(
            x, state2, marker="+", markersize=4, linestyle="", label="second state"
        )
        plt.legend(loc="upper right")
        plt.xlabel("Neuron index")
        plt.ylabel("State")
        plt.title(title)

    bm.random.seed(54362)

    fig, gs = bp.visualize.get_figure(1, 1, 4.5, 4)
    ax = fig.add_subplot(gs[0, 0])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    lambda1, lambda2, lambda3 = 0.9, 1.0, 1.1
    lambda1_label = rf"$|\lambda_{{max}}|={lambda1}$"
    lambda2_label = rf"$|\lambda_{{max}}|={lambda2}$"
    lambda3_label = rf"$|\lambda_{{max}}|={lambda3}$"
    # 画出每个 lambda_max 下两次模拟的网络状态的距离随时间的变化
    state1, state2 = get_esn_states(lambda_max=lambda1)
    distance = np.sqrt(np.sum(np.square(state1 - state2), axis=1))
    plt.plot(np.arange(num_step), distance, label=lambda1_label)
    plt.annotate(
        lambda1_label, xy=(22, 0.4), xytext=(60, 4.0), arrowprops=dict(arrowstyle="->")
    )

    state3, state4 = get_esn_states(lambda_max=lambda2)
    distance = np.sqrt(np.sum(np.square(state3 - state4), axis=1))
    plt.plot(np.arange(num_step), distance, label=lambda2_label)
    plt.annotate(
        lambda2_label,
        xy=(84.5, 0.4),
        xytext=(150, 1.7),
        arrowprops=dict(arrowstyle="->"),
    )

    state5, state6 = get_esn_states(lambda_max=lambda3)
    distance = np.sqrt(np.sum(np.square(state5 - state6), axis=1))
    plt.plot(np.arange(num_step), distance, label=lambda3_label)
    plt.text(337, 10, lambda3_label)

    plt.xlabel("Running step")
    plt.ylabel("Distance")

    # 画出两次模拟时网络的初始状态和最终状态
    fig, gs = bp.visualize.get_figure(2, 1, 2.25, 4)
    ax = fig.add_subplot(gs[0, 0])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plot_states(state1[0], state2[0], title=rf"$|\lambda_{{max}}|={lambda1}, n=0$")
    ax.set_xticks([])
    ax.set_xlabel("")
    ax = fig.add_subplot(gs[1, 0])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plot_states(
        state1[-1], state2[-1], title=rf"$|\lambda_{{max}}|={lambda1}, n={num_step}$"
    )

    fig, gs = bp.visualize.get_figure(2, 1, 2.25, 4)
    ax = fig.add_subplot(gs[0, 0])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plot_states(state5[0], state6[0], title=rf"$|\lambda_{{max}}|={lambda3}, n=0$")
    ax.set_xticks([])
    ax.set_xlabel("")
    ax = fig.add_subplot(gs[1, 0])
    plot_states(
        state5[-1], state6[-1], title=rf"$|\lambda_{{max}}|={lambda3}, n={num_step}$"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.show()


def fit_sine_wave(training_method="force"):
    bm.enable_x64()  # 使用更高精度的 float 以提高训练精度

    num_in, num_res, num_out = 1, 600, 1
    num_step = 1000  # 模拟总步长
    num_discard = 200  # 训练时，丢弃掉前 200 个数据

    def plot_result(output, Y, title):
        assert output.shape == Y.shape
        x = np.arange(output.shape[0])
        plt.plot(x, Y, linestyle="--", label="$y$")
        plt.plot(x, output, label=r"$\hat{y}$")
        plt.legend()
        plt.xlabel("Running step")
        plt.ylabel("State")
        plt.title(title)

    # 生成训练数据
    n = bm.linspace(0.0, bm.pi, num_step)
    U = bm.sin(10 * n) + bm.random.normal(scale=0.01, size=num_step)  # 输入
    U = U.reshape((1, -1, num_in))  # 维度：(num_batch, num_step, num_dim)
    Y = bm.power(bm.sin(10 * n), 7)  # 输出
    Y = Y.reshape((1, -1, num_out))  # 维度：(num_batch, num_step, num_dim)

    model = ESN(num_in, num_res, num_out, lambda_max=1)

    # 训练前，运行模型得到结果
    runner = bp.DSTrainer(model, monitors=["state"])
    untrained_out = runner.predict(U)
    print(
        bp.losses.mean_absolute_error(
            untrained_out[:, num_discard:], Y[:, num_discard:]
        )
    )

    if training_method not in ["ridge", "force"]:
        raise ValueError("training_method must be either 'ridge' or 'force'.")
    elif training_method == "ridge":
        # 用岭回归法训练，注意此处 alpha 为岭回归的正则化参数
        trainer = bp.RidgeTrainer(model, alpha=1e-12)
        trainer.fit([U[:, num_discard:], Y[:, num_discard:]])
    elif training_method == "force":
        # 用 FORCE 学习法训练，注意此处 alpha 为 P 矩阵初始化的参数
        trainer = bp.ForceTrainer(model, alpha=100)
        trainer.fit([U[:, num_discard:], Y[:, num_discard:]])

    # 训练后，运行模型得到结果
    runner = bp.DSTrainer(model, monitors=["state"])
    out = runner.predict(U)
    print(bp.losses.mean_absolute_error(out[:, num_discard:], Y[:, num_discard:]))

    # 可视化
    fig, gs = bp.visualize.get_figure(1, 1, 4.5, 6)
    ax = fig.add_subplot(gs[0, 0])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plot_result(
        untrained_out.flatten()[num_discard:],
        Y.flatten()[num_discard:],
        "Before training",
    )

    fig, gs = bp.visualize.get_figure(1, 1, 4.5, 6)
    ax = fig.add_subplot(gs[0, 0])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plot_result(
        out.flatten()[num_discard:], Y.flatten()[num_discard:], "After training"
    )
    plt.show()

    max_ = 0
    rng = np.random.RandomState(12354)
    i1, i2, i3, i4 = tuple(rng.choice(np.arange(num_res), 4, replace=False))
    fig, gs = bp.visualize.get_figure(1, 3, 3, 4)
    state = runner.mon["state"].squeeze()
    ax1 = fig.add_subplot(gs[0, 0])
    plt.plot(np.arange(num_step - num_discard), state[num_discard:, i1])
    plt.title("Neuron {}".format(i1))
    plt.xlabel("Running step")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    if max_ < state[num_discard:, i1].max():
        max_ = state[num_discard:, i1].max()

    ax2 = fig.add_subplot(gs[0, 1])
    plt.plot(np.arange(num_step - num_discard), state[num_discard:, i2])
    plt.title("Neuron {}".format(i2))
    plt.xlabel("Running step")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    if max_ < state[num_discard:, i2].max():
        max_ = state[num_discard:, i2].max()

    ax3 = fig.add_subplot(gs[0, 2])
    plt.plot(np.arange(num_step - num_discard), state[num_discard:, i3])
    plt.title("Neuron {}".format(i3))
    plt.xlabel("Running step")
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    if max_ < state[num_discard:, i3].max():
        max_ = state[num_discard:, i3].max()

    max_ *= 1.1
    ax1.set_ylim(-max_, max_)
    ax2.set_ylim(-max_, max_)
    ax3.set_ylim(-max_, max_)

    plt.show()


def fit_Lorenz_system(predict_step=200, training_method="force"):
    bm.enable_x64()
    predict_step = int(predict_step)
    if predict_step <= 0:
        raise ValueError("predict_step must be positive.")

    # 生成洛伦兹系统的数据
    lorenz = LorenzEq(100)
    data = bm.hstack([lorenz.xs, lorenz.ys, lorenz.zs])

    # Y 比 X 提前 predict_step 个步长，即需要预测系统未来的 Y
    X, Y = data[:-predict_step], data[predict_step:]
    # 将第 0 维扩展为 batch 的维度
    X = bm.expand_dims(X, axis=0)
    Y = bm.expand_dims(Y, axis=0)

    num_in, num_res, num_out = 3, 200, 3
    num_discard = 50

    model = ESN(num_in, num_res, num_out, lambda_max=0.9)

    def training_lorenz(trainer, title):
        trainer.fit([X[:, :30000, :], Y[:, :30000, :]])  # 用前 30000 个时间的数据来训练

        predict = trainer.predict(X, reset_state=True)
        predict = bm.as_numpy(predict)

        fig, gs = bp.visualize.get_figure(1, 1, 4.5, 6)
        ax = fig.add_subplot(gs[:, 0], projection="3d")
        # 画图时舍去最初 50 个步长的数据，下同
        plt.plot(
            Y[0, num_discard:, 0],
            Y[0, num_discard:, 1],
            Y[0, num_discard:, 2],
            alpha=0.8,
            label="standard output",
            linestyle="--",
        )
        plt.plot(
            predict[0, num_discard:, 0],
            predict[0, num_discard:, 1],
            predict[0, num_discard:, 2],
            alpha=0.8,
            label="prediction",
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.title(title)
        plt.legend()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig, gs = bp.visualize.get_figure(2, 1, 2.25, 6)
        ax = fig.add_subplot(gs[0, 0])
        t = np.arange(Y.shape[1])[num_discard:]
        plt.plot(
            t, Y[0, num_discard:, 0], linewidth=1, label="standard $x$", linestyle="--"
        )  # 洛伦兹系统中的 x 变量
        plt.plot(t, predict[0, num_discard:, 0], linewidth=1, label="predicted $x$")
        plt.ylabel(r"$x$")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xticks([])
        plt.legend()

        ax = fig.add_subplot(gs[1, 0])
        plt.plot(
            t, Y[0, num_discard:, 2], linewidth=1, label="standard $z$", linestyle="--"
        )  # 洛伦兹系统中的 z 变量
        plt.plot(t, predict[0, num_discard:, 2], linewidth=1, label="predicted $z$")
        plt.ylabel(r"$z$")
        plt.xlabel("Time step")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.legend()

        plt.show()

    if training_method not in ["ridge", "force"]:
        raise ValueError("training_method must be either 'ridge' or 'force'.")
    elif training_method == "ridge":
        # 用岭回归法训练
        ridge_trainer = bp.RidgeTrainer(model, alpha=1e-6)
        training_lorenz(ridge_trainer, "Training with Ridge Regression")
    elif training_method == "force":
        # 用 FORCE 学习法训练
        force_trainer = bp.ForceTrainer(model, alpha=1e-6)
        training_lorenz(force_trainer, "Training with FORCE Learning")


if __name__ == "__main__":
    # ------ Basic property of ESN ------
    show_ESN_property()

    # ------ Fit sine wave with ESN ------
    fit_sine_wave(training_method="force")
    fit_sine_wave(training_method="ridge")

    # ------ Fit Lorenz system with ESN ------
    fit_Lorenz_system(200, training_method="force")
    fit_Lorenz_system(200, training_method="ridge")

    # ------ An unsuccessful case ------
    fit_Lorenz_system(2000, training_method="force")
    fit_Lorenz_system(2000, training_method="ridge")

import matplotlib.pyplot as plt
import numpy as np


class LorenzEq:
    """Lorenz-system trajectory generator."""

    def __init__(
        self,
        duration,
        dt=0.001,
        sigma=10,
        beta=8 / 3,
        rho=28,
        method="rk4",
        inits=None,
        t_transform=None,
        x_transform=None,
        y_transform=None,
        z_transform=None,
    ):
        if method != "rk4":
            raise ValueError("Only rk4 is supported in the NumPy implementation.")
        self.t_transform = t_transform
        self.x_transform = x_transform
        self.y_transform = y_transform
        self.z_transform = z_transform

        if inits is None:
            state = np.asarray([8.0, 1.0, 1.0], dtype=float)
        elif isinstance(inits, dict):
            state = np.asarray(
                [inits["x"], inits["y"], inits["z"]], dtype=float
            ).reshape(3)
        else:
            raise ValueError

        num_step = int(duration / dt)
        ts = np.arange(num_step, dtype=float) * dt
        data = np.zeros((num_step, 3), dtype=float)

        def derivative(values):
            x, y, z = values
            return np.asarray(
                [sigma * (y - x), x * (rho - z) - y, x * y - beta * z], dtype=float
            )

        for i in range(num_step):
            data[i] = state
            k1 = derivative(state)
            k2 = derivative(state + 0.5 * dt * k1)
            k3 = derivative(state + 0.5 * dt * k2)
            k4 = derivative(state + dt * k3)
            state = state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

        self.ts = ts
        self.xs = data[:, 0:1]
        self.ys = data[:, 1:2]
        self.zs = data[:, 2:3]

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


class ESN:
    def __init__(
        self,
        num_in,
        num_rec,
        num_out,
        lambda_max=0.9,
        W_in_initializer=None,
        W_rec_initializer=None,
        in_connectivity=0.05,
        rec_connectivity=0.05,
    ):
        self.num_in = num_in
        self.num_rec = num_rec
        self.num_out = num_out
        self.rng = np.random.default_rng(1)  # 随机数生成器

        if W_in_initializer is None:

            def W_in_initializer(shape):
                return np.random.default_rng(345).uniform(-0.1, 0.1, shape)

        if W_rec_initializer is None:

            def W_rec_initializer(shape):
                return np.random.default_rng(456).normal(0.0, 0.1, shape)

        # 初始化连接矩阵
        self.W_in = W_in_initializer((num_in, num_rec))
        conn_mat = self.rng.random((num_in, num_rec)) > in_connectivity
        self.W_in = np.where(conn_mat, 0.0, self.W_in)  # 按连接概率削减连接度

        self.W = W_rec_initializer((num_rec, num_rec))
        conn_mat = self.rng.random(self.W.shape) > rec_connectivity
        self.W = np.where(conn_mat, 0.0, self.W)  # 按连接概率削减连接度

        # 定义输出层
        self.W_out = np.random.default_rng(789).normal(0.0, 1.0, (num_rec, num_out))
        self.b_out = np.zeros(num_out, dtype=float)

        # 缩放 W，使 ESN 具有回声性质
        spectral_radius = np.max(np.abs(np.linalg.eigvals(self.W)))  # 计算谱半径
        self.W *= lambda_max / spectral_radius  # 根据谱半径缩放 W

        # 初始化变量
        self.state = np.zeros((1, num_rec), dtype=float)  # 神经元状态
        self.y = np.zeros((1, num_out), dtype=float)  # 库网络输出

    # 重置函数：重置模型中各变量的值
    def reset_state(self, batch_size=None):
        if batch_size is None:
            self.state = np.zeros_like(self.state)
            self.y = np.zeros_like(self.y)
        else:
            self.state = np.zeros((int(batch_size), self.num_rec), dtype=float)
            self.y = np.zeros((int(batch_size), self.num_out), dtype=float)

    def update(self, u):
        self.state = np.tanh(u @ self.W_in + self.state @ self.W)
        out = self.state @ self.W_out + self.b_out
        self.y = out
        return out

    def predict(self, inputs, reset_state=False, collect_state=False):
        inputs = np.asarray(inputs, dtype=float)
        if inputs.ndim != 3:
            raise ValueError("inputs must have shape (batch, time, feature).")
        if reset_state or self.state.shape[0] != inputs.shape[0]:
            self.reset_state(batch_size=inputs.shape[0])

        outputs = np.zeros(inputs.shape[:2] + (self.num_out,), dtype=float)
        states = np.zeros(inputs.shape[:2] + (self.num_rec,), dtype=float)
        for i in range(inputs.shape[1]):
            outputs[:, i] = self.update(inputs[:, i])
            if collect_state:
                states[:, i] = self.state
        return (outputs, states) if collect_state else outputs

    def fit_ridge(self, inputs, targets, alpha=1e-7, reset_state=False):
        _, states = self.predict(inputs, reset_state=reset_state, collect_state=True)
        x = states.reshape(-1, self.num_rec)
        y = np.asarray(targets, dtype=float).reshape(-1, self.num_out)
        x_aug = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
        reg = alpha * np.eye(x_aug.shape[1])
        weights = np.linalg.pinv(x_aug.T @ x_aug + reg) @ (x_aug.T @ y)
        self.b_out = weights[0]
        self.W_out = weights[1:]
        return self.predict(inputs, reset_state=reset_state)

    def fit_force(self, inputs, targets, alpha=1.0, reset_state=False):
        inputs = np.asarray(inputs, dtype=float)
        targets = np.asarray(targets, dtype=float)
        if reset_state or self.state.shape[0] != inputs.shape[0]:
            self.reset_state(batch_size=inputs.shape[0])

        P = np.eye(self.num_rec + 1) * alpha
        weights = np.vstack([self.b_out.reshape(1, -1), self.W_out])
        outputs = np.zeros(targets.shape, dtype=float)

        for i in range(inputs.shape[1]):
            self.state = np.tanh(inputs[:, i] @ self.W_in + self.state @ self.W)
            x_aug = np.concatenate(
                [np.ones((self.state.shape[0], 1)), self.state], axis=1
            )
            out = x_aug @ weights
            outputs[:, i] = out

            for b in range(inputs.shape[0]):
                x = x_aug[b : b + 1]
                k = P @ x.T
                c = 1.0 / (1.0 + (x @ k)[0, 0])
                P -= c * (k @ k.T)
                error = out[b : b + 1] - targets[b : b + 1, i]
                weights -= c * (k @ error)

        self.b_out = weights[0]
        self.W_out = weights[1:]
        return outputs


def mean_absolute_error(output, target):
    return np.mean(np.abs(output - target))


def hide_top_right_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


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

        inputs = np.random.randn(
            num_batch, int(num_step / num_batch), num_in
        )  # 第 0 个维度为 batch 的大小

        # 第一次运行
        model.state = np.random.default_rng(123).uniform(
            -1.0, 1.0, (num_batch, num_res)
        )  # 随机初始化网络状态
        _, state1 = model.predict(inputs, collect_state=True)
        state1 = state1.reshape(-1, num_res)

        # 第二次运行
        model.state = np.random.default_rng(234).uniform(
            -1.0, 1.0, (num_batch, num_res)
        )  # 再次随机初始化网络状态
        _, state2 = model.predict(inputs, collect_state=True)
        state2 = state2.reshape(-1, num_res)

        return state1, state2

    # 画出两次模拟中某一时刻网络的状态
    def plot_states(ax, state1, state2, title):
        assert len(state1) == len(state2)
        x = np.arange(len(state1))
        ax.plot(x, state1, marker=".", markersize=4, linestyle="", label="first state")
        ax.plot(x, state2, marker="+", markersize=4, linestyle="", label="second state")
        ax.legend(loc="upper right")
        ax.set_xlabel("Neuron index")
        ax.set_ylabel("State")
        ax.set_title(title)
        hide_top_right_spines(ax)

    np.random.seed(54362)

    fig = plt.figure(figsize=(12, 7), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.8, 1.0, 1.0])
    ax_dist = fig.add_subplot(gs[:, 0])
    hide_top_right_spines(ax_dist)

    lambda1, lambda2, lambda3 = 0.9, 1.0, 1.1
    lambda1_label = rf"$|\lambda_{{max}}|={lambda1}$"
    lambda2_label = rf"$|\lambda_{{max}}|={lambda2}$"
    lambda3_label = rf"$|\lambda_{{max}}|={lambda3}$"
    # 画出每个 lambda_max 下两次模拟的网络状态的距离随时间的变化
    state1, state2 = get_esn_states(lambda_max=lambda1)
    distance = np.sqrt(np.sum(np.square(state1 - state2), axis=1))
    ax_dist.plot(np.arange(num_step), distance, label=lambda1_label)
    ax_dist.annotate(
        lambda1_label, xy=(22, 0.4), xytext=(60, 4.0), arrowprops=dict(arrowstyle="->")
    )

    state3, state4 = get_esn_states(lambda_max=lambda2)
    distance = np.sqrt(np.sum(np.square(state3 - state4), axis=1))
    ax_dist.plot(np.arange(num_step), distance, label=lambda2_label)
    ax_dist.annotate(
        lambda2_label,
        xy=(84.5, 0.4),
        xytext=(150, 1.7),
        arrowprops=dict(arrowstyle="->"),
    )

    state5, state6 = get_esn_states(lambda_max=lambda3)
    distance = np.sqrt(np.sum(np.square(state5 - state6), axis=1))
    ax_dist.plot(np.arange(num_step), distance, label=lambda3_label)
    ax_dist.text(337, 10, lambda3_label)

    ax_dist.set_xlabel("Running step")
    ax_dist.set_ylabel("Distance")
    ax_dist.set_title("Distance between two reservoir states")

    # 画出两次模拟时网络的初始状态和最终状态
    ax = fig.add_subplot(gs[0, 1])
    plot_states(ax, state1[0], state2[0], title=rf"$|\lambda_{{max}}|={lambda1}, n=0$")
    ax.set_xticks([])
    ax.set_xlabel("")
    ax = fig.add_subplot(gs[1, 1])
    plot_states(
        ax,
        state1[-1],
        state2[-1],
        title=rf"$|\lambda_{{max}}|={lambda1}, n={num_step}$",
    )

    ax = fig.add_subplot(gs[0, 2])
    plot_states(ax, state5[0], state6[0], title=rf"$|\lambda_{{max}}|={lambda3}, n=0$")
    ax.set_xticks([])
    ax.set_xlabel("")
    ax = fig.add_subplot(gs[1, 2])
    plot_states(
        ax,
        state5[-1],
        state6[-1],
        title=rf"$|\lambda_{{max}}|={lambda3}, n={num_step}$",
    )
    plt.show()


def fit_sine_wave(training_method="force"):
    num_in, num_res, num_out = 1, 600, 1
    num_step = 1000  # 模拟总步长
    num_discard = 200  # 训练时，丢弃掉前 200 个数据

    def plot_result(ax, output, Y, title):
        assert output.shape == Y.shape
        x = np.arange(output.shape[0])
        ax.plot(x, Y, linestyle="--", label="$y$")
        ax.plot(x, output, label=r"$\hat{y}$")
        ax.legend()
        ax.set_xlabel("Running step")
        ax.set_ylabel("State")
        ax.set_title(title)
        hide_top_right_spines(ax)

    # 生成训练数据
    n = np.linspace(0.0, np.pi, num_step)
    U = np.sin(10 * n) + np.random.normal(scale=0.01, size=num_step)  # 输入
    U = U.reshape((1, -1, num_in))  # 维度：(num_batch, num_step, num_dim)
    Y = np.power(np.sin(10 * n), 7)  # 输出
    Y = Y.reshape((1, -1, num_out))  # 维度：(num_batch, num_step, num_dim)

    model = ESN(num_in, num_res, num_out, lambda_max=1)

    # 训练前，运行模型得到结果
    untrained_out, _ = model.predict(U, reset_state=True, collect_state=True)
    # print(mean_absolute_error(untrained_out[:, num_discard:], Y[:, num_discard:]))

    if training_method not in ["ridge", "force"]:
        raise ValueError("training_method must be either 'ridge' or 'force'.")
    elif training_method == "ridge":
        # 用岭回归法训练，注意此处 alpha 为岭回归的正则化参数
        model.fit_ridge(U[:, num_discard:], Y[:, num_discard:], alpha=1e-12)
    elif training_method == "force":
        # 用 FORCE 学习法训练，注意此处 alpha 为 P 矩阵初始化的参数
        model.fit_force(U[:, num_discard:], Y[:, num_discard:], alpha=100)

    # 训练后，运行模型得到结果
    out, state = model.predict(U, reset_state=True, collect_state=True)
    print(mean_absolute_error(out[:, num_discard:], Y[:, num_discard:]))

    # 可视化
    fig = plt.figure(figsize=(12, 7), constrained_layout=True)
    gs = fig.add_gridspec(2, 6)
    ax = fig.add_subplot(gs[0, :3])
    plot_result(
        ax,
        untrained_out.flatten()[num_discard:],
        Y.flatten()[num_discard:],
        "Before training",
    )

    ax = fig.add_subplot(gs[0, 3:])
    plot_result(
        ax, out.flatten()[num_discard:], Y.flatten()[num_discard:], "After training"
    )

    max_ = 0
    rng = np.random.RandomState(12354)
    i1, i2, i3, i4 = tuple(rng.choice(np.arange(num_res), 4, replace=False))
    state = state.squeeze()
    ax1 = fig.add_subplot(gs[1, :2])
    ax1.plot(np.arange(num_step - num_discard), state[num_discard:, i1])
    ax1.set_title("Neuron {}".format(i1))
    ax1.set_xlabel("Running step")
    hide_top_right_spines(ax1)
    if max_ < state[num_discard:, i1].max():
        max_ = state[num_discard:, i1].max()

    ax2 = fig.add_subplot(gs[1, 2:4])
    ax2.plot(np.arange(num_step - num_discard), state[num_discard:, i2])
    ax2.set_title("Neuron {}".format(i2))
    ax2.set_xlabel("Running step")
    hide_top_right_spines(ax2)
    if max_ < state[num_discard:, i2].max():
        max_ = state[num_discard:, i2].max()

    ax3 = fig.add_subplot(gs[1, 4:])
    ax3.plot(np.arange(num_step - num_discard), state[num_discard:, i3])
    ax3.set_title("Neuron {}".format(i3))
    ax3.set_xlabel("Running step")
    hide_top_right_spines(ax3)
    if max_ < state[num_discard:, i3].max():
        max_ = state[num_discard:, i3].max()

    max_ *= 1.1
    ax1.set_ylim(-max_, max_)
    ax2.set_ylim(-max_, max_)
    ax3.set_ylim(-max_, max_)

    plt.show()


def fit_Lorenz_system(predict_step=200, training_method="force"):
    predict_step = int(predict_step)
    if predict_step <= 0:
        raise ValueError("predict_step must be positive.")

    # 生成洛伦兹系统的数据
    lorenz = LorenzEq(100)
    data = np.hstack([lorenz.xs, lorenz.ys, lorenz.zs])

    # Y 比 X 提前 predict_step 个步长，即需要预测系统未来的 Y
    X, Y = data[:-predict_step], data[predict_step:]
    # 将第 0 维扩展为 batch 的维度
    X = np.expand_dims(X, axis=0)
    Y = np.expand_dims(Y, axis=0)

    num_in, num_res, num_out = 3, 200, 3
    num_discard = 50

    model = ESN(num_in, num_res, num_out, lambda_max=0.9)

    def training_lorenz(title):
        if training_method == "ridge":
            model.fit_ridge(
                X[:, :30000, :], Y[:, :30000, :], alpha=1e-6
            )  # 用前 30000 个时间的数据来训练
        else:
            model.fit_force(
                X[:, :30000, :], Y[:, :30000, :], alpha=1e-6
            )  # 用前 30000 个时间的数据来训练

        predict = model.predict(X, reset_state=True)

        fig = plt.figure(figsize=(12, 6), constrained_layout=True)
        gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1.0])
        ax = fig.add_subplot(gs[:, 0], projection="3d")
        # 画图时舍去最初 50 个步长的数据，下同
        ax.plot(
            Y[0, num_discard:, 0],
            Y[0, num_discard:, 1],
            Y[0, num_discard:, 2],
            alpha=0.8,
            label="standard output",
            linestyle="--",
        )
        ax.plot(
            predict[0, num_discard:, 0],
            predict[0, num_discard:, 1],
            predict[0, num_discard:, 2],
            alpha=0.8,
            label="prediction",
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(title)
        ax.legend()

        ax = fig.add_subplot(gs[0, 1])
        t = np.arange(Y.shape[1])[num_discard:]
        ax.plot(
            t, Y[0, num_discard:, 0], linewidth=1, label="standard $x$", linestyle="--"
        )  # 洛伦兹系统中的 x 变量
        ax.plot(t, predict[0, num_discard:, 0], linewidth=1, label="predicted $x$")
        ax.set_ylabel(r"$x$")
        hide_top_right_spines(ax)
        ax.set_xticks([])
        ax.legend()

        ax = fig.add_subplot(gs[1, 1])
        ax.plot(
            t, Y[0, num_discard:, 2], linewidth=1, label="standard $z$", linestyle="--"
        )  # 洛伦兹系统中的 z 变量
        ax.plot(t, predict[0, num_discard:, 2], linewidth=1, label="predicted $z$")
        ax.set_ylabel(r"$z$")
        ax.set_xlabel("Time step")
        hide_top_right_spines(ax)
        ax.legend()

        plt.show()

    if training_method not in ["ridge", "force"]:
        raise ValueError("training_method must be either 'ridge' or 'force'.")
    elif training_method == "ridge":
        # 用岭回归法训练
        training_lorenz("Training with Ridge Regression")
    elif training_method == "force":
        # 用 FORCE 学习法训练
        training_lorenz("Training with FORCE Learning")


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

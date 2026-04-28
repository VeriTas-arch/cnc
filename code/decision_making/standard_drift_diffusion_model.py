import math

import matplotlib.pyplot as plt
import numpy as np

## 参数
T = 5000
dt = 1
Tinter = math.ceil(T / dt)
n_trial = 50

run_type = 1  # 1 / 2

if run_type == 1:
    # 不同的 evidence 强度 A
    A0 = 1.5
    A = np.array(np.arange(-1, 1.2, 0.2)) * A0 * 1e-3
    A = A.reshape(1, len(A))
    c = np.array([20 * 1e-3])
    len_para = A.size
elif run_type == 2:
    # 不同的 noise 强度 c
    A = np.array([0 * 1e-3])
    C0 = 50
    c = np.array(np.arange(0, 1.1, 0.1)) * C0 * 1e-4
    c = c.reshape(1, len(c))
    len_para = c.size
else:
    raise ValueError("Invalid run type.")


def standard_drift_diffusion(A=A, c=c, n_trial=n_trial):
    # 返回 para * trials * T 的矩阵 X
    AA = A if A.size == 1 else np.repeat(A, n_trial, axis=0).T
    cc = np.repeat(c, n_trial, axis=0).T

    # X 为 para * trial * Time 矩阵
    X = np.zeros((len_para, n_trial, Tinter), dtype=float)
    x = np.zeros((AA.shape[0], n_trial), dtype=float)
    sqrt_dt = np.sqrt(dt)

    for i in range(1, Tinter):
        x = x + AA * dt + cc * np.random.randn(*x.shape) / sqrt_dt
        # 当 evidence 的绝对值超过 1 时，认为做出了决策，记录决策结果并将其余时间步设为 NaN
        x = np.where(np.abs(x) > 1, np.nan, x)
        X[:, :, i] = x

    return X


def get_DM(X):  # 得到做出的决策及做出决策的时刻
    # X 为 para * trial * Time 矩阵
    # 从时间轴上找到第一个 NaN 的位置，即做出决策的时刻，并记录对应的决策结果
    All_time = np.arange(0, T, dt)
    is_nan = np.isnan(X)
    first_nan = np.argmax(is_nan, axis=2)
    has_decision = np.any(is_nan, axis=2)

    decision = np.zeros(X.shape[:2])
    time_of_dm = np.full(X.shape[:2], np.nan)

    i_inds, j_inds = np.nonzero(has_decision)
    k_inds = first_nan[has_decision] - 1
    time_of_dm[i_inds, j_inds] = All_time[k_inds]
    decision[i_inds, j_inds] = X[i_inds, j_inds, k_inds]

    return [time_of_dm, decision]


def plot_one_paramrter(X):
    # 观察 A 取正负的影响
    a_vals = A.squeeze()
    neg_index = 2
    pos_index = -3
    neg_value = a_vals[neg_index]
    pos_value = a_vals[pos_index]

    _, ax = plt.subplots(1, 2, figsize=(15, 7), sharey="all")
    for j in range(0, 10):
        ax[0].plot(np.arange(0, T, dt), X[neg_index, j, :])
    ax[0].axhline(1, color="black", linestyle="--")
    ax[0].axhline(-1, color="black", linestyle="--")
    ax[0].set_xlabel("Time step", fontsize=18)
    ax[0].set_ylabel("Difference in Evidence", fontsize=18)
    ax[0].set_xlim(0, T)
    ax[0].set_ylim(-1.2, 1.2)
    ax[0].set_title(f"A < 0, 10 trials (A={neg_value:.3e})", fontsize=18)

    for j in range(0, 10):
        ax[1].plot(np.arange(0, T, dt), X[pos_index, j, :])
    ax[1].axhline(1, color="black", linestyle="--")
    ax[1].axhline(-1, color="black", linestyle="--")
    ax[1].set_xlabel("Time step", fontsize=18)
    ax[1].set_ylabel("Difference in Evidence", fontsize=18)
    ax[1].set_xlim(0, T)
    ax[1].set_ylim(-1.2, 1.2)
    ax[1].set_title(f"A > 0, 10 trials (A={pos_value:.3e})", fontsize=18)
    plt.tight_layout()
    plt.show()

def plot_diff_parameter(X):
    if run_type == 1:
        para = A.squeeze()
        tex_para = "A"
    else:
        para = c.squeeze()
        tex_para = "c"

    # 调用 get_DM 得到不同 A 取值下模型做出决策花费的时间，并对所有 trials 平均
    # 在这个过程中，用 have_done_decision 记录做出决策的 trials 数量
    Tlabel = para * 1e3
    time_of_DM = get_DM(X)[0]
    valid = ~np.isnan(time_of_DM)
    have_done_decision = np.sum(valid, axis=1, keepdims=True)
    mean_time_DM = np.nansum(time_of_DM, axis=1, keepdims=True) / have_done_decision

    plt.figure(figsize=[15, 7])

    for j in range(len_para):
        plt.plot(
            np.arange(0, T, dt),
            X[j, 0, :],
            label=r"$para*10^3$={}".format("%.2f" % Tlabel[j]),
        )
    plt.axhline(1, color="black", linestyle="--")
    plt.axhline(-1, color="black", linestyle="--")
    plt.xlabel("Time step", fontsize=18)
    plt.ylabel("Difference in Evidence", fontsize=18)
    plt.xlim(0, T)
    plt.ylim(-1.2, 1.2)
    plt.title("one trial, different {}".format(tex_para), fontsize=18)
    plt.legend(loc="lower right", fontsize=12)
    plt.show()

    _, px = plt.subplots(1, 2, figsize=(12, 5))
    px[0].plot((para * (10**3)).T, mean_time_DM)
    px[0].set_xlabel(r"different {} ($*10^3$)".format(tex_para), fontsize=18)
    px[0].set_title("Time of decision making", fontsize=18)

    px[1].plot((para * (10**3)).T, have_done_decision / n_trial)
    px[1].set_xlabel(r"different {} ($*10^3$)".format(tex_para), fontsize=18)
    px[1].set_title("Accuracy", fontsize=18)
    plt.show()


def plot_psy_A():
    # 画关于 A 的心理测量曲线
    Decision = get_DM(X)[1]
    Decision = np.where(Decision > 0, 1, Decision)
    Decision = np.where(Decision < 0, -1, Decision)

    # 计算在每个 A 的取值下，固定的时间 T 内，x 达到决策 +1 的概率 pA
    # 需要注意的是，当没有在 T 内做出决策时，pA 置为 0.5
    pA = np.mean(np.where(Decision > 0, Decision, 0), 1) + np.mean(
        np.where(Decision == 0, 0.5, 0), 1
    )
    plt.figure()
    xA = (A * (10**3)).T
    plt.plot(xA, pA.reshape(np.shape(A)[1], 1), "-", lw=2)
    plt.axvline(0, color="black", linestyle="--")
    plt.axhline(0.5, color="black", linestyle="--")
    plt.xlim(xA[0], xA[-1])
    plt.ylim(0, 1)
    plt.xlabel(r"$A*10^3$", fontsize=18)
    plt.ylabel("P (Decision=1)", fontsize=18)
    plt.show()


X = standard_drift_diffusion()
time_of_DM, Decision = get_DM(X)

if run_type == 1:
    plot_one_paramrter(X)
    plot_psy_A()

plot_diff_parameter(X)

import matplotlib.pyplot as plt
import numpy as np

# 参数区间
r_min, r_max = 3.4, 4.0
num_r = 5000      # r 取多少个点
iters = 1000      # 总迭代次数
last = 200        # 用最后多少次迭代画图

# r 的取值
r = np.linspace(r_min, r_max, num_r)
# 初值随机一点，避免对称性
x = np.random.rand(num_r)

plt.figure(figsize=(6, 4))

for i in range(iters):
    x = r * x * (1 - x)
    if i >= iters - last:
        # ',' 是非常小的点，'k' 是黑色
        plt.plot(r, x, ',k', alpha=0.4)

plt.xlim(r_min, r_max)
plt.ylim(0.0, 1.0)
plt.xlabel('r')
plt.ylabel('x')
plt.tight_layout()
plt.show()

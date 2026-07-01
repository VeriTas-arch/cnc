from pathlib import Path

import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np

# data_dir = Path(__file__).resolve().parent / "other_images.npy"


# test_data contains 2 samples, each of which is a 28*28 image flattened to a 784-dim vector.
# other_images contains 5 samples instead.
# data[i] is the i-th sample, with shape (784,), where i = 0, 1 for test_data and i = 0, 1, 2, 3, 4 for other_images.
data_dir = Path(__file__).resolve().parent / "test_data.npy"
data = (np.load(str(data_dir))).astype(np.float32)
image1 = data[0]
image2 = data[1]
same_of_pics = (image1.reshape(28, 28) + image2.reshape(28, 28)) / 2
same_of_pics = np.where(same_of_pics < 1, -1, same_of_pics)


# Show the training data - it's just two different samples that we would like the net to store.
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 7.5))
ax[0].imshow(image1.reshape(28, 28), interpolation="None", cmap="winter")
ax[0].set_title("image1", fontsize=24)
ax[1].imshow(image2.reshape(28, 28), interpolation="None", cmap="winter")
ax[1].set_title("image2", fontsize=24)
ax[2].imshow(same_of_pics, interpolation="None", cmap="winter")
ax[2].set_title("intersection of two images", fontsize=24)
# plt.show()
plt.savefig(
    Path(__file__).resolve().parent / "training_data.png", dpi=300, bbox_inches="tight"
)


def add_bernouli_noise(img, prob=0.4, mask_value=1.0):
    # mask_value -> 0: original; 1: corrupted

    rand = np.random.random(img.shape) < prob
    return np.where(rand, mask_value, img)


def add_mask_noise(img, mask_value=-1, mask_matrix=np.zeros((2, 2))):
    img = np.copy(img)

    # 取 mask_matrix 第一行元素作为覆盖原图片矩阵数据的行范围
    # 取 mask_matrix 第二行元素作为覆盖原图片矩阵数据的列范围
    img[
        mask_matrix[0, 0] : mask_matrix[0, 1], mask_matrix[1, 0] : mask_matrix[1, 1]
    ] = mask_value
    return img


class AmariHopfieldNet(bp.DynamicalSystem):
    def __init__(self, num):
        super().__init__()
        self.num = num
        self.weight = bm.Variable(bm.zeros([num, num]))

    # Train function for the net. (By updating the weights)
    @bm.cls_jit  # JIT 编译加速
    def store_patterns(
        self, samples
    ):  # samples: 需要存储的图片信息数据，即图片的 memory pattern
        assert samples.ndim == 2
        assert samples.shape[1] == self.num

        bm.for_loop(self.store, samples)  # 循环 2 次，每次存储一张图片的信息
        self.weight /= samples.shape[0]  # 对所有 memory patterns 的外积取平均
        bm.fill_diagonal(
            self.weight, 0
        )  # 神经元自身和自身没有连接，每次循环，将对角元素置 0

    # Storing one sample pattern
    @bm.cls_jit
    def store(self, sample):
        # sample is an array with the shape of (N,)
        assert self.num == sample.shape[0]

        # Data outer-product gives neural hopfield update rule.
        w_update = bm.outer(sample, sample)

        # Sum all pattern outer-products.
        self.weight += w_update

    def async_recover(
        self, sample, n, energy=False
    ):  # sample：cue，原始图片或加入 noise 的图片
        # 需要迭代 n 次以更新输入 images 后网络所有神经元的响应，故随机选择 n 个神经元进行更新
        idxs = bm.random.randint(0, self.num, n)
        sample = bm.Variable(sample)

        def recover(i):
            # 每次迭代，某一个神经元根据来自其他神经元的输入更新自己的响应
            sample[i] = bm.sign(bm.inner(self.weight[i], sample))
            if energy:
                # 每次迭代后计算网络的能量
                return self.energy(sample)

        r = bm.for_loop(recover, idxs)  # for loop JIT
        return (sample, r) if energy else sample

    # Computeing the nets' energy.
    def energy(self, x):
        return 0.5 * bm.inner(-x @ self.weight, x)


# There are as many neurons as pixels per pattern, i.e., 784.
net = AmariHopfieldNet(num=data.shape[1])
net.store_patterns(np.vstack((image1, image2)))

# corrupt the two images with noise and reconstruct them
# Corrupt the image2 with bernouli noise.
corrupted_image2 = add_bernouli_noise(image2, prob=0.4)

# The number of iterations where we randomly update neurons/pixels
n_iterations = 10000
cleaned_image, energy_vec = net.async_recover(corrupted_image2, n_iterations, True)

# original vs. corrupted vs. recovered image2
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 7.5))
ax[0].imshow(image2.reshape(28, 28), interpolation="None", cmap="winter")
ax[0].set_title("Original Image", fontsize=24)
ax[1].imshow(corrupted_image2.reshape(28, 28), interpolation="None", cmap="winter")
ax[1].set_title("Corrupted Image", fontsize=24)
ax[2].imshow(cleaned_image.reshape(28, 28), interpolation="None", cmap="winter")
ax[2].set_title("Recovered Image", fontsize=24)
plt.savefig(
    Path(__file__).resolve().parent / "recovery_image2.png",
    dpi=300,
    bbox_inches="tight",
)
# plt.show()

# Plot the Hopfield energy during recovery
plt.figure()
plt.plot(bm.as_numpy(energy_vec))
plt.xlabel("Iteration")
plt.ylabel("Energy")
plt.title("Image2")
# plt.grid("on")
# plt.show()
plt.savefig(
    Path(__file__).resolve().parent / "energy_image2.png", dpi=300, bbox_inches="tight"
)

# Masking and Bernouli noise
mask_matrix = np.asarray([[7, 20], [8, 25]], dtype=np.int32)
corrupted_image1 = add_mask_noise(
    image1.reshape(28, 28), mask_value=1, mask_matrix=mask_matrix
)
# corrupted_image1 = add_bernouli_noise(corrupted_image1, prob=0.3)

# The number of iterations where we randomly update neurons/pixels
n_iterations = 10000
cleaned_image, energy_vec = net.async_recover(
    corrupted_image1.flatten(), n_iterations, True
)

# (For imaging purposes)
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 7.5))
ax[0].imshow(image1.reshape(28, 28), interpolation="None", cmap="winter")
ax[0].set_title("Original Image", fontsize=24)
ax[1].imshow(corrupted_image1.reshape(28, 28), interpolation="None", cmap="winter")
ax[1].set_title("Corrupted Image", fontsize=24)
ax[2].imshow(cleaned_image.reshape(28, 28), interpolation="None", cmap="winter")
ax[2].set_title("Recovered Image", fontsize=24)
# plt.show()
plt.savefig(
    Path(__file__).resolve().parent / "recovery_image1.png",
    dpi=300,
    bbox_inches="tight",
)


# Plot the Hopfield energy during recovery
plt.figure()
plt.plot(bm.as_numpy(energy_vec))
plt.xlabel("Iteration")
plt.ylabel("Energy")
plt.title("Image1")
# plt.grid("on")
# plt.show()
plt.savefig(
    Path(__file__).resolve().parent / "energy_image1.png", dpi=300, bbox_inches="tight"
)


# Try to reconstruct the intersection of the two images, one more trials
corrupted_intersection = add_bernouli_noise(same_of_pics, prob=0.5)
recovered_interseciton, energy_vec = net.async_recover(
    corrupted_intersection.flatten(), n_iterations, True
)
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 7.5))
ax[0].imshow(same_of_pics, interpolation="None", cmap="winter")
ax[0].set_title("Original intersection", fontsize=24)
ax[1].imshow(corrupted_intersection, interpolation="None", cmap="winter")
ax[1].set_title("Corrupted intersection", fontsize=24)
ax[2].imshow(
    recovered_interseciton.reshape(28, 28), interpolation="None", cmap="winter"
)
ax[2].set_title("Recovered", fontsize=24)
# plt.show()
plt.savefig(
    Path(__file__).resolve().parent / "recovery_intersection.png",
    dpi=300,
    bbox_inches="tight",
)

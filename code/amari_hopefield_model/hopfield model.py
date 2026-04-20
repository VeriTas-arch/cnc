import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

data_dir = Path(__file__).resolve().parent / "test_data.npy"
# data_dir = Path(__file__).resolve().parent / "other_images.npy"


# test_data contains 2 samples, each of which is a 28*28 image flattened to a 784-dim vector.
# other_images contains 5 samples instead.
# data[i] is the i-th sample, with shape (784,), where i = 0, 1 for test_data and i = 0, 1, 2, 3, 4 for other_images.
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
plt.show()


def add_bernouli_noise(img, prob=0.4, mask_value=1.0):
    rand = np.random.random(img.shape) < prob
    return np.where(rand, mask_value, img)


def add_mask_noise(img, mask_value=-1, mask_matrix=np.zeros((2, 2))):
    img = np.copy(img)
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
    @bm.cls_jit
    def store_patterns(self, samples):
        # data: An array with [d, N]. 'd' is the number of data samples used to train. (In this case 2)
        assert samples.ndim == 2
        assert samples.shape[1] == self.num
        bm.for_loop(self.store, samples)

        # we need to make sure that the diagonal elements of the final weight matrix are zero.
        bm.fill_diagonal(self.weight, 0)

    # Storing one sample pattern
    @bm.cls_jit
    def store(self, sample):
        # sample is an array with the shape of (N,)
        assert sample.shape[0] == self.num

        # Data cross-product gives neural hopfield update rule.
        w_update = bm.outer(sample, sample)

        # Sum all pattern cross-products.
        self.weight += w_update

    def async_recover(self, sample, n, energy=False):
        # n: the number of iterations to recover
        # energy: calculate the energy function
        idxs = bm.random.randint(0, self.num, n)  # the sampled positions
        # JIT compilation requires to label the value to be changed as Variable
        sample = bm.Variable(sample)

        def recover(i):
            # i: the position to update
            # update
            sample[i] = bm.sign(bm.inner(self.weight[i], sample))
            # return energy
            if energy:
                return self.energy(sample)

        r = bm.for_loop(recover, idxs)  # for loop JIT
        return (sample, r) if energy else sample

    # Computeing the nets' energy.
    def energy(self, x):
        # x: [N] data vector
        return bm.inner(-x @ self.weight, x)


# There are as many neurons as pixels per pattern, i.e., 784.
net = AmariHopfieldNet(num=data.shape[1])

# store the information of the two images
net.store_patterns(np.vstack((image1, image2)))

# corrupt the two images with noise and reconstruct them
# Corrupt the image2 with bernouli noise.
corrupted_image2 = add_bernouli_noise(image2, prob=0.4)

# The number of iterations where we randomly update neurons/pixels
n_iterations = 10000

# Loop through, and recover the image from it's corrupted self.
# energy_vec: This will store the energy of the Hopfield net.
cleaned_image, energy_vec = net.async_recover(corrupted_image2, n_iterations, True)

# original vs. corrupted vs. recovered image2
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 7.5))
ax[0].imshow(image2.reshape(28, 28), interpolation="None", cmap="winter")
ax[0].set_title("Original Image2", fontsize=24)
ax[1].imshow(corrupted_image2.reshape(28, 28), interpolation="None", cmap="winter")
ax[1].set_title("Corrupted Image2", fontsize=24)
ax[2].imshow(cleaned_image.reshape(28, 28), interpolation="None", cmap="winter")
ax[2].set_title("Recovered Image2", fontsize=24)
plt.show()

# Plot the Hopfield energy during recovery
plt.figure(figsize=[9, 6])
plt.plot(bm.as_numpy(energy_vec))
plt.xlabel("Iteration", fontsize=24)
plt.ylabel("Energy", fontsize=24)
plt.title("Image2", fontsize=24)
plt.show()

# Masking and Bernouli noise
mask_matrix = np.asarray([[7, 20], [8, 25]], dtype=np.int32)
# image = data[0].reshape(28, 28)
corrupted_image1 = add_mask_noise(
    image1.reshape(28, 28), mask_value=1, mask_matrix=mask_matrix
)
# corrupted_image1 = add_bernouli_noise(corrupted_image1, prob=0.3)

# The number of iterations where we randomly update neurons/pixels
n_iterations = 10000

# Loop through, and recover the image from it's corrupted self.
# energy_vec: storing the energy of the net.
cleaned_image, energy_vec = net.async_recover(
    corrupted_image1.flatten(), n_iterations, True
)

# (For imaging purposes)
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 7.5))
ax[0].imshow(image1.reshape(28, 28), interpolation="None", cmap="winter")
ax[0].set_title("Original Image1", fontsize=24)
ax[1].imshow(corrupted_image1.reshape(28, 28), interpolation="None", cmap="winter")
ax[1].set_title("Corrupted Image1", fontsize=24)
ax[2].imshow(cleaned_image.reshape(28, 28), interpolation="None", cmap="winter")
ax[2].set_title("Recovered Image1", fontsize=24)
plt.show()


# Plot the network energy during recovery
plt.figure(figsize=[9, 6])
plt.plot(bm.as_numpy(energy_vec))
plt.xlabel("Iteration", fontsize=24)
plt.ylabel("Energy", fontsize=24)
plt.title("Image1", fontsize=24)
plt.show()


# Try to reconstruct the intersection of the two images, one more trials
corrupted_intersection = add_bernouli_noise(
    same_of_pics, prob=0.5
)  # corrupt the intersection
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
plt.show()

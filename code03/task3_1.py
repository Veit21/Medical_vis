import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from scipy import signal
from scipy import special

# Create a figure (window)
fig = plt.figure(figsize=(11, 5))

# load the image (already grayscale 2D)
img = mpimg.imread('brain_slice.jpg')

axs1 = fig.add_subplot(1, 4, 1)
axs1.set_title('Original')
axs1.imshow(img, cmap='gray')


###############################
# Task 1a
###############################
axs2 = fig.add_subplot(1, 4, 2)
axs2.set_title('Mean')

axs3 = fig.add_subplot(1, 4, 3)
axs3.set_title('Binomial')


def meanKernel(k):
	mask = np.full((k, k), 1/k)
	return mask


def binomialKernel(k):
	mask = np.zeros((k, k))
	for row in range(mask.shape[0]):
		for col in range(mask.shape[1]):
			mask[row, col] = special.binom(k - 1, row) * special.binom(k - 1, col)
	return 1/(np.power(4, k - 1)) * mask


# apply masks to original image
img_mean_filtered = signal.convolve2d(img, meanKernel(21))
img_binomial_filtered = signal.convolve2d(img, binomialKernel(21))

# plot filtered images
axs2.imshow(img_mean_filtered, cmap='gray')
axs3.imshow(img_binomial_filtered, cmap='gray')


###############################
# Task 1b
###############################
axs4 = fig.add_subplot(1, 4, 4)
axs4.set_title('Sigma')

sigma = 15
img_sigma_filtered = img_mean_filtered
for row in range(img.shape[0]):
	for col in range(img.shape[1]):
		abs_diff = abs(img_mean_filtered[row, col] - img[row, col])
		if abs_diff > sigma:
			img_sigma_filtered[row, col] = img[row, col]

# plot the filtered image
axs4.imshow(img_sigma_filtered, cmap='gray')

# Always run this, to make sure everything is displayed.
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from scipy import ndimage

# Create a figure (window)
fig = plt.figure(figsize=(11, 5))

# load the image
img = mpimg.imread('gray.png')
img = img[:, :, 0]
axs1 = fig.add_subplot(1, 3, 1)
axs1.set_title('Original')
axs1.imshow(img, cmap='gray')

# compute Gauss-filtered gradients
img_gauss = ndimage.gaussian_filter(img, 10)
grad_x = ndimage.sobel(img_gauss, axis=1)
grad_y = ndimage.sobel(img_gauss, axis=0)

# create noise image
noise = np.random.random_sample(img.shape)
axs3 = fig.add_subplot(1, 3, 2)
axs3.set_title('Noise')
axs3.imshow(noise, cmap='gray')

###############################
# Task 1
###############################

number_samples = 10
img_test = img[:50, :50]
noise_test = noise[:50, :50]
img_shape = img.shape
LIC_image = np.zeros(img_shape)


def nearest_neighbor(x, y):
    if x < (int(x) + 0.5):
        x = int(x)
    else:
        x = int(x) + 1
    if y < (int(y) + 0.5):
        y = int(y)
    else:
        y = int(y) + 1
    return x, y


weight = 0
for i in range(number_samples):
    weight += (number_samples - i) / number_samples


for row in range(img_shape[0]):     # y-dim
    for col in range(img_shape[1]):     # x-dim
        C_xy = 0
        xi, yi = col, row
        for i in range(number_samples):
            if i == 0:
                C_xy += noise[yi, xi]
            else:
                xi_last, yi_last = xi, yi
                xi = xi_last + grad_x[row, col] / np.linalg.norm([grad_x[row, col], grad_y[row, col]])
                yi = yi_last + grad_y[row, col] / np.linalg.norm([grad_x[row, col], grad_y[row, col]])
                if xi < img_shape[1] - 1 and yi < img_shape[0] - 1:
                    xi, yi = nearest_neighbor(xi, yi)
                    C_xy += noise[yi, xi] * (number_samples - i) / number_samples
                else:
                    break
        LIC_image[row, col] = C_xy / weight


# show LIC image
axs3 = fig.add_subplot(1, 3, 3)
axs3.set_title('LIC')
axs3.imshow(LIC_image, cmap='gray')

# Always run this, to make sure everything is displayed.
plt.show()
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from scipy import ndimage

# Create a figure (window)
fig = plt.figure(figsize=(11,5))

# load the image
img = mpimg.imread('gray.png')
img = img[:,:,0]
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

C = np.zeros(img.shape)
w = 0
n= 10
for w in range(n):
    w += (n-1)/n

for r in range(0, img.shape[0]):
    for k in range(0, img.shape[1]):
        C_xy = 0
        for i in range(10):
            if i == 0:
                C_xy = noise[r,k]
                xi, yi = r, k
            else:
                xi_1, yi_1 = xi, yi
                xi = xi_1 + grad_y[xi_1, yi_1]/np.linalg.norm([grad_x[xi_1,yi_1], grad_y[xi_1, yi_1]])
                yi = yi_1 + grad_x[xi_1, yi_1]/np.linalg.norm([grad_x[xi_1,yi_1], grad_y[xi_1, yi_1]])
                if xi > img.shape[0] -1 or yi > img.shape[1] -1:
                    break
                else:
                    xi, yi = nearest_neighbor(xi, yi)
                    C_xy += noise[xi, yi] * (n - i)/n
        C[r,k] = C_xy/w
             

axs4 = fig.add_subplot(1, 3, 3)
axs4.set_title('LIC')
axs4.imshow(C, cmap='gray')

# Always run this, to make sure everything is displayed.
plt.show()
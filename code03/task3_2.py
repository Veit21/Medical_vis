import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Create a figure (window)
fig = plt.figure(figsize=(10, 10))

# load the image, scale to discrete values in [0,255]
img = mpimg.imread('landscape.png')
img_bw = (img[:, :, 0] * 255).astype(int)

# display the original
axs1 = fig.add_subplot(2, 2, 1)
axs1.set_title('Original')
axs1.imshow(img_bw, cmap='gray', vmin=0, vmax=255)

###############################
# Task 2a
###############################
axs2 = fig.add_subplot(2, 2, 2)
axs2.set_title('Equalized')

# histogram and cumulative sum of the bw image.
hist_img_bw = np.histogram(img_bw, 255, (0, 255))
csum_img_bw = np.cumsum(hist_img_bw[0])

# calculate the new grey values.
v_max = max(csum_img_bw)
v_min = min(csum_img_bw)
equalized_grey_values = np.asarray([255 * (v - v_min) / (v_max - v_min) for v in csum_img_bw])

# with the help of a dictionary, match the old grey values with the new ones.
original_equalized_match_dict = {key: value for key, value in enumerate(equalized_grey_values)}

# set the new grey values for the equalized image
img_shape = np.shape(img_bw)
equalized_img = np.zeros(img_shape)
for i in range(img_shape[0]):
    for j in range(img_shape[1]):
        equalized_img[i][j] = original_equalized_match_dict[img_bw[i][j]]

# display the equalized image.
axs2.imshow(equalized_img, cmap='gray', vmin=0, vmax=255)

# histogram and cumulative sum of the equalized bw image.
hist_img_bw_equalized = np.histogram(equalized_img, 255, (0, 255))
csum_img_bw_equalized = np.cumsum(hist_img_bw_equalized[0])

###############################
# Task 2b
###############################
axs3 = fig.add_subplot(2, 2, 3)
axs3.set_title('Cumulative Histogram')
axs3.plot(np.arange(len(csum_img_bw)), csum_img_bw)
axs4 = fig.add_subplot(2, 2, 4)
axs4.set_title('Cumulative Histogram')
axs4.plot(np.arange(len(csum_img_bw_equalized)), csum_img_bw_equalized)


# Always run this, to make sure everything is displayed.
plt.show()
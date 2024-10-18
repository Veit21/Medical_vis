import numpy as np
import matplotlib.pyplot as plt

img = np.array([[0.0, 0.0, 1.0, 4.0, 4.0, 5.0],
                [0.0, 1.0, 3.0, 4.0, 3.0, 4.0],
                [1.0, 3.0, 4.0, 2.0, 1.0, 3.0],
                [4.0, 4.0, 3.0, 1.0, 0.0, 0.0],
                [5.0, 4.0, 2.0, 1.0, 0.0, 0.0],
                [5.0, 5.0, 4.0, 3.0, 1.0, 0.0]])

# Create a figure (window)
fig = plt.figure(figsize=(11, 5))

axs1 = fig.add_subplot(1, 2, 1)
axs1.set_title('Original')
axs1.imshow(img, cmap='gray')

###############################
# Task 1
###############################
num_thresholds = 6  # number of thresholds to try
min_value = np.min(img)
max_value = np.max(img)
thresholds = np.linspace(min_value, max_value, num_thresholds, dtype=int)

# create a histogram of the original image
img_histogram = np.histogram(img, 6, (0, 6))
sum_all_values = np.sum(img_histogram[0])

# calculating all the sigmas depending on the threshold and adding them to a list
variances = []
for threshold in thresholds:
    sub_array1_values = img_histogram[0][:threshold]
    sub_array2_values = img_histogram[0][threshold:]
    sub_array1_bins = img_histogram[1][:threshold]
    sub_array2_bins = img_histogram[1][threshold:6]
    sum_values1 = np.sum(sub_array1_values)
    sum_values2 = np.sum(sub_array2_values)
    p1 = sum_values1 / sum_all_values
    p2 = 1 - p1
    if sum_values1 != 0:
        mu1 = np.dot(sub_array1_values, sub_array1_bins) / sum_values1
    else:
        mu1 = 0
    if sum_values2 != 0:
        mu2 = np.dot(sub_array2_values, sub_array2_bins) / sum_values2
    else:
        mu2 = 0
    sigma2_b = p1 * p2 * (mu1 - mu2)**2
    variances.append(sigma2_b)

# get the maximum of all sigmas and thus the optimal threshold
max_sigma2_b = np.max(variances)
t_opt = variances.index(max_sigma2_b)

# do the segmentation
img_segmented = img.copy()
img_shape = np.shape(img_segmented)
for i in range(img_shape[0]):
    for j in range(img_shape[1]):
        if img_segmented[i][j] >= t_opt:
            img_segmented[i][j] = 1
        else:
            img_segmented[i][j] = 0

# plot the new segmented image
axs2 = fig.add_subplot(1, 2, 2)
axs2.set_title('Segmented')
axs2.imshow(img_segmented, cmap='gray')

# Always run this, to make sure everything is displayed.
plt.show()

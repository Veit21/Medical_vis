import numpy as np  # arrays, maths
import nrrd     # reading nrrd files
import matplotlib.pyplot as plt     # plotting
from scipy import ndimage   # image processing

# Create a figure (window)
fig = plt.figure(figsize=(15, 4))

#############################
# Plot a function
#############################
# A number sequence can be created with the arange function.
x = np.arange(0, 10, 0.05)

# Arbitrary functions can be evaluated on numpy arrays.
y = np.exp(-x) * np.cos(2 * np.pi * x)

# To read out arrays/matrices directly, they can be printed.
print(y) 

# Creating a subplot allows to plot the results.
# Here, a 1 x 4 grid of subplots is declared and position 1 is used.
axs1 = fig.add_subplot(1, 4, 1)
axs1.plot(x, y, label='function')

# Optionally define titles and labels
axs1.set_title('Simple Plot')
axs1.set_xlabel('Label (x)')
axs1.set_ylabel('Label (y)')
axs1.legend()


#############################
# Plot a 2D function with meshgrid
#############################
# This creates two arrays (x and y) with values between -3 and 3 in 0.025 increments.
x = np.arange(-3.0, 3.0, 0.025)
y = np.arange(-3.0, 3.0, 0.025)

# Meshgrid builds a matrix from two given axes.
# X is a 2D array containing all row indices.
# Y is a 2D array containing all column indices.
X, Y = np.meshgrid(x, y)

# Evaluate something, like the sin around the center (0, 0).
# Note the power-function shortcut '**', which is applied to all elements in a numpy array.
Z = np.sin(np.sqrt(X**2 + Y**2))

# Plot the result with a 3D projection
axs2 = fig.add_subplot(1, 4, 2, projection='3d')
axs2.plot_surface(X, Y, Z)
axs2.set_title('3D Projection')


#############################
# Extract slice 32 of an MRI volume and show as image
#############################
# load the volume
img_data, header = nrrd.read('MRHead.nrrd')

# extract slice 65 in z-direction
img = img_data[:,:,65]

# display using gray levels
axs3 = fig.add_subplot(1, 4, 3)
axs3.set_title('MRI Slice')
axs3.imshow(img, cmap='gray')


#############################
# Apply a filter to the image
#############################
# Scipy comes with some pre-defined filters, like the gradient magnitude
img_grad = ndimage.gaussian_gradient_magnitude(img, sigma=1)

axs4 = fig.add_subplot(1, 4, 4)
axs4.set_title('Filtered MRI Slice')
axs4.imshow(img_grad, cmap='gray')


#############################
# Always run plt.show() at the end of a file.
# This makes sure everything is displayed.
#############################
plt.show()
import numpy as np # math functionality
import matplotlib.pyplot as plt # plotting
import matplotlib.image as mpimg # loading images

# Create a figure (window)
fig = plt.figure(figsize=(14, 6))

# load the image (already grayscale 2D)
img_bw = mpimg.imread('brain_slice.jpg')

# display the original
axs1 = fig.add_subplot(1, 4, 1)
axs1.set_title('Original')
axs1.imshow(img_bw, cmap='gray')

# display the applied Fourier Transform with shift
axs2 = fig.add_subplot(1, 4, 2)
axs2.set_title('FT spectrum (shifted)')
img_spectrum = np.fft.fftshift(np.fft.fft2(img_bw))

# to display the spectrum, the complex numbers are converted
img_spectrum_abs = np.abs(img_spectrum)

# then, the array is clipped to make the value range visible
img_spectrum_abs = np.clip(img_spectrum_abs, 0, 10000)

axs2.imshow(img_spectrum_abs, cmap='gray')

###############################
# Task 2
###############################
smooth_mask = np.zeros(img_spectrum_abs.shape)
cx = img_spectrum_abs.shape[0]/2
cy = img_spectrum.shape[1]/2
r = 60

for x in range(img_spectrum_abs.shape[0]):
    for y in range(img_spectrum.shape[1]):
        if (x - cx)**2 + (y - cy)**2 <= r**2:
            smooth_mask[x][y] = 1 - (np.sqrt((x-cx)**2 + (y - cy)**2)/r)

# display the smooth mask
axs3 = fig.add_subplot(1, 4, 3)
axs3.set_title('Mask')
axs3.imshow(smooth_mask, cmap='gray')

# apply smooth mask to the fourier spectrum and reconstruct the smoothed image
img_apply_mask = smooth_mask * img_spectrum
img_spectrum_inverse = np.fft.ifft2(img_apply_mask)
img_smoothed = np.abs(img_spectrum_inverse)

# display the result
axs4 = fig.add_subplot(1, 4, 4)
axs4.set_title('Result (low pass filtered)')
axs4.imshow(img_smoothed, cmap='gray')

# Always run this, to make sure everything is displayed.
plt.show()
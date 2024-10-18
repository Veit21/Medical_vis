import nrrd
import numpy as np
import matplotlib.pyplot as plt

# load the image volume
img_data, header = nrrd.read('MRHead.nrrd')

# normalize image values
image = np.divide(img_data, float(np.max(img_data)))

# create a figure
fig = plt.figure(figsize=(16, 5))


####################
# Task 1a 
####################

ax1 = fig.add_subplot(1, 3, 1)
iMIP = np.zeros((256, 256))

for i in range(256):
    for j in range(256):
        iMIP[i][j] = np.max(image[i][j])

im1 = ax1.imshow(iMIP, cmap='gray')

####################
# Task 1b 
####################

ax2 = fig.add_subplot(1, 3, 2)
iXRAY = np.zeros((256, 256))

for i in range(256):
    for j in range(256):
        tau = np.sum(image[i][j])/len(image[i][j])
        iXRAY[i][j] = 1 - np.exp(-tau)

im2 = ax2.imshow(iXRAY, cmap='gray')

####################
# Task 1c 
####################
gamma = 0.0
alpha = 0.06
dimensions = np.shape(img_data)

ax3 = fig.add_subplot(1, 3, 3)
iMIDA = np.zeros((256, 256))

for i in range(256):
    for j in range(256):
        delta = 0
        C_in = [0]
        a_in = [alpha]
        intensity = np.zeros(dimensions[2])
        for k in range(dimensions[2]):
            if image[i][j][k] > np.max(intensity):
                delta = image[i][j][k] - np.max(intensity)
            else:
                delta = 0
            intensity[k] = image[i][j][k]
            beta = 1 - delta
            c_next = beta * C_in[k] + (1 - beta * a_in[k]) * image[i][j][k]
            a_next = beta * a_in[k] + (1 - beta * a_in[k]) * alpha
            C_in.append(c_next)
            a_in.append(a_next)
        iMIDA[i][j] = C_in[-1]

im3 = ax3.imshow(iMIDA, cmap='gray')

# Always run show, to make sure everything is displayed.
plt.show()

import nrrd
import numpy as np
import matplotlib.pyplot as plt

# load the image volume
img_data, header = nrrd.read('MRHead.nrrd')

# normalize image values
image = np.divide(img_data, float(np.max(img_data)))

# create a figure
fig = plt.figure(figsize=(6, 6))


# Solution of last exercise
# computes the MIDA interpolation along the orthografic z-axis
def MIDA_z(img, gamma):
    # default alpha
    alpha = 0.01

    x_dim = img.shape[0]
    y_dim = img.shape[1]
    z_dim = img.shape[2]

    # Input is the first image slice, the default alpha and 0 as maximum
    mida = img[:, :, 0] * alpha
    alpha_in = np.full((x_dim, y_dim), alpha)
    I_max = np.zeros((x_dim, y_dim))

    # Samples are in z-direction
    for k in range(1, z_dim):
        # sample intensities of one slice
        I = img[:, :, k]

        # subtract if intensity is larger than max intensity, 0 otherwise
        delta = np.zeros((x_dim, y_dim))
        ind = I > I_max
        delta[ind] = I[ind] - I_max[ind]

        # recalculate max intensities
        I_max = np.maximum(I_max, I)

        # derive beta (for MIDA->DVR)
        if gamma <= 0:
            beta = np.ones((x_dim, y_dim)) - delta * (1 + gamma)
        else:
            beta = np.ones((x_dim, y_dim)) - delta

        # add to solution
        # - alpha_in is equivalent to alpha_out if one line is used for computation
        # - the result 'mida' is C_out (and C_in) in the formula
        C = I * alpha
        mida = np.multiply(beta, mida) + np.multiply(1 - np.multiply(beta, alpha_in), C)
        alpha_in = np.multiply(beta, alpha_in) + (1 - np.multiply(beta, alpha_in)) * alpha

    # Optionally interpolate MIDA->MIP
    if gamma > 0:
        mida = mida * (1 - gamma) + I_max * gamma

    return mida


####################
# Task 1  
####################
x_dim = image.shape[0]
y_dim = image.shape[1]
z_dim = image.shape[2]

gradient_image = list()
for k in range(1, x_dim - 1):
    gx = 0.5 * (image[k + 1, 2:y_dim, 2:z_dim] - image[k - 1, 0:y_dim - 2, 0:z_dim - 2])
    gy = 0.5 * (image[k, 2:y_dim, 2:z_dim] - image[k, 0:y_dim - 2, 0:z_dim - 2])
    gz = 0.5 * (image[k, 2:y_dim, 2:z_dim] - image[k, 0:y_dim - 2, 0:z_dim - 2])

    g = np.sqrt(gx**2 + gy**2 + gz**2)
    gradient_image.append(g)

gradient_mida = MIDA_z(np.asarray(gradient_image), 0)

# plot the image
axs = fig.add_subplot()
axs.set_title('MIDA of Gradient')
axs.imshow(gradient_mida, cmap='gray')

# Always run show, to make sure everything is displayed.
plt.show()
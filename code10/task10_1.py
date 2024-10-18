import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(15, 5))

ax1 = fig.add_subplot(1,3,1, projection='3d')
ax1.set_title('Input')
ax1.view_init(elev=0, azim=-90)

ax2 = fig.add_subplot(1, 3, 2, projection='3d')
ax2.set_title('Individual Patches')
ax2.view_init(elev=0, azim=-90)

ax3 = fig.add_subplot(1,3,3, projection='3d')
ax3.set_title('Triangle Strip')
ax3.view_init(elev=0, azim=-90)

# Parameters
n = 100 # number of points
v = np.array([0, 1, 0]) # view direction
omega = 0.1 # thickness parameter

# Set up the line
theta = np.linspace(-4*np.pi, 4*np.pi, n)
z = np.linspace(-2, 2, n)
r = z**2 + 1
x = r * np.sin(theta)
y = r * np.cos(theta)

# Show the original line
ax1.plot(x,y,z)

####################
# Task 1  
####################

# a)
x_coordinates = []
y_coordinates = []
z_coordinates = []

for i in range(n - 1):
    x0 = np.array([x[i], y[i], z[i]])
    x1 = np.array([x[i + 1], y[i + 1], z[i + 1]])
    t_hat = (x1 - x0) / np.linalg.norm(x1 - x0)
    cross_product = omega * np.cross(t_hat, v)
    x_out1 = x0 + cross_product
    x_out2 = x0 - cross_product
    x_out3 = x1 + cross_product
    x_out4 = x1 - cross_product
    coordinates_matrix = np.array([x_out1, x_out2, x_out4, x_out1, x_out3, x_out4])
    x_coordinates.extend(coordinates_matrix[:, 0])
    y_coordinates.extend(coordinates_matrix[:, 1])
    z_coordinates.extend(coordinates_matrix[:, 2])

x_coordinates = np.asarray(x_coordinates)
y_coordinates = np.asarray(y_coordinates)
z_coordinates = np.asarray(z_coordinates)

t = []
for j in range(len(x_coordinates) - 2):
    t.append([j, j+1, j+2])


# Plot
ax2.plot_trisurf(x_coordinates, y_coordinates, z_coordinates, triangles=t)


# b)
x_coordinates = []
y_coordinates = []
z_coordinates = []

for i in range(n - 1):
    x0 = np.array([x[i], y[i], z[i]])
    x1 = np.array([x[i + 1], y[i + 1], z[i + 1]])
    t_hat = (x1 - x0) / np.linalg.norm(x1 - x0)
    cross_product = omega * np.cross(t_hat, v)

    if i < n - 2:
        x_out1 = x0 + cross_product
        x_out2 = x0 - cross_product
        coordinates_matrix = np.array([x_out1, x_out2])
    else:
        x_out1 = x0 + cross_product
        x_out2 = x0 - cross_product
        x_out3 = x1 + cross_product
        x_out4 = x1 - cross_product
        coordinates_matrix = np.array([x_out1, x_out2, x_out4, x_out1, x_out3, x_out4])

    x_coordinates.extend(coordinates_matrix[:, 0])
    y_coordinates.extend(coordinates_matrix[:, 1])
    z_coordinates.extend(coordinates_matrix[:, 2])

x_coordinates = np.asarray(x_coordinates)
y_coordinates = np.asarray(y_coordinates)
z_coordinates = np.asarray(z_coordinates)
print(len(x_coordinates))

# Indexmarix
t = []
for i in range(198):
    t.append([i, i + 1, i + 3])
    t.append([i, i + 2, i + 3])
ax3.plot_trisurf(x_coordinates, y_coordinates, z_coordinates, triangles=t)


plt.show()
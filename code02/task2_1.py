import numpy as np
import matplotlib.pyplot as plt

# Create a figure (window)
fig = plt.figure(figsize=(9, 4))

###############################
# Task 1a
###############################
axs1 = fig.add_subplot(1, 2, 1)
axs1.set_title('a) Taylor')

# Initialize y with zeros
x = np.arange(-4, 4, 0.05)
y0 = np.zeros(x.shape[0])
y1 = np.zeros(x.shape[0])
y2 = np.zeros(x.shape[0])
y3 = np.zeros(x.shape[0])


# plot reference function
axs1.plot(x, np.power(np.e, x), 'b--')

# TODO: plot a function for degrees 0 to 4
# Taylor series for degree 0
for i in range(1):
    y0 += np.power(x, i)/np.math.factorial(i)
axs1.plot(x, y0)

# Taylor series for degree 1
for i in range(2):
    y1 += np.power(x, i)/np.math.factorial(i)
axs1.plot(x, y1, 'orange')

# Taylor series for degree 2
for i in range(3):
    y2 += np.power(x, i)/np.math.factorial(i)
axs1.plot(x, y2, 'green')

# Taylor series for degree 3
for i in range(4):
    y3 += np.power(x, i)/np.math.factorial(i)
axs1.plot(x, y3, 'r-')

###############################
# Task 1b
###############################
axs2 = fig.add_subplot(1, 2, 2)
axs2.set_title('b) Fourier')

M = np.array([1, 2, 10, 1000])
x = np.arange(0, 2*np.pi, 0.01)

# TODO: implement 1D Fourier Transform
y0 = np.zeros(x.shape[0])
y1 = np.zeros(x.shape[0])
y2 = np.zeros(x.shape[0])
y3 = np.zeros(x.shape[0])

# Fourier Transform for m = 1
for i in range(1, 2):
    y0 += (-4 / ((2 * i - 1) * np.pi)) * np.sin((2 * i - 1) * x)
axs2.plot(x, y0)

# Fourier Transform for m = 2
for i in range(1, 3):
    y1 += (-4 / ((2 * i - 1) * np.pi)) * np.sin((2 * i - 1) * x)
axs2.plot(x, y1, 'orange')

# Fourier Transform for m = 10
for i in range(1, 11):
    y2 += (-4 / ((2 * i - 1) * np.pi)) * np.sin((2 * i - 1) * x)
axs2.plot(x, y2, 'green')

# Fourier Transform for m = 1000
for i in range(1, 1001):
    y3 += (-4 / ((2 * i - 1) * np.pi)) * np.sin((2 * i - 1) * x)
axs2.plot(x, y3, 'red')


# Always run this, to make sure everything is displayed.
plt.show()
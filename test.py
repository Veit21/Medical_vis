import matplotlib.pyplot as plt
import numpy as np
import random

# initialize array
x = np.zeros(10000)
x0 = 4
x[0] = x0

# simulation of random walk
for i in range(1, len(x)):
    if random.random() > 0.5:
        x[i] = x[i - 1] + 1
    else:
        x[i] = x[i - 1] - 1

plt.plot(x)
plt.show()

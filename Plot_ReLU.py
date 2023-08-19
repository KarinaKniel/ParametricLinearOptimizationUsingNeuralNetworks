import numpy as np
import matplotlib.pyplot as plt

def ReLU(x):
    return np.maximum(0, x)

def derivative_ReLU(x):
    return np.where(x < 0, 0, np.where(x > 0, 1, np.nan))

fig, ax = plt.subplots(figsize=(8, 8))

x = np.linspace(-7, 4, 100)

ax.plot(x, ReLU(x), label = "ReLU")
ax.plot(x, derivative_ReLU(x), label = "ReLU\'")

ax.set_xlabel("Σ")
ax.set_ylabel("Output")

ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0)

ax.set_title("ReLU activation function")

plt.tight_layout()

plt.savefig('plot.eps', format='eps')

# Show plot
plt.show()

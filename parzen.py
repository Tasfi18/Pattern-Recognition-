# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt # type: ignore

# Step 1: Generate synthetic dataset
np.random.seed(42)  # reproducibility
n = 1000
data = np.concatenate([
    np.random.normal(0, 1, int(0.6 * n)),  # 60% N(0,1)
    np.random.normal(5, 1, int(0.4 * n))   # 40% N(5,1)
])

# Step 2: Gaussian kernel
def gaussian_kernel(x, x_i, h):
    return (1 / (np.sqrt(2 * np.pi) * h)) * np.exp(-((x - x_i) ** 2) / (2 * h ** 2))

# Step 3: Parzen density estimator
def parzen_density(data, x_grid, h):
    n = len(data)
    density = np.zeros_like(x_grid, dtype=float)
    for x_i in data:
        density += gaussian_kernel(x_grid, x_i, h)
    return density / n

# Step 4: Prepare grid for density estimation
x_grid = np.linspace(min(data), max(data), 500)

# Bandwidths to test
bandwidths = [0.1, 0.25, 0.5, 1, 1.5, 2, 2.5]

# Step 5: Plotting for multiple bandwidths
plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, density=True, alpha=0.4, color="brown", label="Histogram")

for h in bandwidths:
    density = parzen_density(data, x_grid, h)
    plt.plot(x_grid, density, label=f"h = {h}")

plt.title("Parzen Density Estimation with Gaussian Kernel")
plt.xlabel("x")
plt.ylabel("Density")
plt.legend()
plt.show()

# Step 6: Interactive option - choose your own bandwidth
h = float(input("Enter a custom bandwidth: "))
custom_density = parzen_density(data, x_grid, h)

plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, density=True, alpha=0.4, color="brown", label="Histogram")
plt.plot(x_grid, custom_density, label=f"h = {h}")
plt.legend()
plt.title(f"Parzen Density Estimation (h = {h})")
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import laplace, norm

# Parameters
sigma_X = 1
sigma_Z = np.sqrt(0.1)

# Function to compute MMSE estimate for given y
def mmse_estimate(y):
    # Define the conditional PDF p(X|Y=y)
    def conditional_pdf(x):
        return laplace.pdf(x, scale=sigma_X) * norm.pdf(y, loc=x, scale=sigma_Z)
    
    # Compute the MMSE estimate by minimizing mean squared error
    x_values = np.linspace(-2, 2, 100)  # Define range of x values
    # mse_values = [(x - y)**2 * conditional_pdf(x) for x in x_values]
    # mmse_estimate = x_values[np.argmin(mse_values)]  # Find x that minimizes MSE
    # return mmse_estimate
    mse_values = sum([x * conditional_pdf(x) for x in x_values])
    den = sum([conditional_pdf(x) for x in x_values])
    return mse_values/den

# Generate range of y values
y_values = np.linspace(-2, 2, 1000)

# Compute MMSE estimates for each y
mmse_estimates = [mmse_estimate(y) for y in y_values]

# Plot MMSE estimate as a function of y
plt.plot(y_values, mmse_estimates)
plt.xlabel('Y')
plt.ylabel('MMSE Estimate of X')
plt.title('MMSE Estimation for Laplacian Source')
plt.grid(True)
plt.savefig('msme.png')

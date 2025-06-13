# ---------------------------------------------------------------------------------------
# <!--
# Name: Rajan Vijaykumar Singh
# Matriculation Number: 1567294
# Course: Computational Intelligence
# Topic: Bayesian Deep Learning and Uncertainty Estimation
#
# Description:
# This code demonstrates Bayesian inference and uncertainty estimation in a simple
# one-layer neural network using weight sampling. It compares the predictive output
# distributions for sigmoid and ReLU activations by sampling weights from a prior
# distribution. This illustrates how different activations and uncertainty in weights
# affect output uncertainty.
# -->
# ---------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# <!-- Custom sigmoid activation function -->
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ---------------------------------------------------------------------------------------
# <!-- Step 1: Define prior distribution over weights
# We assume a Gaussian prior: w ~ N(0, 1), representing our belief about the weight
# before observing any data.
# -->
# ---------------------------------------------------------------------------------------
prior_mean = 0               # Mean of the prior distribution
prior_std = 1                # Standard deviation of the prior
num_samples = 1000           # Number of weight samples to approximate the posterior
w_samples = np.random.normal(prior_mean, prior_std, size=num_samples)  # Sampling weights

# ---------------------------------------------------------------------------------------
# <!-- Step 2: Define input to the neuron
# Here, we simulate a single input x = 2.0, to observe the predictive distribution.
# -->
# ---------------------------------------------------------------------------------------
x_input = 2.0  # Input to the single-layer network

# ---------------------------------------------------------------------------------------
# <!-- Step 3a: Compute output using sigmoid activation
# For each sampled weight, we compute the network output using sigmoid(w * x).
# This captures the uncertainty in the output due to uncertainty in weights.
# -->
# ---------------------------------------------------------------------------------------
y_sigmoid_preds = sigmoid(w_samples * x_input)

# ---------------------------------------------------------------------------------------
# <!-- Step 3b: Compute output using ReLU activation
# Same as above, but using ReLU activation instead of sigmoid.
# This helps visualize how different activations affect predictive distributions.
# -->
# ---------------------------------------------------------------------------------------
def relu(z):
    return np.maximum(0, z)

y_relu_preds = relu(w_samples * x_input)

# ---------------------------------------------------------------------------------------
# <!-- Step 4: Plot the predictive output distributions for both activations
# This visualizes how uncertain predictions are, depending on the sampled weights.
# -->
# ---------------------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# <!-- Histogram for sigmoid-based predictions -->
axs[0].hist(y_sigmoid_preds, bins=50, density=True, color='skyblue', edgecolor='black')
axs[0].set_title("Predictive Distribution using Sigmoid Activation")
axs[0].set_xlabel("Predicted Output (ŷ)")
axs[0].set_ylabel("Density")
axs[0].grid(True)

# <!-- Histogram for ReLU-based predictions -->
axs[1].hist(y_relu_preds, bins=50, density=True, color='orange', edgecolor='black')
axs[1].set_title("Predictive Distribution using ReLU Activation")
axs[1].set_xlabel("Predicted Output (ŷ)")
axs[1].set_ylabel("Density")
axs[1].grid(True)

plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------------------
# <!--
# Explanation:
# - This example shows how uncertainty in weights propagates to uncertainty in predictions.
# - We simulate this uncertainty using random samples from the weight prior (Bayesian approach).
# - The histograms reveal how the choice of activation function influences the shape
#   and spread of the predictive output distribution.
# - In a full Bayesian neural network, posterior over weights would be updated using data,
#   but here we focus on prior uncertainty for simplicity and clarity.
# -->
# ---------------------------------------------------------------------------------------


import numpy as np
from scipy.linalg import inv, pinv

# Example parameters
N = 50  # Total number of securities
a = 1  # Risk aversion parameter

# Simulate random expected returns
np.random.seed(42)
R = np.random.rand(N) * 0.1  # Example expected returns

# Simulate a random positive definite covariance matrix
A = np.random.rand(N, N)
C = np.dot(A, A.T)

# Constructing G
G = np.zeros((2, N))
G[0, :] = 1  # Budget constraint
G[1, :17] = 0.1  # Sector allocation constraint

# Constructing c vector for constraints
c = np.array([1, 0.1])  # Target values for the constraints

# Calculate G C^{-1} G^T
C_inv = inv(C)
G_C_inv_G_T = np.dot(G, np.dot(C_inv, G.T))

# Solve for lambda using a stable method
lambda_vec = np.dot(pinv(G_C_inv_G_T), (np.dot(G, np.dot(C_inv, R)) - 2 * a * c))

# Calculate portfolio weights w
w = (1 / (2 * a)) * np.dot(C_inv, (R - np.dot(G.T, lambda_vec)))

# Display results
print("Portfolio Weights:")
print(w)

# Analysis of the portfolio
print("\nAnalysis:")
print(f"Total Weight (should be 1): {np.sum(w)}")
print(f"Sector Allocation (first 17 securities): {np.sum(w[:17])}")

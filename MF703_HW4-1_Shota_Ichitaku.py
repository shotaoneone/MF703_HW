
# Define sample parameters
np.random.seed(42)  # For reproducibility
n_assets = 5  # Number of assets
n_constraints = 3  # Number of constraints

# Simulated data
R = np.random.rand(n_assets)  # Expected returns vector
C = np.random.rand(n_assets, n_assets)  # Covariance matrix
C = (C + C.T) / 2  # Make covariance matrix symmetric
w = np.random.rand(n_assets)  # Portfolio weights vector
G = np.random.rand(n_constraints, n_assets)  # Constraints matrix
c = np.random.rand(n_constraints)  # Constraint values vector
a = 0.5  # Risk aversion coefficient
λ = np.random.rand(n_constraints)  # Lagrange multipliers vector

# Calculate gradient with respect to w (∇wL)
grad_w = R - 2 * a * np.dot(C, w) - np.dot(G.T, λ)

# Calculate gradient with respect to λ (∇λL)
grad_lambda = np.dot(G, w) - c

# Print results
print("Gradient with respect to w (∇wL):")
print(grad_w)

print("\nGradient with respect to λ (∇λL):")
print(grad_lambda)


# Solve for λ
GC_inv = np.dot(G, np.linalg.inv(C))
GC_inv_GT = np.dot(GC_inv, G.T)
GC_inv_R = np.dot(GC_inv, R)

lambda_values = np.linalg.solve(GC_inv_GT, GC_inv_R - 2 * a * c)

# Solve for w
w = (1 / (2 * a)) * np.dot(np.linalg.inv(C), (R - np.dot(G.T, lambda_values)))

# Display results
print("Lagrange multipliers (λ):")
print(lambda_values)

print("\nOptimal portfolio weights (w):")
print(w)




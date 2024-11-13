#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 16:52:23 2024

@author: ichitakushouta
"""

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Download Historical Data
# List of S&P 500 companies (tickers)
tickers = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "SYK", "JNJ", "JPM", "V",
    "PG", "NVDA", "HD", "UNH", "BAC", "MA", "DIS", "ADBE", "CRM", "NFLX",
    "CMCSA", "XOM", "VZ", "INTC", "PFE", "ABT", "KO", "PEP", "T", "MRK",
    "WMT", "CVX", "CSCO", "WFC", "MCD", "ACN", "ABBV", "AVGO", "TMO", "COST",
    "DHR", "BMY", "UNP", "NEE", "LLY", "PM", "ORCL", "HON", "UPS", "IBM",
    "QCOM", "AMT", "CVS", "LIN", "TXN", "SBUX", "BA", "RTX", "GS", "BLK",
    "MMM", "CAT", "AXP", "INTU", "SPGI", "GILD", "MO", "C", "MDLZ", "AMGN",
    "LOW", "DE", "ADP", "ISRG", "BKNG", "GE", "NOW", "MS", "TGT", "AMD",
    "ZTS", "ADI", "BDX", "DUK", "SO", "SCHW", "PLD", "CB", "CME", "PNC",
    "USB", "CCI", "CL", "TJX", "TMUS", "D", "CSX", "CI", "EQIX", "ICE"
]
data = yf.download(tickers, start="2019-11-12", end="2024-11-12")['Adj Close']

# Step 2: Clean the Data
# Fill missing values using forward fill
data = data.fillna(method='ffill')

# Step 3: Generate Daily Returns
returns = data.pct_change().dropna()

# Step 4: Calculate Covariance Matrix
cov_matrix = returns.cov()

# Step 5: Eigenvalue Decomposition
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Count positive and negative eigenvalues
num_positive_eigenvalues = np.sum(eigenvalues > 0)
num_negative_eigenvalues = np.sum(eigenvalues < 0)

# Check for negative eigenvalues
if num_negative_eigenvalues > 0:
    print(f"There are {num_negative_eigenvalues} negative eigenvalues. This should not happen theoretically.")
else:
    print("There are no negative eigenvalues, as expected.")

# Step 6: Variance Explanation
total_variance = np.sum(eigenvalues)
explained_variance = [(i / total_variance) for i in sorted(eigenvalues, reverse=True)]
cumulative_variance = np.cumsum(explained_variance)

# Determine the number of eigenvalues needed to explain 50% and 90% of the variance
num_eigenvalues_50 = np.searchsorted(cumulative_variance, 0.50) + 1
num_eigenvalues_90 = np.searchsorted(cumulative_variance, 0.90) + 1

print(f"Number of eigenvalues needed to explain 50% of the variance: {num_eigenvalues_50}")
print(f"Number of eigenvalues needed to explain 90% of the variance: {num_eigenvalues_90}")

# Plot the cumulative variance explained
plt.figure(figsize=(10, 6))
plt.plot(cumulative_variance, marker='o')
plt.xlabel('Number of Eigenvalues')
plt.ylabel('Cumulative Variance Explained')
plt.title('Cumulative Variance Explained by Eigenvalues')
plt.grid(True)
plt.show()

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




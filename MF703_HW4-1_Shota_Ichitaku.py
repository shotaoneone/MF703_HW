import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#1
#Download Historical Data
#Pick 100 companies from S&P500
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

#Clean the Data
#Fill in missing values using forward fill
data = data.fillna(method='ffill')

#2
#Generate Daily Returns
returns = data.pct_change().dropna()

#3
#Calculate Covariance Matrix
cov_matrix = returns.cov()

#Eigenvalue Decomposition
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

#Count positive and negative eigenvalues
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

#Plot the cumulative variance explained
plt.figure(figsize=(10, 6))
plt.plot(cumulative_variance, marker='o')
plt.xlabel('Number of Eigenvalues')
plt.ylabel('Cumulative Variance Explained')
plt.title('Cumulative Variance Explained by Eigenvalues')
plt.grid(True)
plt.show()




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

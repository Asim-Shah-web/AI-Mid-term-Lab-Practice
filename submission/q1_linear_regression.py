# %% [markdown]
# # Q1: Multiple Linear Regression (from scratch)
# Hypothesis h(x) = theta^T x, cost J = 1/(2m) [ sum(err^2) + lambda * sum_{j>=1} theta_j^2 ]
# Gradients: j=0: (1/m) sum(err * x0);  j>=1: (1/m) sum(err * xj) + (lambda/m) theta_j
# Update: theta_j := theta_j - alpha * dJ/dtheta_j  (vector form: theta -= alpha * grad)

# %% Imports
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# %% Load data (last column = target)
df = pd.read_csv("../ASSIGNMENT01/multiple_linear_regression_dataset.csv")
X, y = df.iloc[:, :-1].values.astype(float), df.iloc[:, -1].values.astype(float)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# %% Feature scaling: standardization (fit on train only)
mu, sigma = X_tr.mean(0), X_tr.std(0)
add_bias = lambda A: np.hstack([np.ones((len(A), 1)), A])
Xtr, Xte = add_bias((X_tr - mu) / sigma), add_bias((X_te - mu) / sigma)

# %% Hypothesis, regularized cost, gradient descent
def hypothesis(X, theta): return X @ theta

def cost(X, y, theta, lam):
    m = len(y)
    return (np.sum((hypothesis(X, theta) - y) ** 2) + lam * np.sum(theta[1:] ** 2)) / (2 * m)

def gradient_descent(X, y, alpha, lam, iters):
    m, theta, hist = len(y), np.zeros(X.shape[1]), []
    for _ in range(iters):
        grad = X.T @ (hypothesis(X, theta) - y) / m
        grad[1:] += (lam / m) * theta[1:]          # no regularization on theta_0
        theta -= alpha * grad
        hist.append(cost(X, y, theta, lam))
    return theta, hist

# %% Train: alpha=0.1 (converges fast on standardized data), lambda=0.1
alpha, lam, iters = 0.1, 0.1, 500
theta, hist = gradient_descent(Xtr, y_tr, alpha, lam, iters)
print("Final theta:", theta, "\nFinal cost:", hist[-1])

# %% Convergence curve
plt.plot(hist); plt.xlabel("Iterations"); plt.ylabel("J(theta)"); plt.title("Convergence"); plt.grid()
plt.savefig("results/q1_convergence.png", dpi=120); plt.show()

# %% Compare with scikit-learn (MSE on test set)
sk = LinearRegression().fit(Xtr[:, 1:], y_tr)
mse_ours = mean_squared_error(y_te, hypothesis(Xte, theta))
mse_sk = mean_squared_error(y_te, sk.predict(Xte[:, 1:]))
print(f"MSE ours: {mse_ours:.1f} | MSE sklearn: {mse_sk:.1f}")
print("sklearn intercept/coef:", sk.intercept_, sk.coef_)

# %% Effect of learning rate and lambda (for the discussion)
for a in [0.001, 0.01, 0.1, 0.5]:
    plt.plot(gradient_descent(Xtr, y_tr, a, lam, 200)[1], label=f"alpha={a}")
plt.yscale("log"); plt.legend(); plt.title("Learning rate effect"); plt.show()
for l in [0, 0.1, 1, 10, 100]:
    t, _ = gradient_descent(Xtr, y_tr, alpha, l, iters)
    print(f"lambda={l:>5}: test MSE={mean_squared_error(y_te, hypothesis(Xte, t)):.1f}")

# %% [markdown]
# **Discussion (fill in from YOUR printed results):** small alpha converges slowly, too large diverges;
# standardization lets one alpha work for all features; larger lambda shrinks theta (can under-fit),
# lambda=0 matches sklearn. Dataset is tiny (20 rows), so the test MSE is noisy.

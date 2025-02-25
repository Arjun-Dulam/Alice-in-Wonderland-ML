import numpy as np

matrix = np.random.rand(10, 2)
stepSize = 0.30

# Define the function
def f(x1, x2):
    return np.sin(x1) * np.cos(x2) + np.sin(0.5 * x1) * np.cos(0.5 * x2)

# Partial derivatives
def wrtX2(x1, x2):
    return (0.5 * np.sin(0.5 * x1) * np.sin(0.5 * x2) - np.sin(x1) * np.sin(x2))

def wrtX1(x1, x2):
    return 0.5 * np.cos(0.5 * x1) * np.cos(0.5 * x2) + np.cos(x1) * np.cos(x2)

# Compute gradient
def gradient(x1, x2):
    x1_prime_col, x2_prime_col = wrtX1(x1, x2), wrtX2(x1, x2)

    return np.array([x1_prime_col, x2_prime_col])  # Ensure shape (2,)

# Gradient Descent function
def gradient_descent(threshold=1e-6, max_iters=1000):
    curr_point = np.random.rand(2)  # Start from a random point
    iteration = 0

    while np.linalg.norm(gradient(curr_point[0], curr_point[1])) > threshold and iteration < max_iters:
        grad = gradient(curr_point[0], curr_point[1])
        curr_point = curr_point - stepSize * grad  # Ensure proper shape
        iteration += 1

    print(f"Number of iterations: {iteration}")
    return curr_point, f(curr_point[0], curr_point[1])

# Run Gradient Descent
opt_point, opt_value = gradient_descent()
print(f"Optimal point: {opt_point}, Function value: {opt_value}")

# Extract columns from matrix
x1_col = matrix[:, 0]
x2_col = matrix[:, 1]

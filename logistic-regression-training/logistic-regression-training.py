import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    X = np.asarray(X, float)
    y = np.asarray(y, float)

    n_samples, n_features = X.shape

    w = np.zeros(n_features)
    b = 0.0
    
    # print(w)
    for _ in range(steps):
        # Liner prediction
        z = X @ w + b

        #using sigmoid activation
        p = _sigmoid(z)

        # gradient wrt w and b
        error = p - y
        grad_w = X.T @ error / n_samples
        grad_b = np.mean(error)

        # gradient descent updates
        w -= lr * grad_w
        b -= lr * grad_b

    return w, b
    pass
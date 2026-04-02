import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # Write code here
    x = np.asarray(x, dtype=float)        # convert to ndarray of floats
    # numerically stable sigmoid
    return 1 / (1 + np.exp(-x))
    pass
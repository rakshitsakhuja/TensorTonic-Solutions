import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    Works with scalars, NumPy arrays, and tensors convertible to np.array.
    """
    x=np.asarray(x, dtype=float)
    return 1 / (1 + np.exp(-x))
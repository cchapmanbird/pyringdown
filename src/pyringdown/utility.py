import numpy as np

def ensure_arrays(*args):
    out = []
    for arg in args:
        out.append(np.atleast_1d(arg))
    return out


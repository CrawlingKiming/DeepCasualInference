import numpy as np

def npmse(A, B, ax = 0):
    return (np.square(A - B)).mean(axis=ax)
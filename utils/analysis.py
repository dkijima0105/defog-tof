import numpy as np

def normalize(v, axis=-1, order=2):
        l2 = np.linalg.norm(v, ord = order, axis=axis, keepdims=True)
        l2[l2==0] = 1
        
        return v/l2
        
def standardization(l, mean, std):
    return [(i - mean) / std for i in l]

def min_max(x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min)
    
    return result

def value2index(value, max, min, size):
    return int(np.round(value / (max - min) * size))

    
def index2value(index, max, min, size):
    return index * (max - min) / size + min
    
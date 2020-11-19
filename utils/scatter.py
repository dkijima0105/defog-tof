import numpy as np

def phase_function(g, phase_angle):
    assert (g >= -1.0) and (g <=1.0), 'g = [-1.0, 1.0]'
    return 1 / (4 * np.pi) * (1 - g**2) / (1 + g**2 - 2 * g * np.cos(phase_angle))**(3/2)



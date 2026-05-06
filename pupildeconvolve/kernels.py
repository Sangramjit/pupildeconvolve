import numpy as np

def h_pupil(t, n=10.1, t_max=930):
    h = (t ** n) * np.exp(-n * t / t_max)
    h[0] = 0
    return h

def h_plr(t, scale=-0.6):
    return scale * h_pupil(t, n=6)

def normalize_kernel(h):
    s = np.sum(np.abs(h))
    return h if s == 0 else h / s
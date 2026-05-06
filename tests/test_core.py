import numpy as np
from pupildeconvolve import deconvolve_dual

def test_runs():
    pupil = np.random.randn(200)
    res = deconvolve_dual(pupil, sampling_rate=100)
    assert res is not None
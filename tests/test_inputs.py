import pytest
from pupildeconvolve import deconvolve_dual

def test_list_input_error():
    with pytest.raises(ValueError):
        deconvolve_dual([1,2,3], sampling_rate=100)
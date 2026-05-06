import numpy as np

def validate_pupil_input(pupil):

    if isinstance(pupil, list):
        raise ValueError(
            "Input is a list. Convert to numpy array using np.array()."
        )

    if not isinstance(pupil, np.ndarray):
        raise ValueError("Input must be a numpy array.")

    if pupil.ndim != 1:
        raise ValueError(
            f"Pupil must be 1D (one trial). Got shape {pupil.shape}"
        )

    if np.any(np.isnan(pupil)):
        raise ValueError("Pupil contains NaNs. Clean data first.")
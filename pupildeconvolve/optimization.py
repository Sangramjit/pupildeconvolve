import numpy as np
from scipy.signal import fftconvolve

def build_input(length, pulse_locs, strengths):
    x = np.zeros(length)
    for loc, s in zip(pulse_locs, strengths):
        if 0 <= loc < length:
            x[int(loc)] = s
    return x

def convolve(x, h):
    return fftconvolve(x, h, mode="full")[:len(x)]

def pupil_model(pars, pupil_len, plr_locs, attn_locs, h_plr_k, h_attn_k):
    n_plr = len(plr_locs)

    plr_w = pars[:n_plr]
    attn_w = pars[n_plr:-1]
    slope = pars[-1]

    i_plr = build_input(pupil_len, plr_locs, plr_w)
    i_attn = build_input(pupil_len, attn_locs, attn_w)

    y = (
        convolve(i_plr, h_plr_k)
        + convolve(i_attn, h_attn_k)
        + slope * np.arange(pupil_len)
    )
    return y

def objective(pars, pupil, plr_locs, attn_locs, h_plr_k, h_attn_k):
    pred = pupil_model(
        pars,
        len(pupil),
        plr_locs,
        attn_locs,
        h_plr_k,
        h_attn_k
    )
    return np.nanmean((pred - pupil) ** 2)
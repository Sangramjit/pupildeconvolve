from .utils import validate_pupil_input
import numpy as np
from scipy.optimize import minimize

from .kernels import h_pupil, h_plr, normalize_kernel
from .optimization import objective


# =====================================================
# ----------- PULSE LOCATION HANDLING ------------------
# =====================================================
def get_pulse_indices(time, pulse_times=None, pulse_interval=None):

    if pulse_times is not None and pulse_interval is not None:
        raise ValueError("Provide either pulse_times OR pulse_interval, not both.")

    if pulse_times is not None:
        pulse_times = np.asarray(pulse_times)
    else:
        pulse_times = np.arange(0, time[-1] + 1, pulse_interval)

    pulse_idx = np.searchsorted(time, pulse_times)
    pulse_idx = pulse_idx[pulse_idx < len(time)]

    return pulse_idx, pulse_times


# =====================================================
# ----------- PLR LATENCY (SLOPE BASED) ---------------
# =====================================================
def estimate_plr_latency(time, pupil, window=(0, 400), smooth_ms=50):

    time = np.asarray(time)
    pupil = np.asarray(pupil)

    dt = np.nanmedian(np.diff(time))

    # ---- smoothing ----
    win = int(smooth_ms / dt)
    if win < 3:
        win = 3

    kernel = np.ones(win) / win
    pupil_smooth = np.convolve(pupil, kernel, mode="same")

    # ---- derivative ----
    dp = np.gradient(pupil_smooth, dt)

    # ---- restrict window ----
    mask = (time >= window[0]) & (time <= window[1])

    if not np.any(mask):
        return 100.0  # fallback

    dp_window = dp[mask]
    time_window = time[mask]

    idx = np.argmin(dp_window)
    return time_window[idx]


# =====================================================
# ---------------- MAIN FUNCTION -----------------------
# =====================================================
def deconvolve_dual(
    pupil,
    time=None,
    sampling_rate=None,
    pulse_times=None,
    pulse_interval=150,
    n_runs=50,
    random_state=None,
    verbose=True
):

    # ----------------------------
    # INPUT VALIDATION
    # ----------------------------

    validate_pupil_input(pupil)
    pupil = np.asarray(pupil, dtype=float)

#  NaN safety check 
    if np.any(np.isnan(pupil)):
        raise ValueError(
            "NaNs detected in pupil input. "
            "Please preprocess data or use fit_dataframe(nan_policy=...)"
        )

    if random_state is not None:
        np.random.seed(random_state)

    # ----------------------------
    # TIME HANDLING
    # ----------------------------
    if time is None:
        if sampling_rate is None:
            raise ValueError("Provide either time or sampling_rate")

        dt = 1000 / sampling_rate
        time = np.arange(len(pupil)) * dt
    else:
        time = np.asarray(time)

        if len(time) != len(pupil):
            raise ValueError(
                f"time and pupil must have same length. Got {len(time)} vs {len(pupil)}"
            )

        dt = np.nanmedian(np.diff(time))

    # ----------------------------
    # BASIC SIGNAL CHECKS
    # ----------------------------
    pupil_std = np.nanstd(pupil)

    if pupil_std < 1e-6:
        raise ValueError("Pupil signal variance too small")

    # ----------------------------
    # ATTENTION PULSES
    # ----------------------------
    attn_locs, attn_times = get_pulse_indices(
        time,
        pulse_times=pulse_times,
        pulse_interval=pulse_interval
    )

    if len(attn_locs) < 2:
        raise ValueError("Too few attention pulses defined")

    # ----------------------------
    # PLR LATENCY (AUTO)
    # ----------------------------
    plr_latency_ms = estimate_plr_latency(
        time,
        pupil,
        window=(0, 400),
        smooth_ms=50
    )

    plr_locs = np.searchsorted(time, [plr_latency_ms])
    plr_locs = plr_locs[plr_locs < len(pupil)]

    # ----------------------------
    # KERNELS
    # ----------------------------
    kernel_time = np.arange(len(pupil)) * dt

    h_plr_k = normalize_kernel(h_plr(kernel_time))
    h_attn_k = normalize_kernel(h_pupil(kernel_time))

    # ----------------------------
    # OPTIMIZATION
    # ----------------------------
    attn_runs, plr_runs, slope_runs = [], [], []

    for _ in range(n_runs):

        x0 = np.concatenate([
            np.random.uniform(0.05, 0.3, len(plr_locs)),
            np.random.uniform(0.05, 0.3, len(attn_locs)),
            [0.0]
        ])

        bounds = (
            [(0, pupil_std * 2)] * (len(plr_locs) + len(attn_locs)) +
            [(-pupil_std, pupil_std)]
        )

        res = minimize(
            objective,
            x0,
            args=(pupil, plr_locs, attn_locs, h_plr_k, h_attn_k),
            method="L-BFGS-B",
            bounds=bounds
        )

        if res.success:
            plr_runs.append(res.x[:len(plr_locs)])
            attn_runs.append(res.x[len(plr_locs):-1])
            slope_runs.append(res.x[-1])

    if len(attn_runs) == 0:
        raise RuntimeError("All optimization runs failed")

    # ----------------------------
    # AVERAGE RESULTS
    # ----------------------------
    plr_mean = np.mean(plr_runs, axis=0)
    attn_mean = np.mean(attn_runs, axis=0)
    slope_mean = np.mean(slope_runs)

    # ----------------------------
    # OUTPUT
    # ----------------------------
    return {
        "attention_amplitude": attn_mean,
        "attention_time": attn_locs,
        "attention_time_ms": attn_times,
        "plr_amplitude": plr_mean,
        "plr_time": plr_locs,
        "plr_latency_ms": plr_latency_ms,
        "slope": slope_mean
    }
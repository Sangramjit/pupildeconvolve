import matplotlib.pyplot as plt
import numpy as np


# =====================================================
# MEAN + CI
# =====================================================
def compute_mean_ci(data):

    data = np.asarray(
        data,
        dtype=float
    )

    # -------------------------------------------------
    # single signal
    # -------------------------------------------------
    if data.ndim == 1:

        return data, None

    # -------------------------------------------------
    # multiple participants
    # -------------------------------------------------
    mean = np.nanmean(
        data,
        axis=0
    )

    sem = (
        np.nanstd(data, axis=0)
        / np.sqrt(data.shape[0])
    )

    ci = 1.96 * sem

    return mean, ci


# =====================================================
# MAIN PLOTTING FUNCTION
# =====================================================
def plot_pupil_and_pulses(

    pupil,

    result,

    time,

    title="Grand Average Pupil + Attention",

    show_individual=False
):

    """
    ====================================================

    INPUTS
    ----------------------------------------------------

    pupil:
        shape:
        (participants x time)

    result["attention_amplitude"]:
        shape:
        (participants x pulses)

    result["attention_time"]:
        pulse indices in time axis

    ====================================================
    """

    # =================================================
    # INPUTS
    # =================================================
    pupil = np.asarray(
        pupil,
        dtype=float
    )

    attn_amp = np.asarray(
        result["attention_amplitude"],
        dtype=float
    )

    # -------------------------------------------------
    # pulse locations required
    # -------------------------------------------------
    if "attention_time" not in result:

        raise ValueError(
            "result must contain "
            "'attention_time'"
        )

    attn_idx = np.asarray(
        result["attention_time"]
    )

    # =================================================
    # LIMIT TO 2000 ms
    # =================================================
    max_time_ms = 2000

    time_mask = (
        time <= max_time_ms
    )

    time_plot = time[
        time_mask
    ]

    # -------------------------------------------------
    # pupil truncated
    # -------------------------------------------------
    if pupil.ndim == 1:

        pupil_plot = pupil[
            time_mask
        ]

    else:

        pupil_plot = pupil[
            :,
            time_mask
        ]

    # =================================================
    # PULSE TIMES
    # =================================================
    pulse_times = time[
        attn_idx
    ]

    pulse_mask = (
        pulse_times <= max_time_ms
    )

    pulse_times = pulse_times[
        pulse_mask
    ]

    # -------------------------------------------------
    # attention truncated
    # -------------------------------------------------
    if attn_amp.ndim == 1:

        attn_amp = attn_amp[
            pulse_mask
        ]

    else:

        attn_amp = attn_amp[
            :,
            pulse_mask
        ]

    # =================================================
    # STATS
    # =================================================
    pupil_mean, pupil_ci = compute_mean_ci(
        pupil_plot
    )

    attn_mean, attn_ci = compute_mean_ci(
        attn_amp
    )

    # =================================================
    # FIGURE
    # =================================================
    fig, axes = plt.subplots(

        2,

        1,

        figsize=(12, 10),

        sharex=False
    )

    # =================================================
    # PUPIL
    # =================================================
    ax = axes[0]

    # -------------------------------------------------
    # individual participants
    # -------------------------------------------------
    if show_individual and pupil_plot.ndim > 1:

        for tr in pupil_plot:

            ax.plot(

                time_plot,

                tr,

                alpha=0.15,

                linewidth=1
            )

    # -------------------------------------------------
    # grand mean
    # -------------------------------------------------
    ax.plot(

        time_plot,

        pupil_mean,

        linewidth=2.5
    )

    # -------------------------------------------------
    # CI
    # -------------------------------------------------
    if pupil_ci is not None:

        ax.fill_between(

            time_plot,

            pupil_mean - pupil_ci,

            pupil_mean + pupil_ci,

            alpha=0.3
        )

    ax.set_ylabel(
        "Pupil size"
    )

    ax.set_title(
        title
    )

    ax.set_xlim(
        0,
        max_time_ms
    )

    # =================================================
    # ATTENTION PULSES
    # =================================================
    ax2 = axes[1]

    # -------------------------------------------------
    # individual participants
    # -------------------------------------------------
    if show_individual and attn_amp.ndim > 1:

        for tr in attn_amp:

            ax2.plot(

                pulse_times,

                tr,

                alpha=0.15,

                linewidth=1.5,

                color="orange"
            )

    # -------------------------------------------------
    # grand mean line
    # -------------------------------------------------
    ax2.plot(

        pulse_times,

        attn_mean,

        linewidth=2.5,

        color="orange"
    )

    # -------------------------------------------------
    # pulse markers
    # -------------------------------------------------
    ax2.scatter(

        pulse_times,

        attn_mean,

        s=50,

        color="orange"
    )

    # -------------------------------------------------
    # confidence interval
    # -------------------------------------------------
    if attn_ci is not None:

        ax2.fill_between(

            pulse_times,

            attn_mean - attn_ci,

            attn_mean + attn_ci,

            color="orange",

            alpha=0.25
        )

    ax2.set_ylabel(
        "Attention pulse"
    )

    ax2.set_xlabel(
        "Time (ms)"
    )

    ax2.set_xlim(
        0,
        max_time_ms
    )

    # =================================================
    # FINALIZE
    # =================================================
    plt.tight_layout()

    plt.show()
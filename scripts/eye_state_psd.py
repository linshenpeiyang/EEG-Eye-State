#!/usr/bin/env python3
"""
Group the data by eye state and plot the eyes-open and eyes-closed PSD comparison.

eye_state values, from the UCI dataset description:
    0 = eyes open
    1 = eyes closed

Workflow:
    1. Read the ARFF file and rename the label column to eye_state;
    2. Group by eye_state and remove spike artifacts per channel, |x - median| > threshold;
    3. Estimate the PSD of each channel with the Welch method, averaged over all channels by default;
    4. Plot a semilog comparison figure with band shading.

Usage:
    python scripts/eye_state_psd.py                 # average over all channels
    python scripts/eye_state_psd.py O1              # single channel
    python scripts/eye_state_psd.py O1 50           # channel and artifact threshold
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import arff
from scipy.signal import welch

# ---- Constants ----
DEFAULT_FS = 128.0
ARTIFACT_THRESHOLD = 100.0  # Artifact threshold relative to each channel median

STATE_LABELS = {0: "Eyes open", 1: "Eyes closed"}
STATE_COLORS = {0: "#0072B2", 1: "#D55E00"}

BANDS = {
    "Delta": (0.5, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta": (13.0, 30.0),
    "Gamma": (30.0, 45.0),
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARFF_PATH = PROJECT_ROOT / "data" / "EEG Eye State.arff"
OUTPUT_IMAGE = PROJECT_ROOT / "results" / "eeg_eye_state_psd.png"

# Publication style: Times New Roman for Latin characters
plt.rcParams["font.family"] = [
    "Times New Roman",
    "Songti SC",
    "Arial Unicode MS",
    "DejaVu Serif",
]
plt.rcParams["axes.unicode_minus"] = False


def load_arff(arff_path):
    """Read the ARFF file, convert it to a DataFrame, and rename the label column to eye_state."""
    with arff_path.open("r", encoding="utf-8") as fh:
        data, _ = arff.loadarff(fh)

    df = pd.DataFrame(data)
    for column in df.columns:
        if df[column].dtype == object:
            df[column] = df[column].str.decode("utf-8")
    df = df.rename(columns={"eyeDetection": "eye_state"})
    df["eye_state"] = df["eye_state"].astype(int)
    return df


def group_psd(
    df,
    channel=None,
    state_column="eye_state",
    fs=DEFAULT_FS,
    reject=True,
    threshold=ARTIFACT_THRESHOLD,
    nperseg=None,
):
    """Compute the PSD for each eye-state group; return {state: {"freqs": ..., "psd": ..., "n": ...}}.

    channel=None averages the PSD over all EEG channels; otherwise only that channel is used.
    """
    eeg_channels = [c for c in df.columns if c != state_column]
    if channel is not None:
        if channel not in eeg_channels:
            available = ", ".join(eeg_channels)
            raise ValueError(f"Channel {channel!r} does not exist; available channels: {available}")
        channels = [channel]
    else:
        channels = eeg_channels

    if nperseg is None:
        nperseg = int(2 * fs)  # 2-second window, 0.5 Hz frequency resolution

    result = {}
    for state, group in df.groupby(state_column):
        psd_list = []
        freqs_list = []
        kept = []
        for ch in channels:
            x = group[ch].to_numpy(dtype=float)
            if reject:
                median = np.median(x)
                keep = np.abs(x - median) <= threshold
                kept.append(int(keep.sum()))
                x = x[keep]
            else:
                kept.append(len(x))

            nseg = min(nperseg, len(x))
            freqs, psd = welch(x, fs=fs, nperseg=nseg, detrend="constant")
            psd_list.append(psd)
            freqs_list.append(freqs)

        result[state] = {
            "freqs": freqs_list[0],
            "psd": np.mean(psd_list, axis=0),
            "n": int(np.mean(kept)),
        }

    return result


def plot_psd_comparison(result, channel_label, output_path=None):
    """Plot the eyes-open versus eyes-closed PSD comparison."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for state in sorted(result):
        info = result[state]
        label = f"{STATE_LABELS[state]} (n={info['n']:,})"
        linestyle = "-" if state == 0 else "--"
        ax.semilogy(
            info["freqs"],
            info["psd"],
            color=STATE_COLORS[state],
            linestyle=linestyle,
            linewidth=1.8,
            label=label,
        )

    # Band shading and labels
    for name, (low, high) in BANDS.items():
        ax.axvspan(low, high, color="gray", alpha=0.07, linewidth=0)
        ax.text(
            (low + high) / 2,
            0.985,
            name,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9,
            color="0.35",
        )

    ax.set_title(
        f"EEG power spectral density: eyes open versus eyes closed ({channel_label})",
        fontsize=14,
        pad=14,
    )
    ax.set_xlabel("Frequency (Hz)", fontsize=12)
    ax.set_ylabel("Power spectral density (a.u.²/Hz)", fontsize=12)
    ax.set_xlim(0.5, 45.0)
    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.tick_params(labelsize=11, direction="in")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=11)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Comparison plot saved: {output_path}")
    if plt.get_backend().lower() != "agg":
        plt.show()
    return fig


def main():
    channel = sys.argv[1] if len(sys.argv) > 1 else None
    if channel and channel.lower() in {"all", "average", "avg"}:
        channel = None
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else ARTIFACT_THRESHOLD

    if not ARFF_PATH.exists():
        raise FileNotFoundError(
            f"EEG data not found: {ARFF_PATH}\n"
            "Run scripts/download_data.py first to download the dataset"
        )

    df = load_arff(ARFF_PATH)
    result = group_psd(df, channel=channel, threshold=threshold)

    channel_label = channel if channel else "All-channel average"
    print(f"Channel: {channel_label}, sampling rate: {DEFAULT_FS:.0f} Hz")
    for state in sorted(result):
        info = result[state]
        print(
            f"  {STATE_LABELS[state]}: {int(df['eye_state'].eq(state).sum()):,} raw samples, "
            f"{info['n']:,} after artifact rejection (threshold ±{threshold:g})"
        )

    plot_psd_comparison(result, channel_label, OUTPUT_IMAGE)


if __name__ == "__main__":
    main()

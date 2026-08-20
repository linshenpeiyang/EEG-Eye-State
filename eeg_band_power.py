#!/usr/bin/env python3
"""
Compute the power spectral density (PSD) of an EEG signal and the power ratios of five frequency bands, and plot a pie chart.

Band definitions, standard EEG segments that can be overridden:
    Delta: 0.5-4 Hz
    Theta: 4-8 Hz
    Alpha: 8-13 Hz
    Beta:  13-30 Hz
    Gamma: 30-45 Hz

Usage:
    # Run the demo and generate a pie chart
    python eeg_band_power.py [channel]   # defaults to AF3

    # Reuse as a module
    from eeg_band_power import compute_psd, compute_band_power
    freqs, psd = compute_psd(x, fs=128.0)
    band_power, ratios = compute_band_power(freqs, psd)
"""

from pathlib import Path
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import arff
from scipy.signal import welch

try:  # numpy >= 2.0 provides trapezoid; older versions provide trapz
    from numpy import trapezoid as _trapz
except ImportError:
    from numpy import trapz as _trapz

# ---- Constants ----
DEFAULT_FS = 128.0
DEFAULT_BANDS = {
    "Delta": (0.5, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta": (13.0, 30.0),
    "Gamma": (30.0, 45.0),
}

BAND_COLORS = {
    "Delta": "#4C72B0",
    "Theta": "#55A868",
    "Alpha": "#C44E52",
    "Beta": "#8172B3",
    "Gamma": "#CCB974",
}

BASE_DIR = Path(__file__).resolve().parent
ARFF_PATH = BASE_DIR / "uci_eeg_eye_state_data" / "EEG Eye State.arff"
OUTPUT_IMAGE = BASE_DIR / "eeg_band_power_pie.png"

# Font settings for matplotlib on macOS
plt.rcParams["font.sans-serif"] = [
    "Arial Unicode MS",
    "PingFang SC",
    "Hiragino Sans GB",
    "Heiti SC",
    "STHeiti",
    "SimHei",
    "sans-serif",
]
plt.rcParams["axes.unicode_minus"] = False


def compute_psd(x, fs=DEFAULT_FS, nperseg=None, window="hann", detrend="constant"):
    """Estimate the one-sided PSD with the Welch method; return (freqs, psd).

    The unit of psd is input-unit squared per Hz; integrating over frequency
    yields power in input units squared, depending on the physical unit of x.

    Parameters
    ----------
    x : array-like
        One-dimensional EEG signal.
    fs : float
        Sampling rate in Hz, default 128.
    nperseg : int or None
        Samples per segment; defaults to a 2-second window, giving 0.5 Hz resolution.
    window, detrend
        Passed directly to scipy.signal.welch.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("compute_psd supports one-dimensional signals only")

    if nperseg is None:
        nperseg = min(int(2 * fs), x.size)
    if nperseg > x.size:
        raise ValueError(f"nperseg={nperseg} exceeds the signal length {x.size}")

    freqs, psd = welch(
        x, fs=fs, window=window, nperseg=nperseg, detrend=detrend
    )
    return freqs, psd


def compute_band_power(freqs, psd, bands=None):
    """Integrate the PSD over each band; return absolute band powers and normalized ratios.

    Ratios are normalized by the total power of the five bands so they sum to 100%.
    """
    if bands is None:
        bands = DEFAULT_BANDS

    band_power = {}
    total = 0.0
    for name, (low, high) in bands.items():
        if low >= freqs[-1]:
            warnings.warn(
                f"{name} band ({low}-{high} Hz) lies entirely outside "
                f"the measured range, up to {freqs[-1]:.1f} Hz; set to 0"
            )
            band_power[name] = 0.0
            continue

        high = min(high, freqs[-1])
        mask = (freqs >= low) & (freqs <= high)
        if mask.sum() >= 2:
            power = float(_trapz(psd[mask], freqs[mask]))
        else:
            power = 0.0
        band_power[name] = power
        total += power

    if total <= 0:
        raise ValueError("Total power of all bands is zero, cannot compute ratios")

    ratios = {name: power / total for name, power in band_power.items()}
    return band_power, ratios


def plot_band_pie(ratios, bands=None, channel=None, output_path=None):
    """Plot band power ratios as a pie chart."""
    if bands is None:
        bands = DEFAULT_BANDS

    labels = list(ratios)
    sizes = [max(ratios[name], 0.0) for name in labels]
    colors = [BAND_COLORS.get(name, None) for name in labels]

    fig, ax = plt.subplots(figsize=(7.5, 6))
    wedges, _, autotexts = ax.pie(
        sizes,
        colors=colors,
        autopct=lambda pct: f"{pct:.1f}%",
        startangle=90,
        counterclock=False,
        pctdistance=0.72,
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
    )
    for text in autotexts:
        text.set_color("white")
        text.set_fontsize(11)
        text.set_fontweight("bold")

    ax.set_title(
        f"Band power ratios for channel {channel}" if channel else "Band power ratios",
        fontsize=14,
    )
    legend_labels = [
        f"{name} ({bands[name][0]:g}-{bands[name][1]:g} Hz): "
        f"{ratios[name] * 100:.1f}%"
        for name in labels
    ]
    ax.legend(
        wedges,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        title="Band",
    )
    ax.axis("equal")

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Pie chart saved: {output_path}")
    if plt.get_backend().lower() != "agg":
        plt.show()
    return fig


def load_arff(arff_path):
    """Read an ARFF file and convert it to a DataFrame, consistent with the download script."""
    with arff_path.open("r", encoding="utf-8") as fh:
        data, _ = arff.loadarff(fh)

    df = pd.DataFrame(data)
    for column in df.columns:
        if df[column].dtype == object:
            df[column] = df[column].str.decode("utf-8")
    if "eyeDetection" in df.columns:
        df["eyeDetection"] = df["eyeDetection"].astype(int)
    return df


def main():
    channel = sys.argv[1] if len(sys.argv) > 1 else "AF3"
    if not ARFF_PATH.exists():
        raise FileNotFoundError(
            f"EEG data not found: {ARFF_PATH}\n"
            "Run uci_eeg_eye_state.py first to download the dataset"
        )

    df = load_arff(ARFF_PATH)
    if channel not in df.columns:
        available = ", ".join(df.columns)
        raise ValueError(f"Channel {channel!r} does not exist; available channels: {available}")

    raw = df[channel].to_numpy(dtype=float)
    freqs, psd = compute_psd(raw, fs=DEFAULT_FS)
    band_power, ratios = compute_band_power(freqs, psd)

    resolution = freqs[1] - freqs[0]
    print(f"Channel: {channel}, samples: {len(raw)}, sampling rate: {DEFAULT_FS:.0f} Hz")
    print(f"PSD: Welch method, frequency resolution {resolution:.2f} Hz\n")

    print("Band power statistics:")
    for name in DEFAULT_BANDS:
        low, high = DEFAULT_BANDS[name]
        print(
            f"  {name:<6} {low:g}-{high:g} Hz: "
            f"power {band_power[name]:.3f}, ratio {ratios[name] * 100:.2f}%"
        )
    print(f"  Total ratio: {sum(ratios.values()) * 100:.2f}%")

    plot_band_pie(ratios, DEFAULT_BANDS, channel, OUTPUT_IMAGE)


if __name__ == "__main__":
    main()

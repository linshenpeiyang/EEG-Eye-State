#!/usr/bin/env python3
"""
EEG band-pass filtering with scipy.signal.butter + filtfilt, with a test routine.

Default parameters: 128 Hz sampling rate, 1-40 Hz passband, 4th-order Butterworth.

Usage:
    # Run the test and generate a comparison plot
    python eeg_bandpass_filter.py [channel]   # defaults to AF3

    # Reuse as a module
    from eeg_bandpass_filter import bandpass_filter
    y = bandpass_filter(x, lowcut=1.0, highcut=40.0, fs=128.0, order=4, axis=-1)
"""

from pathlib import Path
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import arff
from scipy.signal import butter, filtfilt, welch

# ---- Constants ----
DEFAULT_FS = 128.0
DEFAULT_LOWCUT = 1.0
DEFAULT_HIGHCUT = 40.0
DEFAULT_ORDER = 4

BASE_DIR = Path(__file__).resolve().parent
ARFF_PATH = BASE_DIR / "uci_eeg_eye_state_data" / "EEG Eye State.arff"
OUTPUT_IMAGE = BASE_DIR / "eeg_bandpass_filter_result.png"

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


def butter_bandpass(lowcut, highcut, fs, order=DEFAULT_ORDER):
    """Design Butterworth band-pass filter coefficients; return (b, a)."""
    nyquist = 0.5 * fs
    if not (0 < lowcut < highcut < nyquist):
        raise ValueError(
            f"Invalid cutoff frequencies: require 0 < {lowcut} < {highcut} < "
            f"{nyquist} Hz (fs={fs} Hz, Nyquist={nyquist} Hz)"
        )
    return butter(order, [lowcut, highcut], btype="band", fs=fs)


def _interpolate_nan(x, axis):
    """Linearly interpolate NaN values along an axis; filtfilt does not accept NaN."""
    x = np.moveaxis(x, axis, -1)
    result = np.empty_like(x)
    for index in np.ndindex(x.shape[:-1]):
        column = x[index]
        invalid = np.isnan(column)
        if invalid.any():
            valid = ~invalid
            if not valid.any():
                raise ValueError("Input contains a column of all NaN values and cannot be interpolated")
            column = column.copy()
            column[invalid] = np.interp(
                np.flatnonzero(invalid),
                np.flatnonzero(valid),
                column[valid],
            )
        result[index] = column
    return np.moveaxis(result, -1, axis)


def bandpass_filter(
    data,
    lowcut=DEFAULT_LOWCUT,
    highcut=DEFAULT_HIGHCUT,
    fs=DEFAULT_FS,
    order=DEFAULT_ORDER,
    axis=None,
):
    """Apply zero-phase band-pass filtering to an EEG signal and return the result in the same type as the input.

    Parameters
    ----------
    data : array-like
        1-D or multidimensional signal; accepts numpy arrays and pandas Series / DataFrame.
        DataFrame / Series are filtered along the rows by default, that is, axis=0;
        other arrays default to axis=-1.
    lowcut, highcut : float
        Low and high passband cutoff frequencies in Hz.
    fs : float
        Sampling rate in Hz, default 128.
    order : int
        Butterworth filter order, default 4.
    axis : int or None
        Axis to filter along; when None, it is selected with the rule above.
    """
    is_series = isinstance(data, pd.Series)
    is_frame = isinstance(data, pd.DataFrame)
    if axis is None:
        axis = 0 if (is_series or is_frame) else -1

    b, a = butter_bandpass(lowcut, highcut, fs, order)
    x = np.asarray(data, dtype=float)

    if np.isnan(x).any():
        warnings.warn("Input contains NaN; linearly interpolated along the filter axis before filtering", stacklevel=2)
        x = _interpolate_nan(x, axis)

    y = filtfilt(b, a, x, axis=axis)

    if is_frame:
        return pd.DataFrame(y, index=data.index, columns=data.columns)
    if is_series:
        return pd.Series(y, index=data.index, name=data.name)
    return y


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


def plot_comparison(raw, filtered, fs, channel, output_path, display_seconds=4.0):
    """Plot the waveform and power spectrum before and after filtering."""
    start = int(fs)  # Start at 1 s to avoid filtfilt edge effects at the beginning
    n_display = int(display_seconds * fs)
    end = start + n_display
    t = np.arange(start, end) / fs

    nperseg = min(int(2 * fs), len(raw))
    freqs, psd_raw = welch(raw, fs=fs, nperseg=nperseg, detrend="constant")
    _, psd_filt = welch(filtered, fs=fs, nperseg=nperseg, detrend="constant")

    fig, (ax_raw, ax_filt, ax_psd) = plt.subplots(
        3, 1, figsize=(12, 9), constrained_layout=True
    )

    ax_raw.plot(t, raw[start:end], color="tab:blue", lw=0.8, label="Raw signal")
    ax_raw.set_title(
        f"{channel} channel: waveform before and after filtering "
        f"({start / fs:.0f}-{end / fs:.0f} s, fs={fs:.0f} Hz)"
    )
    ax_raw.set_ylabel("Amplitude (μV)")
    ax_raw.legend(loc="upper right")
    ax_raw.grid(alpha=0.3)
    ax_raw.tick_params(labelbottom=False)

    ax_filt.plot(
        t, filtered[start:end], color="tab:red", lw=0.8,
        label="Band-pass filtered (1-40 Hz)",
    )
    ax_filt.set_ylabel("Amplitude (μV)")
    ax_filt.set_xlabel("Time (s)")
    ax_filt.legend(loc="upper right")
    ax_filt.grid(alpha=0.3)

    ax_psd.semilogy(freqs, psd_raw, color="tab:blue", lw=0.8, label="Raw PSD")
    ax_psd.semilogy(freqs, psd_filt, color="tab:red", lw=0.8, label="Filtered PSD")
    ax_psd.axvspan(1.0, 40.0, color="green", alpha=0.12, label="Passband 1-40 Hz")
    ax_psd.set_xlabel("Frequency (Hz)")
    ax_psd.set_ylabel("Power spectral density (μV²/Hz)")
    ax_psd.legend(loc="upper right")
    ax_psd.grid(alpha=0.3)

    fig.savefig(output_path, dpi=150)
    print(f"Comparison plot saved: {output_path}")
    if plt.get_backend().lower() != "agg":
        plt.show()
    return fig


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
    print(f"Channel: {channel}, samples: {len(raw)}, sampling rate: {DEFAULT_FS:.0f} Hz")
    print(
        f"Filter: Butterworth band-pass, order {DEFAULT_ORDER}, "
        f"passband {DEFAULT_LOWCUT}-{DEFAULT_HIGHCUT} Hz"
    )

    filtered = bandpass_filter(raw, fs=DEFAULT_FS)
    plot_comparison(raw, filtered, DEFAULT_FS, channel, OUTPUT_IMAGE)


if __name__ == "__main__":
    main()

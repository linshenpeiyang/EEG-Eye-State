#!/usr/bin/env python3
"""
Streamlit web app for EEG eyes-open and eyes-closed state analysis.

Run with:
    /opt/anaconda3/bin/streamlit run app.py

Features:
    - Sidebar controls for selecting the eye state and the waveform channel;
    - Displays the Alpha-band power bar chart for all channels and the raw
      waveform of the selected channel.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.io import arff
from scipy.signal import butter, filtfilt

# ---- Parameters ----
FS = 128.0                     # Sampling rate
ALPHA_BAND = (8.0, 13.0)       # Alpha band
ARTIFACT_THRESHOLD = 100.0     # Artifact threshold, relative to each channel median
STATE_LABELS = {0: "Eyes open", 1: "Eyes closed"}
STATE_COLORS = {0: "#0072B2", 1: "#D55E00"}

# Publication-style fonts: Times New Roman for Latin characters
plt.rcParams["font.family"] = [
    "Times New Roman",
    "Songti SC",
    "Arial Unicode MS",
    "DejaVu Serif",
]
plt.rcParams["axes.unicode_minus"] = False

st.set_page_config(page_title="EEG Eyes Open / Closed Analysis", page_icon="🧠", layout="wide")

# Candidate data folders: the original project folder and the delivery package
DATA_FOLDERS = ("uci_eeg_eye_state_data", "data")


def find_arff():
    """Locate the downloaded data file in the working directory or the home directory."""
    for base in (Path.cwd(), Path.home()):
        for folder in DATA_FOLDERS:
            path = base / folder / "EEG Eye State.arff"
            if path.exists():
                return path
    raise FileNotFoundError(
        "EEG Eye State.arff not found in uci_eeg_eye_state_data/ or data/. "
        "Run uci_eeg_eye_state.py first to download the data"
    )


@st.cache_data(show_spinner="Loading EEG data...")
def load_eeg():
    """Read the ARFF file and convert it to a DataFrame. The result is cached by Streamlit."""
    with find_arff().open("r", encoding="utf-8") as fh:
        data, _ = arff.loadarff(fh)

    df = pd.DataFrame(data)
    for column in df.columns:
        if df[column].dtype == object:
            df[column] = df[column].str.decode("utf-8")
    df = df.rename(columns={"eyeDetection": "eye_state"})
    df["eye_state"] = df["eye_state"].astype(int)
    return df


def reject_artifacts(x, threshold=ARTIFACT_THRESHOLD):
    """Remove spike artifacts whose deviation from the channel median exceeds the threshold."""
    keep = np.abs(x - np.median(x)) <= threshold
    return x[keep]


def alpha_energy(x, fs=FS, band=ALPHA_BAND, order=4):
    """Band-pass filter the signal and estimate Alpha power as the mean square."""
    b, a = butter(order, band, btype="band", fs=fs)
    y = filtfilt(b, a, x)
    return float(np.mean(y ** 2))


def longest_run(df, state):
    """Return the start sample and length of the longest continuous run of the given state."""
    mask = (df["eye_state"] == state).to_numpy()
    edges = np.diff(np.r_[False, mask, False].astype(int))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    i = int(np.argmax(ends - starts))
    return int(starts[i]), int(ends[i] - starts[i])


def plot_alpha_bars(alpha, state, state_label):
    """Horizontal bar chart of Alpha power per channel, sorted in ascending order."""
    channels = sorted(alpha, key=alpha.get)  # Ascending so the bars grow upward
    values = [alpha[ch] for ch in channels]

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    bars = ax.barh(channels, values, color=STATE_COLORS[state], alpha=0.85)
    ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
    ax.set_xlabel("Alpha power (a.u.²)")
    ax.set_title(f"{state_label}: Alpha power per channel (8-13 Hz)")
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_raw_waveform(x, state, state_label, channel, seconds=5.0):
    """Raw, unfiltered waveform of the selected channel in the given state."""
    n = int(seconds * FS)
    segment = x[:n]
    t = np.arange(len(segment)) / FS

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(t, segment, color=STATE_COLORS[state], linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (a.u.)")
    ax.set_title(
        f"{state_label}: raw waveform of channel {channel} "
        f"(first {seconds:.0f} s, unfiltered)"
    )
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


# ---- Data loading ----
df = load_eeg()
eeg_channels = [c for c in df.columns if c != "eye_state"]

# ---- Sidebar controls ----
with st.sidebar:
    st.header("⚙️ Controls")
    state_label = st.selectbox("Eye state", ["Eyes open", "Eyes closed"])
    channel = st.selectbox("Waveform channel", eeg_channels, index=0)
    st.divider()
    st.markdown("**Notes**")
    st.caption("Alpha power is computed after 8-13 Hz band-pass filtering, with spikes deviating from each channel median by more than ±100 removed first.")
    st.caption("The raw waveform is unfiltered and shows the actual recording.")

state = 1 if state_label == "Eyes closed" else 0
sub = df[df["eye_state"] == state]

# ---- Alpha power per channel ----
alpha = {}
for ch in eeg_channels:
    x = sub[ch].to_numpy(dtype=float)
    alpha[ch] = alpha_energy(reject_artifacts(x))

# ---- Charts ----
st.title("🧠 EEG Eyes Open / Closed Analysis")
st.caption(
    f"Selection: {state_label} ({len(sub):,} samples, "
    f"about {len(sub) / FS:.1f} s)"
)

col1, col2 = st.columns(2, gap="medium")
with col1:
    st.subheader("Alpha power bar chart")
    st.pyplot(plot_alpha_bars(alpha, state, state_label), width="stretch")
with col2:
    st.subheader("Raw waveform")
    start, _ = longest_run(df, state)
    waveform = df[channel].to_numpy(dtype=float)[start : start + int(5 * FS)]
    st.pyplot(
        plot_raw_waveform(waveform, state, state_label, channel),
        width="stretch",
    )

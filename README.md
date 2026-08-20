# EEG Eye State Analysis

EEG spectral analysis of the eyes-open and eyes-closed states.

## Overview

This project uses the public EEG Eye State dataset from UCI and covers the full pipeline from data acquisition and denoising to frequency-domain analysis. It contains two complementary deliverables:

- **Interactive app `app.py`**: a Streamlit application for browsing the data. The sidebar selects the eye state and the waveform channel; the main panel shows the Alpha power bar chart for every channel and the raw waveform.
- **Analysis report `eeg_eye_state_psd.ipynb`**: a Jupyter notebook that documents artifact rejection, Welch PSD estimation, band decomposition, plotting, and result interpretation. All outputs are embedded.

Key methods:

- Data: 128 Hz sampling, 14 EEG channels plus the eye-state label `eye_state`, 14,980 samples in total;
- Denoising: samples deviating from the channel median by more than ±100 are removed as spike artifacts;
- Frequency analysis: the Welch method estimates the power spectral density PSD, divided into Delta 0.5-4 Hz, Theta 4-8 Hz, Alpha 8-13 Hz, Beta 13-30 Hz, and Gamma 30-45 Hz bands;
- Main finding: Delta low-frequency power is higher with eyes closed than with eyes open; the classical eyes-closed Alpha enhancement is weak in this dataset.

Data source: Roesler, O. 2013. *EEG Eye State* [Dataset]. UCI Machine Learning Repository. <https://doi.org/10.24432/C57G7J>

## File Structure

```
EEG-Eye-State/
├── app/
│   └── app.py                 # Streamlit application for visualization
├── notebooks/
│   └── eeg_eye_state_psd.ipynb # Analysis report with embedded outputs
├── scripts/
│   ├── download_data.py       # Downloads the dataset
│   ├── bandpass_filter.py     # Butterworth band-pass filtering
│   ├── band_power.py          # PSD and band power ratios
│   └── eye_state_psd.py       # Command-line PSD comparison plot
├── data/
│   └── EEG Eye State.arff     # Dataset
├── requirements.txt           # Python dependencies
└── README.md                  # This document
```

- `app/app.py` is for demonstration: it loads the data and refreshes the charts immediately when the state changes.
- `notebooks/eeg_eye_state_psd.ipynb` is for reporting and reproduction: it explains the motivation and implementation of each step.
- The command-line scripts save generated figures to `results/`.

## Environment Setup

Create a dedicated Python 3.9 environment with Anaconda:

```bash
conda create -n eeg_project python=3.9 -y
conda activate eeg_project
pip install -r requirements.txt
```

On slow networks, use a mirror:

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

The pinned versions match the development environment and are compatible with Python 3.9. The scripts read ARFF data with `scipy.io.arff`; `liac-arff` is included as a fallback parser.

If `data/` lacks `EEG Eye State.arff`, download it first:

```bash
python scripts/download_data.py
```

## Usage

### Launch the Web App

```bash
conda activate eeg_project
streamlit run app/app.py
```

Open <http://localhost:8501>. Select the eye state in the sidebar; the Alpha power bar chart and raw waveform update immediately. To use a different port:

```bash
streamlit run app/app.py --server.port 8502
```

### Open the Analysis Report

```bash
conda activate eeg_project
jupyter lab notebooks/eeg_eye_state_psd.ipynb
```

All results are embedded. To rerun the notebook, execute the cells in order with Shift+Enter.

## Demo Suggestions

Present the theory first and the live demo second, in about 8-10 minutes:

1. Open the notebook and explain the method and theory.
   - Dataset and label: 128 Hz, 14 channels, 0 = eyes open, 1 = eyes closed.
   - Why artifact rejection is needed: blinks, muscle activity, and poor electrode contact produce broadband interference.
   - The Welch PSD and the physiological roles of the five bands from Delta to Gamma.
   - The conclusion: Delta low-frequency power rises with eyes closed, while the alpha effect is weak.

2. Run the Streamlit demo live.
   - Switch the eye-state selector to show changes in Alpha energy and the raw waveform.
   - Switch the displayed channel, for example occipital O1 or O2, to show regional differences.

3. Checklist before the demo.
   - Start the app in advance and confirm <http://localhost:8501> is reachable.
   - Confirm the data file exists in `data/` to avoid downloading on site.
   - The notebook outputs are embedded, so no re-execution is needed.

## Reference

Roesler, O. 2013. *EEG Eye State* [Dataset]. UCI Machine Learning Repository. <https://doi.org/10.24432/C57G7J>

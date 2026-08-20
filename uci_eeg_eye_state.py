#!/usr/bin/env python3
"""
Download the UCI EEG Eye State dataset and load it as a pandas DataFrame.

Workflow:
1. Download the official zip archive with urllib;
2. Extract the .arff file with zipfile;
3. Parse the ARFF with scipy.io.arff and convert it to a pandas DataFrame;
4. Print the first five rows and basic information.

Dependencies:
    pip install pandas scipy

Usage:
    python uci_eeg_eye_state.py
"""

from pathlib import Path
import urllib.request
import zipfile

import pandas as pd
from scipy.io import arff

# Official UCI download URL, dynamically packaged zip from the new site
DATA_URL = "https://archive.ics.uci.edu/static/public/264/eeg+eye+state.zip"

# Data directory: uci_eeg_eye_state_data/ next to this script
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "uci_eeg_eye_state_data"
ARCHIVE_PATH = DATA_DIR / "eeg-eye-state.zip"

CHUNK_SIZE = 1024 * 1024  # Read 1 MB at a time


def download(url: str, destination: Path) -> Path:
    """Download a file with urllib and print the progress."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(url) as response, destination.open("wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        downloaded = 0
        while True:
            chunk = response.read(CHUNK_SIZE)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            if total:
                percent = downloaded * 100 // total
                print(
                    f"\rDownload progress: {percent}% "
                    f"({downloaded:,} / {total:,} bytes)",
                    end="",
                    flush=True,
                )
            else:
                print(f"\rDownloaded: {downloaded:,} bytes", end="", flush=True)
    print()
    return destination


def extract_arff(archive_path: Path, output_dir: Path) -> Path:
    """Extract the single .arff file from the zip archive."""
    with zipfile.ZipFile(archive_path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".arff")]
        if not members:
            raise ValueError("No .arff file found in the archive")
        if len(members) > 1:
            raise ValueError(f"The archive contains multiple .arff files: {members}")

        member = members[0]
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / Path(member).name

        with archive.open(member) as source, target.open("wb") as out:
            out.write(source.read())

    return target


def arff_to_dataframe(arff_path: Path) -> tuple[pd.DataFrame, dict]:
    """Parse an ARFF file and convert it to a pandas DataFrame."""
    # scipy.io.arff requires the file to be opened in text mode
    with arff_path.open("r", encoding="utf-8") as fh:
        data, meta = arff.loadarff(fh)

    df = pd.DataFrame(data)

    # scipy reads nominal attributes such as category columns as bytes; decode them to strings
    for column in df.columns:
        if df[column].dtype == object:
            df[column] = df[column].str.decode("utf-8")

    # Convert the {0,1} category column to integers
    target = "eyeDetection"
    if target in df.columns and set(df[target].unique()) <= {"0", "1"}:
        df[target] = df[target].astype(int)

    return df, meta


def main() -> pd.DataFrame:
    # 1. Download, skipping if the archive already exists
    if ARCHIVE_PATH.exists():
        print(f"Archive already exists, skipping download: {ARCHIVE_PATH}")
    else:
        print(f"Downloading dataset: {DATA_URL}")
        download(DATA_URL, ARCHIVE_PATH)

    # 2. Extract
    arff_path = extract_arff(ARCHIVE_PATH, DATA_DIR)
    print(f"Extracted ARFF file: {arff_path}")

    # 3. Parse and convert to a DataFrame
    df, meta = arff_to_dataframe(arff_path)
    print(f"\nARFF metadata:\n{meta}")

    # 4. Print the first five rows and basic information
    print("\nFirst five rows:")
    print(df.head())
    print("\nDataset information:")
    df.info()

    return df


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
下载 UCI EEG Eye State 数据集并加载为 pandas DataFrame。

流程：
1. 使用 urllib 下载官方 zip 压缩包；
2. 用 zipfile 解压出 .arff 文件；
3. 用 scipy.io.arff 解析 ARFF 并转换为 pandas DataFrame；
4. 打印前 5 行数据和基本信息。

依赖：
    pip install pandas scipy

用法：
    python uci_eeg_eye_state.py
"""

from pathlib import Path
import urllib.request
import zipfile

import pandas as pd
from scipy.io import arff

# UCI 官方下载地址（新站点提供动态打包的 zip）
DATA_URL = "https://archive.ics.uci.edu/static/public/264/eeg+eye+state.zip"

# 数据保存目录：与脚本同级的 uci_eeg_eye_state_data/
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "uci_eeg_eye_state_data"
ARCHIVE_PATH = DATA_DIR / "eeg-eye-state.zip"

CHUNK_SIZE = 1024 * 1024  # 每次读取 1 MB


def download(url: str, destination: Path) -> Path:
    """使用 urllib 下载文件，并打印进度。"""
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
                    f"\r下载进度: {percent}% "
                    f"({downloaded:,} / {total:,} bytes)",
                    end="",
                    flush=True,
                )
            else:
                print(f"\r已下载: {downloaded:,} bytes", end="", flush=True)
    print()
    return destination


def extract_arff(archive_path: Path, output_dir: Path) -> Path:
    """从 zip 中解压唯一的 .arff 文件。"""
    with zipfile.ZipFile(archive_path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".arff")]
        if not members:
            raise ValueError("压缩包中未找到 .arff 文件")
        if len(members) > 1:
            raise ValueError(f"压缩包中包含多个 .arff 文件: {members}")

        member = members[0]
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / Path(member).name

        with archive.open(member) as source, target.open("wb") as out:
            out.write(source.read())

    return target


def arff_to_dataframe(arff_path: Path) -> tuple[pd.DataFrame, dict]:
    """解析 ARFF 文件并转换为 pandas DataFrame。"""
    # scipy.io.arff 需要以文本模式打开文件
    with arff_path.open("r", encoding="utf-8") as fh:
        data, meta = arff.loadarff(fh)

    df = pd.DataFrame(data)

    # scipy 把 nominal 属性（如类别列）读成 bytes，这里统一解码为字符串
    for column in df.columns:
        if df[column].dtype == object:
            df[column] = df[column].str.decode("utf-8")

    # 类别列 {0,1} 转成整数
    target = "eyeDetection"
    if target in df.columns and set(df[target].unique()) <= {"0", "1"}:
        df[target] = df[target].astype(int)

    return df, meta


def main() -> pd.DataFrame:
    # 1. 下载（已存在则跳过）
    if ARCHIVE_PATH.exists():
        print(f"压缩包已存在，跳过下载: {ARCHIVE_PATH}")
    else:
        print(f"正在下载数据集: {DATA_URL}")
        download(DATA_URL, ARCHIVE_PATH)

    # 2. 解压
    arff_path = extract_arff(ARCHIVE_PATH, DATA_DIR)
    print(f"已解压 ARFF 文件: {arff_path}")

    # 3. 解析并转换为 DataFrame
    df, meta = arff_to_dataframe(arff_path)
    print(f"\nARFF 元信息:\n{meta}")

    # 4. 打印前 5 行和基本信息
    print("\n前 5 行数据:")
    print(df.head())
    print("\n数据基本信息:")
    df.info()

    return df


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
按眼动状态（eye_state）分组，绘制睁眼 / 闭眼状态下的 PSD 对比折线图。

eye_state 含义（来自 UCI 数据集说明）:
    0 = 睁眼 (eyes open)
    1 = 闭眼 (eyes closed)

流程:
    1. 读取 ARFF，把标签列重命名为 eye_state；
    2. 按 eye_state 分组，逐通道剔除尖峰伪迹（|x - 中位数| > 阈值）；
    3. 对每个通道用 Welch 法估计 PSD，默认对所有通道取平均；
    4. 绘制科研风格的半对数折线图（中文标题、图例、网格线、频带底纹）。

用法:
    python eeg_eye_state_psd.py                 # 全通道平均
    python eeg_eye_state_psd.py O1              # 指定单通道
    python eeg_eye_state_psd.py O1 50           # 指定通道和伪迹阈值
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import arff
from scipy.signal import welch

# ---- 常量 ----
DEFAULT_FS = 128.0
ARTIFACT_THRESHOLD = 100.0  # 相对各通道中位数的伪迹阈值

STATE_LABELS = {0: "睁眼", 1: "闭眼"}
STATE_COLORS = {0: "#0072B2", 1: "#D55E00"}

BANDS = {
    "Delta": (0.5, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta": (13.0, 30.0),
    "Gamma": (30.0, 45.0),
}

BASE_DIR = Path(__file__).resolve().parent
ARFF_PATH = BASE_DIR / "uci_eeg_eye_state_data" / "EEG Eye State.arff"
OUTPUT_IMAGE = BASE_DIR / "eeg_eye_state_psd.png"

# 科研风格字体: 拉丁字符用 Times New Roman, 中文回退到宋体
plt.rcParams["font.family"] = [
    "Times New Roman",
    "Songti SC",
    "Arial Unicode MS",
    "DejaVu Serif",
]
plt.rcParams["axes.unicode_minus"] = False


def load_arff(arff_path):
    """读取 ARFF，转换为 DataFrame，并把标签列重命名为 eye_state。"""
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
    """按眼动状态分组计算 PSD，返回 {状态: {"freqs": ..., "psd": ..., "n": ...}}。

    channel=None 时对所有 EEG 通道的 PSD 取平均；否则只计算该通道。
    """
    eeg_channels = [c for c in df.columns if c != state_column]
    if channel is not None:
        if channel not in eeg_channels:
            available = ", ".join(eeg_channels)
            raise ValueError(f"通道 {channel!r} 不存在，可用通道: {available}")
        channels = [channel]
    else:
        channels = eeg_channels

    if nperseg is None:
        nperseg = int(2 * fs)  # 2 秒窗口，频率分辨率 0.5 Hz

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
    """绘制科研风格的睁眼 / 闭眼 PSD 对比折线图。"""
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

    # 频带底纹与标注
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
        f"睁眼与闭眼状态 EEG 功率谱密度对比（{channel_label}）",
        fontsize=14,
        pad=14,
    )
    ax.set_xlabel("频率 (Hz)", fontsize=12)
    ax.set_ylabel("功率谱密度 (a.u.²/Hz)", fontsize=12)
    ax.set_xlim(0.5, 45.0)
    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.tick_params(labelsize=11, direction="in")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=11)

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"对比图已保存: {output_path}")
    if plt.get_backend().lower() != "agg":
        plt.show()
    return fig


def main():
    channel = sys.argv[1] if len(sys.argv) > 1 else None
    if channel and channel.lower() in {"all", "平均", "avg"}:
        channel = None
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else ARTIFACT_THRESHOLD

    if not ARFF_PATH.exists():
        raise FileNotFoundError(
            f"未找到 EEG 数据: {ARFF_PATH}\n"
            "请先运行 uci_eeg_eye_state.py 下载数据集"
        )

    df = load_arff(ARFF_PATH)
    result = group_psd(df, channel=channel, threshold=threshold)

    channel_label = channel if channel else "全通道平均"
    print(f"通道: {channel_label}，采样率: {DEFAULT_FS:.0f} Hz")
    for state in sorted(result):
        info = result[state]
        print(
            f"  {STATE_LABELS[state]}: 原始样本 {int(df['eye_state'].eq(state).sum()):,}, "
            f"伪迹剔除后 {info['n']:,}（阈值 ±{threshold:g}）"
        )

    plot_psd_comparison(result, channel_label, OUTPUT_IMAGE)


if __name__ == "__main__":
    main()

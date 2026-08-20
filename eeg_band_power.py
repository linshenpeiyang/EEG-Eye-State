#!/usr/bin/env python3
"""
计算 EEG 信号的功率谱密度（PSD）与五个频带的能量占比，并画饼图。

频带定义（标准 EEG 分段，可按需覆盖）：
    Delta: 0.5-4 Hz
    Theta: 4-8 Hz
    Alpha: 8-13 Hz
    Beta:  13-30 Hz
    Gamma: 30-45 Hz

用法:
    # 运行演示并生成饼图
    python eeg_band_power.py [通道名]   # 默认通道 AF3

    # 作为模块复用
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

try:  # numpy >= 2.0 用 trapezoid, 旧版本用 trapz
    from numpy import trapezoid as _trapz
except ImportError:
    from numpy import trapz as _trapz

# ---- 常量 ----
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

# macOS 下让 matplotlib 正常显示中文
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
    """用 Welch 法估计单边功率谱密度，返回 (freqs, psd)。

    psd 的单位为 (输入信号单位)²/Hz，对频率积分后的功率单位为
    (输入信号单位)²，具体量纲取决于 x 的物理单位。

    参数
    ----
    x : array-like
        一维 EEG 信号。
    fs : float
        采样率（Hz），默认 128。
    nperseg : int or None
        每段样本数；默认取 2 秒窗口（频率分辨率 0.5 Hz）。
    window, detrend
        直接传给 scipy.signal.welch。
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("compute_psd 只支持一维信号")

    if nperseg is None:
        nperseg = min(int(2 * fs), x.size)
    if nperseg > x.size:
        raise ValueError(f"nperseg={nperseg} 大于信号长度 {x.size}")

    freqs, psd = welch(
        x, fs=fs, window=window, nperseg=nperseg, detrend=detrend
    )
    return freqs, psd


def compute_band_power(freqs, psd, bands=None):
    """按频带对 PSD 积分，返回 (各频带绝对功率, 归一化能量占比)。

    占比以五个频带的总功率为分母，保证加起来等于 100%。
    """
    if bands is None:
        bands = DEFAULT_BANDS

    band_power = {}
    total = 0.0
    for name, (low, high) in bands.items():
        if low >= freqs[-1]:
            warnings.warn(
                f"{name} 频带 ({low}-{high} Hz) 完全超出 "
                f"测量范围 (最高 {freqs[-1]:.1f} Hz)，该频带记为 0"
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
        raise ValueError("所有频带的功率总和为 0，无法计算占比")

    ratios = {name: power / total for name, power in band_power.items()}
    return band_power, ratios


def plot_band_pie(ratios, bands=None, channel=None, output_path=None):
    """把频带能量占比画成饼图。"""
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
        f"{channel} 通道频带能量占比" if channel else "频带能量占比",
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
        title="频带",
    )
    ax.axis("equal")

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"饼图已保存: {output_path}")
    if plt.get_backend().lower() != "agg":
        plt.show()
    return fig


def load_arff(arff_path):
    """读取 ARFF 并转换为 DataFrame（与下载脚本一致）。"""
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
            f"未找到 EEG 数据: {ARFF_PATH}\n"
            "请先运行 uci_eeg_eye_state.py 下载数据集"
        )

    df = load_arff(ARFF_PATH)
    if channel not in df.columns:
        available = ", ".join(df.columns)
        raise ValueError(f"通道 {channel!r} 不存在，可用通道: {available}")

    raw = df[channel].to_numpy(dtype=float)
    freqs, psd = compute_psd(raw, fs=DEFAULT_FS)
    band_power, ratios = compute_band_power(freqs, psd)

    resolution = freqs[1] - freqs[0]
    print(f"通道: {channel}，样本数: {len(raw)}，采样率: {DEFAULT_FS:.0f} Hz")
    print(f"PSD: Welch 法，频率分辨率 {resolution:.2f} Hz\n")

    print("频带能量统计:")
    for name in DEFAULT_BANDS:
        low, high = DEFAULT_BANDS[name]
        print(
            f"  {name:<6} {low:g}-{high:g} Hz: "
            f"功率 {band_power[name]:.3f}，占比 {ratios[name] * 100:.2f}%"
        )
    print(f"  合计占比: {sum(ratios.values()) * 100:.2f}%")

    plot_band_pie(ratios, DEFAULT_BANDS, channel, OUTPUT_IMAGE)


if __name__ == "__main__":
    main()

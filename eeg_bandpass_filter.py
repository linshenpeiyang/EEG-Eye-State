#!/usr/bin/env python3
"""
基于 scipy.signal.butter + filtfilt 的 EEG 带通滤波函数与测试。

默认参数: 采样率 128 Hz, 通带 1-40 Hz, 4 阶 Butterworth。

用法:
    # 运行测试并生成对比图
    python eeg_bandpass_filter.py [通道名]   # 默认通道 AF3

    # 作为模块复用
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

# ---- 常量 ----
DEFAULT_FS = 128.0
DEFAULT_LOWCUT = 1.0
DEFAULT_HIGHCUT = 40.0
DEFAULT_ORDER = 4

BASE_DIR = Path(__file__).resolve().parent
ARFF_PATH = BASE_DIR / "uci_eeg_eye_state_data" / "EEG Eye State.arff"
OUTPUT_IMAGE = BASE_DIR / "eeg_bandpass_filter_result.png"

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


def butter_bandpass(lowcut, highcut, fs, order=DEFAULT_ORDER):
    """设计 Butterworth 带通滤波器系数，返回 (b, a)。"""
    nyquist = 0.5 * fs
    if not (0 < lowcut < highcut < nyquist):
        raise ValueError(
            f"截止频率不合法: 需要 0 < {lowcut} < {highcut} < "
            f"{nyquist} Hz (fs={fs} Hz, Nyquist={nyquist} Hz)"
        )
    return butter(order, [lowcut, highcut], btype="band", fs=fs)


def _interpolate_nan(x, axis):
    """沿指定轴对 NaN 做线性插值（filtfilt 不支持 NaN）。"""
    x = np.moveaxis(x, axis, -1)
    result = np.empty_like(x)
    for index in np.ndindex(x.shape[:-1]):
        column = x[index]
        invalid = np.isnan(column)
        if invalid.any():
            valid = ~invalid
            if not valid.any():
                raise ValueError("输入中存在整列全为 NaN 的数据，无法插值")
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
    """对 EEG 信号做零相位带通滤波，返回与输入同类型的结果。

    参数
    ----
    data : array-like
        1-D 或多维信号；支持 numpy 数组、pandas Series / DataFrame。
        DataFrame / Series 默认沿行（时间轴）滤波，即 axis=0；
        其他数组默认 axis=-1。
    lowcut, highcut : float
        通带低频 / 高频截止频率（Hz）。
    fs : float
        采样率（Hz），默认 128。
    order : int
        Butterworth 滤波器阶数，默认 4。
    axis : int or None
        滤波轴；None 时按上面的规则自动选择。
    """
    is_series = isinstance(data, pd.Series)
    is_frame = isinstance(data, pd.DataFrame)
    if axis is None:
        axis = 0 if (is_series or is_frame) else -1

    b, a = butter_bandpass(lowcut, highcut, fs, order)
    x = np.asarray(data, dtype=float)

    if np.isnan(x).any():
        warnings.warn("输入含 NaN，已沿滤波轴线性插值后再滤波", stacklevel=2)
        x = _interpolate_nan(x, axis)

    y = filtfilt(b, a, x, axis=axis)

    if is_frame:
        return pd.DataFrame(y, index=data.index, columns=data.columns)
    if is_series:
        return pd.Series(y, index=data.index, name=data.name)
    return y


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


def plot_comparison(raw, filtered, fs, channel, output_path, display_seconds=4.0):
    """画出滤波前后的波形和功率谱对比图。"""
    start = int(fs)  # 从 1 s 开始展示，避开 filtfilt 起始端的边缘效应
    n_display = int(display_seconds * fs)
    end = start + n_display
    t = np.arange(start, end) / fs

    nperseg = min(int(2 * fs), len(raw))
    freqs, psd_raw = welch(raw, fs=fs, nperseg=nperseg, detrend="constant")
    _, psd_filt = welch(filtered, fs=fs, nperseg=nperseg, detrend="constant")

    fig, (ax_raw, ax_filt, ax_psd) = plt.subplots(
        3, 1, figsize=(12, 9), constrained_layout=True
    )

    ax_raw.plot(t, raw[start:end], color="tab:blue", lw=0.8, label="原始信号")
    ax_raw.set_title(
        f"{channel} 通道 · 滤波前后波形对比 "
        f"({start / fs:.0f}-{end / fs:.0f} 秒, fs={fs:.0f} Hz)"
    )
    ax_raw.set_ylabel("幅值 (μV)")
    ax_raw.legend(loc="upper right")
    ax_raw.grid(alpha=0.3)
    ax_raw.tick_params(labelbottom=False)

    ax_filt.plot(
        t, filtered[start:end], color="tab:red", lw=0.8,
        label="带通滤波后 (1-40 Hz)",
    )
    ax_filt.set_ylabel("幅值 (μV)")
    ax_filt.set_xlabel("时间 (s)")
    ax_filt.legend(loc="upper right")
    ax_filt.grid(alpha=0.3)

    ax_psd.semilogy(freqs, psd_raw, color="tab:blue", lw=0.8, label="原始 PSD")
    ax_psd.semilogy(freqs, psd_filt, color="tab:red", lw=0.8, label="滤波后 PSD")
    ax_psd.axvspan(1.0, 40.0, color="green", alpha=0.12, label="通带 1-40 Hz")
    ax_psd.set_xlabel("频率 (Hz)")
    ax_psd.set_ylabel("功率谱密度 (μV²/Hz)")
    ax_psd.legend(loc="upper right")
    ax_psd.grid(alpha=0.3)

    fig.savefig(output_path, dpi=150)
    print(f"对比图已保存: {output_path}")
    if plt.get_backend().lower() != "agg":
        plt.show()
    return fig


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
    print(f"通道: {channel}，样本数: {len(raw)}，采样率: {DEFAULT_FS:.0f} Hz")
    print(
        f"滤波器: Butterworth 带通，阶数 {DEFAULT_ORDER}，"
        f"通带 {DEFAULT_LOWCUT}-{DEFAULT_HIGHCUT} Hz"
    )

    filtered = bandpass_filter(raw, fs=DEFAULT_FS)
    plot_comparison(raw, filtered, DEFAULT_FS, channel, OUTPUT_IMAGE)


if __name__ == "__main__":
    main()

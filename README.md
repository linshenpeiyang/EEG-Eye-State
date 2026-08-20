# EEG 眼动状态分析项目

> 交互式探索 × 深度频谱分析：睁眼 / 闭眼状态下脑电节律的可视化与量化研究。

## 项目简介

本项目基于 UCI「EEG Eye State」公开数据集，实现了一条从数据获取、去噪预处理到频域分析的完整流程，交付物包含两个互补的部分：

- **交互式应用 `app.py`**（Streamlit）：无需编写代码即可快速浏览数据。在侧边栏选择「睁眼 / 闭眼」状态，右侧实时展示各通道 Alpha 波（8-13 Hz）能量柱状图与原始波形图。
- **深度分析报告 `eeg_eye_state_psd.ipynb`**（Jupyter Notebook）：完整记录 PSD 分析逻辑，包括伪迹剔除、Welch 谱估计、频带划分、绘图与结果解读，且已嵌入全部运行输出，可直接审阅。

方法要点：

- 数据：128 Hz 采样、14 个 EEG 通道 + 眼动标签 `eye_state`（0 = 睁眼，1 = 闭眼），共 14,980 个样本；
- 去噪：剔除偏离各通道中位数超过 ±100 的尖峰伪迹（眨眼、肌电、电极接触不良等宽带干扰）；
- 频域分析：Welch 法估计功率谱密度（PSD），按 Delta（0.5-4 Hz）、Theta（4-8 Hz）、Alpha（8-13 Hz）、Beta（13-30 Hz）、Gamma（30-45 Hz）划分频带；
- 主要发现：闭眼状态下 Delta 低频功率明显高于睁眼状态；教科书中的「闭眼 Alpha 增强」效应在这份噪声较大的公开数据中较弱。

数据来源：Roesler, O. (2013). EEG Eye State [Dataset]. UCI Machine Learning Repository. <https://doi.org/10.24432/C57G7J>

## 文件结构

```
EEG-Eye-State/
├── app.py                    # Streamlit 交互式应用（快速可视化工具）
├── eeg_eye_state_psd.ipynb   # 深度分析报告（算法验证 + 结果解读，已嵌入输出）
├── uci_eeg_eye_state.py      # 辅助：使用 urllib 自动下载数据集
├── eeg_bandpass_filter.py    # 辅助：butter + filtfilt 带通滤波
├── eeg_band_power.py         # 辅助：PSD 与频带能量占比计算
├── eeg_eye_state_psd.py      # 辅助：命令行版 PSD 对比绘图
├── uci_eeg_eye_state_data/   # 数据目录（EEG Eye State.arff）
├── requirements.txt          # Python 依赖清单
└── README.md                 # 本文档
```

- `app.py` 面向演示与快速检查：加载数据后通过侧边栏下拉框切换状态，图表即时刷新；
- `eeg_eye_state_psd.ipynb` 面向汇报与复现：按步骤逐段解释每一步的动机、生理意义与代码实现，是本项目的核心分析报告。

## 环境安装指南

推荐使用 Anaconda 创建独立的 Python 3.9 虚拟环境：

```bash
# 1. 创建虚拟环境
conda create -n eeg_project python=3.9 -y

# 2. 激活环境
conda activate eeg_project

# 3. 安装依赖
pip install -r requirements.txt
```

网络较慢时可使用国内镜像源：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

依赖清单中的版本号与开发验证环境一致，且全部兼容 Python 3.9。说明：当前脚本使用 `scipy.io.arff` 读取 ARFF 数据，`liac-arff` 作为备用的 ARFF 解析库一并列入。

如果打包目录 `uci_eeg_eye_state_data/` 中缺少 `EEG Eye State.arff`，请先运行下载脚本获取数据：

```bash
python uci_eeg_eye_state.py
```

## 运行操作手册

### 启动网页应用

```bash
conda activate eeg_project
streamlit run app.py
```

在浏览器中访问 <http://localhost:8501>。左侧下拉框选择「睁眼 / 闭眼」，右侧的 Alpha 波能量柱状图与原始波形图会实时刷新。若默认端口被占用，可指定其他端口：

```bash
streamlit run app.py --server.port 8502
```

### 查看分析报告

```bash
conda activate eeg_project
jupyter lab eeg_eye_state_psd.ipynb
```

Notebook 已嵌入全部运行结果，打开即可查看；如需重新执行，依次按 Shift+Enter 运行各单元格。

## 给导师的演示建议（重要）

建议采用「先理论、后交互」的顺序，总时长约 8-10 分钟：

1. **先打开 Notebook 讲方法与理论**（约 5 分钟）
   - 数据集与眼动标签含义（128 Hz、14 通道，0 = 睁眼、1 = 闭眼）；
   - 伪迹剔除的动机：眨眼、肌电与电极接触不良是宽带干扰，会严重污染频谱估计；
   - Welch PSD 原理与五个频带（Delta ~ Gamma）的生理意义对照表；
   - 结论图解读：闭眼时 Delta 低频功率升高，并客观说明 alpha 效应在本数据中较弱。

2. **再启动 Streamlit 现场演示**（约 3-5 分钟）
   - 切换「睁眼 / 闭眼」下拉框，展示 Alpha 能量柱状图与原始波形的即时变化，突出交互性；
   - 可进一步切换波形显示通道（如枕区 O1/O2），展示不同脑区的表现。

3. **演示前准备清单**
   - 提前 `conda activate eeg_project` 并启动服务，确认 <http://localhost:8501> 可访问；
   - 确认数据文件存在于 `uci_eeg_eye_state_data/`，避免现场重新下载；
   - Notebook 已嵌入输出，无需现场重新执行，避免等待。

## 引用

Roesler, O. (2013). *EEG Eye State* [Dataset]. UCI Machine Learning Repository. <https://doi.org/10.24432/C57G7J>

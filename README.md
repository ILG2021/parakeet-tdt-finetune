# 🚀 NVIDIA Parakeet-TDT v3 中文 ASR 微调专家指南

本项目致力于在 **NVIDIA Parakeet-TDT 0.6b-v3** 强大的声学编码器基础上，通过大规模中文数据集（Emilia, WenetSpeech, KeSpeech）以及私有领域数据，训练出一个高精度、原生支持标点的中文自动语音识别模型。

---

## 🛠️ 1. 环境准备

针对 **RTX 5090** 等现代 GPU 优化的依赖环境：

```bash
# 安装基础工具及 NeMo (ASR 核心)
pip install -U nemo_toolkit[asr]>=2.0.0
pip install pytorch-lightning>=2.0 omegaconf librosa cython soundfile

# 安装 FunASR (用于高质量标点恢复)
pip install -U git+https://github.com/ILG2021/FunASR.git
```

---

## 📂 2. 多源数据预处理

为了避开系统盘压力，所有脚本默认将 Hugging Face 缓存放在 `./hf_cache`。

### A. 通用大数据集 (Hugging Face)
针对不同的主流数据集，我们提供了专门的处理脚本：

| 数据集 | 处理脚本 | 字段适配 |
| :--- | :--- | :--- |
| **Emilia-YODAS** | `dataset/prepare_emilia_zh.py` | 适配嵌套 JSON 元数据 |
| **WenetSpeech** | `dataset/prepare_wenet_data.py` | 适配广播/分段字段 |
| **KeSpeech** | `dataset/prepare_kespeech_data.py` | 适配多口音/方言字段 |

**运行示例：**
```bash
# 自动下载、本地保存 wav 并恢复标点
python dataset/prepare_wenet_data.py --output_manifest wenet.json --add_punctuation
```

### B. 私有领域数据 (本地)
*   **AudioFolder 格式** (包含 `metadata.csv`):
    ```bash
    python dataset/prepare_audiofolder.py --data_dir ./my_data --text_col sentence --output_manifest audiofolder.json
    ```
*   **LJSpeech 格式**:
    ```bash
    python dataset/prepare_local_ljspeech.py --data_folder ./ljspeech_root --output_manifest ljspeech.json
    ```

---

## 🧬 3. 核心微调训练 (针对 RTX 5090 优化)

`finetune.py` 实现了自动词表替换逻辑：它会扫描所有训练文本，自动构建**中文字符级 (Character-based)** 词表，并完成模型输出层的热插拔。

### 全量训练启动命令
```bash
# 将所有清单文件作为输入
$train_manifests = "emilia.json,wenet.json,kespeech.json,audiofolder.json,ljspeech.json"

python finetune.py `
    --train_manifest $train_manifests `
    --batch_size 16 `
    --grad_acc 4 `
    --lr 7.5e-5 `
    --epochs 15 `
    --save_path parakeet_tdt_zh_5090_final.nemo
```

**关键优化：**
*   **混合精度**：自动启用 `bf16-mixed`。
*   **词表替换**：自动将英文 BPE 转换为中文 Char 模型。
*   **高吞吐并行**：针对 5090 优化的数据加载与梯度累积。

---

## 🎯 4. 推理演练

使用微调后的模型进行单文件测试：

```bash
python inference.py --model parakeet_tdt_zh_5090_final.nemo --audio samples/demo.wav
```

---

## 🏗️ 项目结构说明

*   `dataset/`: 存放所有针对性数据处理脚本。
*   `finetune.py`: 自动化微调核心，包含词表替换与 TDT 模型配置。
*   `inference.py`: 模型离线测试脚本。
*   `hf_cache/`: (自动生成) 数据集下载缓存，建议放在大容量分区。
*   `data/`: (自动生成) 转换后的本地 wav 存储库。

---

## 📝 经验贴士
1.  **关于收敛**：在大规模数据（2000h+）上，0.6b 模型通常在 5-8 轮左右即可达到极佳效果。
2.  **显存报警**：若 5090 提示显存异常，请适当将 `batch_size` 降至 8。

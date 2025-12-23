# NVIDIA Parakeet-TDT 0.6b-v3 微调指南

本指南详细介绍了如何使用 NVIDIA NeMo 工具包对 `parakeet-tdt-0.6b-v3` 模型进行微调。

## 1. 安装依赖

首先，确保你的环境中安装了 Python 3.10+ 和 PyTorch 2.0+。

```bash
# 安装 NeMo toolkit (包含 ASR 相关依赖)
pip install -U nemo_toolkit['asr']>=2.0.0
pip install -U pytorch-lightning>=2.0 omegaconf librosa cython
```

## 2. 数据预处理

NeMo 使用 `.json` 格式的清单文件（Manifest）。每一行都是一个 JSON 对象，包含音频路径、时长和对应的文本。

你可以准备一个目录，其中每个 `.wav` 音频文件都有一个同名的 `.txt` 文本文件作为标注。

使用 `prepare_data.py` 脚本生成清单：

```bash
python prepare_data.py --audio_dir ./data/train --output_manifest train_manifest.json
python prepare_data.py --audio_dir ./data/val --output_manifest val_manifest.json
```

## 3. 微调代码

微调的核心在于加载预训练模型，并使用 PyTorch Lightning 训练器进行训练。

运行微调任务：

```bash
python finetune.py \
    --train_manifest train_manifest.json \
    --val_manifest val_manifest.json \
    --save_path parakeet_tdt_finetuned.nemo
```

### 微调建议：
- **学习率**：由于是微调，学习率通常设置在 `1e-5` 到 `7.5e-` 之间。
- **混合精度**：脚本默认开启了 `16-mixed` 精度，以节省显存。
- **冻结层**：如果数据集非常小，可以考虑通过 `model.encoder.freeze()` 冻结编码器。

## 4. 推理验证

微调完成后，你可以使用 `inference.py` 来验证效果。

```bash
# 使用微调后的模型进行推理
python inference.py --model parakeet_tdt_finetuned.nemo --audio test_sample.wav

# 也可以对比原始预训练模型的效果
python inference.py --model nvidia/parakeet-tdt-0.6b-v3 --audio test_sample.wav
```

## 项目结构
- `requirements.txt`: 依赖列表。
- `prepare_data.py`: 数据清单生成工具。
- `finetune.py`: 微调核心逻辑。
- `inference.py`: 结果评估脚本。

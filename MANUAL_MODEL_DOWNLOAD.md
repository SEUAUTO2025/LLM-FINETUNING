# 手动下载 Qwen3-4B-Instruct-2507

训练机能够访问 Hugging Face 时，在仓库根目录执行：

```bash
cd /home/seuauto/Codes/LLM-FInetuning/LLM-FINETUNING
bash download_qwen3_4b.sh
```

如果官方地址无法访问，可先指定镜像：

```bash
cd /home/seuauto/Codes/LLM-FInetuning/LLM-FINETUNING
export HF_ENDPOINT=https://hf-mirror.com
bash download_qwen3_4b.sh
```

模型默认保存到：

```text
/home/seuauto/Codes/LLM-FInetuning/LLM-FINETUNING/models/Qwen3-4B-Instruct-2507
```

该命令下载完整模型快照，包括：

- 所有 `.safetensors` 权重和权重索引；
- `config.json`；
- tokenizer 配置、词表和特殊 token；
- `generation_config.json`；
- 模型仓库中的其他必要文件。

下载完成后，将正式和测试 YAML 中的模型地址改为：

```yaml
model_name_or_path: /home/seuauto/Codes/LLM-FInetuning/LLM-FINETUNING/models/Qwen3-4B-Instruct-2507
```

## 完全离线的训练机

如果训练机不能联网，在另一台能联网的 Linux 机器上执行：

```bash
python3 -m pip install --upgrade huggingface_hub
mkdir -p Qwen3-4B-Instruct-2507
hf download Qwen/Qwen3-4B-Instruct-2507 \
  --local-dir ./Qwen3-4B-Instruct-2507
tar -czf Qwen3-4B-Instruct-2507.tar.gz Qwen3-4B-Instruct-2507
```

将 `Qwen3-4B-Instruct-2507.tar.gz` 传到训练机后执行：

```bash
mkdir -p /home/seuauto/Codes/LLM-FInetuning/LLM-FINETUNING/models
tar -xzf Qwen3-4B-Instruct-2507.tar.gz \
  -C /home/seuauto/Codes/LLM-FInetuning/LLM-FINETUNING/models
```

检查文件：

```bash
MODEL_DIR=/home/seuauto/Codes/LLM-FInetuning/LLM-FINETUNING/models/Qwen3-4B-Instruct-2507
test -f "${MODEL_DIR}/config.json"
test -f "${MODEL_DIR}/tokenizer_config.json"
find "${MODEL_DIR}" -maxdepth 1 -type f -name '*.safetensors' -print
```

最后启动小规模测试：

```bash
cd /home/seuauto/Codes/LLM-FInetuning/LLM-FINETUNING
TENSORBOARD_LOGGING_DIR=saves/logs/qwen3-4b/lora/sft/exp1 \
llamafactory-cli train examples/train_lora/qwen3_lora_sft_test.yaml
```

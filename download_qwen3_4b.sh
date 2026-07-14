#!/usr/bin/env bash

# 手动下载 Qwen3-4B-Instruct-2507 的完整模型仓库。
# 包含模型权重、config、tokenizer 和 generation config。
#
# 默认保存位置：
# /home/seuauto/Codes/LLM-FInetuning/LLM-FINETUNING/models/Qwen3-4B-Instruct-2507

set -euo pipefail

REPO_ROOT="/home/seuauto/Codes/LLM-FInetuning/LLM-FINETUNING"
MODEL_ID="Qwen/Qwen3-4B-Instruct-2507"
MODEL_DIR="${MODEL_DIR:-${REPO_ROOT}/models/Qwen3-4B-Instruct-2507}"

# 安装提供 `hf download` 命令的官方客户端。
python3 -m pip install --upgrade huggingface_hub

# 创建本地模型目录。
mkdir -p "${MODEL_DIR}"

# 下载整个模型仓库。不要添加 include 过滤，否则可能漏掉 tokenizer 或权重分片。
# 中国大陆无法连接 huggingface.co 时，可在运行脚本前设置：
# export HF_ENDPOINT=https://hf-mirror.com
hf download "${MODEL_ID}" \
    --local-dir "${MODEL_DIR}"

# 检查训练必需的主要文件。权重可能是单文件，也可能是多个分片。
test -f "${MODEL_DIR}/config.json"
test -f "${MODEL_DIR}/tokenizer_config.json"

if ! find "${MODEL_DIR}" -maxdepth 1 -type f -name '*.safetensors' -print -quit | grep -q .; then
    echo "错误：${MODEL_DIR} 中没有找到 safetensors 模型权重。" >&2
    exit 1
fi

echo
echo "下载完成：${MODEL_DIR}"
echo
echo "请在正式和测试 YAML 中使用本地路径："
echo "model_name_or_path: ${MODEL_DIR}"


# LLaMA Factory 微调配置参数指南

本文只整理研究人员在实际微调中会随**模型、数据、显存、训练阶段或评测方式**调整的参数。参数以当前仓库源码为准：

```python
_TRAIN_ARGS = [
    ModelArguments,
    DataArguments,
    TrainingArguments,
    FinetuningArguments,
    GeneratingArguments,
]
```

不包含部署引擎、模型导出、Ray、Profiler、SwanLab 细节、KTransformers、GaLore、APOLLO、BAdam、特殊 Token 语义初始化等低频或实验性配置。

> YAML 中的 `### model`、`### train` 等标题只是注释。参数根据**字段名**存入对应的 Arguments 对象，与它写在 YAML 的哪个位置无关。

## 1. 参数如何进入五类 Arguments

训练入口调用 `get_train_args()`，返回：

```python
model_args, data_args, training_args, finetuning_args, generating_args
```

处理顺序如下：

```text
YAML + 命令行覆盖
        ↓
OmegaConf 合并为一个 dict
        ↓
HfArgumentParser 按字段名构造五个 dataclass
        ↓
各 dataclass.__post_init__() 做转换和自身校验
        ↓
get_train_args() 做跨对象校验并写入派生值
```

命令行可以覆盖 YAML：

```bash
llamafactory-cli train config.yaml learning_rate=2e-5 cutoff_len=4096
```

未写入 YAML 的参数使用 dataclass 默认值。未知字段默认报错；不要常态化设置 `ALLOW_EXTRA_ARGS=1`，否则拼写错误可能被忽略。

---

## 2. ModelArguments：模型加载、量化和计算优化

### 2.1 常用参数

| 参数 | 声明默认值 | 典型值 | 何时调整 |
|---|---:|---|---|
| `model_name_or_path` | `None`，实际必填 | Hub ID 或本地目录 | 每个实验必须指定基础模型 |
| `adapter_name_or_path` | `None` | LoRA 目录；多个目录用逗号分隔 | 继续训练、推理或合并已有 LoRA |
| `cache_dir` | `None` | 大容量磁盘目录 | 默认缓存盘空间不足时 |
| `model_revision` | `main` | tag、branch、commit | 固定实验所用模型版本 |
| `trust_remote_code` | `False` | `True` | 模型仓库依赖自定义建模代码时；仅用于可信仓库 |
| `use_fast_tokenizer` | `True` | 通常保持 `True` | 模型 fast tokenizer 不兼容时关闭 |
| `flash_attn` | `auto` | `auto` / `fa2` / `sdpa` / `disabled` | 长序列或显存紧张时优先尝试 `fa2`，前提是硬件和依赖支持 |
| `disable_gradient_checkpointing` | `False` | 显存紧张保持 `False` | 显存充足、追求速度时设 `True`；本项目默认会启用梯度检查点 |
| `use_reentrant_gc` | `True` | `True` / `False` | 某些模型或编译后端与重入式检查点不兼容时改为 `False` |
| `upcast_layernorm` | `False` | 量化训练常设 `True` | 4/8-bit 训练数值不稳定时 |
| `resize_vocab` | `False` | `True` | 给 tokenizer 增加 token 并需要扩展 embedding 时 |
| `add_tokens` | `None` | `token1,token2` | 增加普通 token |
| `add_special_tokens` | `None` | `<tag1>,<tag2>` | 增加特殊 token |
| `train_from_scratch` | `False` | 通常 `False` | 真正从随机权重预训练时才开启 |

`adapter_name_or_path`、`add_tokens`、`add_special_tokens` 最终会从逗号分隔字符串转为列表。

### 2.2 QLoRA/量化参数

| 参数 | 默认值 | 典型值 | 说明 |
|---|---:|---|---|
| `quantization_method` | `bnb` | `bnb` | 在线 4/8-bit 量化通常使用 bitsandbytes |
| `quantization_bit` | `None` | `4` | QLoRA 常用 4 bit；显存较宽裕可用 8 bit；不量化保持 `None` |
| `quantization_type` | `nf4` | `nf4` | 4-bit 训练通常优先 NF4 |
| `double_quantization` | `True` | `True` | 进一步压缩量化元数据，通常保持开启 |
| `quantization_device_map` | `None` | 推理时可用 `auto` | 训练时禁止设为 `auto` |

### 2.3 多模态场景才需要调整

| 参数 | 默认值 | 使用场景 |
|---|---:|---|
| `image_max_pixels` | `768 × 768` | 图像过大导致显存不足时降低；细节任务可提高 |
| `image_min_pixels` | `32 × 32` | 控制过小图像的最低处理尺寸 |
| `video_max_pixels` | `256 × 256` | 视频显存不足时降低 |
| `video_fps` | `2.0` | 动作变化快时提高；长视频或显存不足时降低 |
| `video_maxlen` | `128` | 限制最大采样帧数 |
| `audio_sampling_rate` | `16000` | 必须与音频模型预期采样率一致 |

要求：`image_max_pixels >= image_min_pixels`，`video_max_pixels >= video_min_pixels`。

### 2.4 训练后自动写入的字段

以下字段不是用户配置，而是 `get_train_args()` 根据其他对象计算：

```python
model_args.compute_dtype = torch.bfloat16  # 来自 bf16/pure_bf16
model_args.device_map = {"": current_device}
model_args.model_max_length = data_args.cutoff_len
model_args.block_diag_attn = data_args.neat_packing
```

---

## 3. DataArguments：数据集、模板和序列构造

### 3.1 常用参数

| 参数 | 默认值 | 典型值 | 何时调整 |
|---|---:|---|---|
| `dataset` | `None` | `dataset_a,dataset_b` | 训练时必填；多个数据集用逗号分隔 |
| `eval_dataset` | `None` | 单独验证集名称 | 已准备独立验证集时使用 |
| `dataset_dir` | `data` | 数据配置所在目录 | 数据集注册文件不在默认目录时 |
| `media_dir` | `dataset_dir` | 图片/视频/音频根目录 | 多模态文件与数据配置不在同一目录时 |
| `template` | `None` | `qwen3_nothink`、`llama3` | 必须与模型聊天格式匹配 |
| `cutoff_len` | `2048` | `2048` / `4096` / `8192` | 根据样本长度、模型上下文和显存调整 |
| `max_samples` | `None` | `1000` | 调试或小规模试跑；正式训练通常不设 |
| `preprocessing_num_workers` | `None` | CPU 核数的约 1/2～1 倍 | 分词预处理太慢时增加，内存不足时降低 |
| `preprocessing_batch_size` | `1000` | `100`～`1000` | 预处理内存溢出时降低 |
| `overwrite_cache` | `False` | `True` | 数据或模板变化后需要强制重建缓存时 |
| `tokenized_path` | `None` | 本地缓存目录 | 重复实验复用预分词数据时 |

`dataset` 和 `eval_dataset` 构造后会变成列表：

```yaml
dataset: identity,alpaca_en_demo
```

```python
data_args.dataset == ["identity", "alpaca_en_demo"]
```

### 3.2 验证集参数

| 参数 | 默认值 | 典型值 | 说明 |
|---|---:|---|---|
| `val_size` | `0.0` | `0.05`、`0.1` 或整数 | 未提供 `eval_dataset` 时，从训练集划分验证集 |
| `eval_num_beams` | `None` | `1`～`4` | 生成式验证使用的 beam 数；越大越慢 |
| `eval_on_each_dataset` | `False` | `True` | 多个验证集需要分别报告指标时 |

限制：

- `eval_dataset` 和正数 `val_size` 不能同时设置。
- `dataset=None` 时不能设置正数 `val_size`。
- streaming 模式下 `val_size` 必须是整数条数，不能用 `0.1` 之类比例。

### 3.3 Prompt、历史消息和思考模式

| 参数 | 默认值 | 适用场景 |
|---|---:|---|
| `train_on_prompt` | `False` | 希望 prompt token 也计算 loss 时开启 |
| `mask_history` | `False` | 多轮对话只训练最后一轮回答时开启 |
| `default_system` | `None` | 覆盖模板默认 system prompt |
| `enable_thinking` | `True` | 推理模型是否启用 thinking 模式 |
| `preserve_thinking` | `False` | 是否保留历史轮次中的思考内容 |

限制：

- `train_on_prompt` 和 `mask_history` 不能同时为 `True`。
- `train_on_prompt`、`mask_history` 仅允许 SFT 阶段。
- thinking 设置必须与所选模板和训练数据格式一致。

### 3.4 Packing 与长文本

| 参数 | 默认值 | 典型值 | 说明 |
|---|---:|---|---|
| `packing` | `None` | 短样本训练可设 `True` | 将多个短样本拼入一个序列，提高 token 利用率；PT 阶段未指定时自动开启 |
| `neat_packing` | `False` | `True` | 使用块对角注意力隔离拼接样本；仅支持 SFT |
| `streaming` | `False` | 超大或远程数据集设 `True` | 不将完整数据集加载到本地内存 |
| `buffer_size` | `16384` | 数千至数万 | streaming 随机打乱缓冲区大小 |

注意：

- `neat_packing=True` 会自动令 `packing=True`。
- 显式 `packing=True` 时，`DataArguments.__post_init__()` 会让 `cutoff_len -= 1`。
- `streaming=True` 时不能设置 `max_samples`，并且必须在 `TrainingArguments` 中设置正数 `max_steps`。

### 3.5 多数据集混合

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `mix_strategy` | `concat` | 可选 `concat`、`interleave_under`、`interleave_over`、`interleave_once` |
| `interleave_probs` | `None` | 如 `0.7,0.3`，仅用于 interleave 策略 |

`interleave_probs` 的数量必须与数据集数量一致，且不能配合 `mix_strategy=concat`。

---

## 4. TrainingArguments：优化、batch、精度、日志和检查点

`TrainingArguments` 继承 `transformers.Seq2SeqTrainingArguments`。因此部分默认值会随安装的 Transformers 版本变化；下表给出当前配置体系中的常见起点，而不是对所有模型都最优的结论。

### 4.1 训练规模与优化器

| 参数 | 常见默认值 | 典型值/建议 |
|---|---:|---|
| `do_train` | `False` | 训练配置必须设 `True` |
| `per_device_train_batch_size` | `8` | 大模型常设 `1`～`4`，以不 OOM 为准 |
| `gradient_accumulation_steps` | `1` | 用于补足目标有效 batch size，常见 `4`～`32` |
| `learning_rate` | `5e-5` | LoRA：`1e-4`～`2e-4`；全量/Freeze：`1e-5`～`5e-5`；偏好优化通常更低 |
| `num_train_epochs` | `3.0` | 常见 `1`～`3`；小数据需重点防止过拟合 |
| `max_steps` | `-1` | 指定后覆盖 epoch；streaming 必须设置正数 |
| `lr_scheduler_type` | `linear` | SFT 常用 `cosine` 或 `linear` |
| `warmup_ratio` | `0.0` | 常用 `0.03`～`0.1` |
| `weight_decay` | `0.0` | 全量训练常尝试 `0.01`；LoRA 常用 `0` |
| `max_grad_norm` | `1.0` | 通常保持 `1.0`；梯度不稳定时重点观察 |
| `seed` | `42` | 对比实验保持一致，复现实验需记录 |

有效 batch size：

```text
per_device_train_batch_size
× gradient_accumulation_steps
× 数据并行进程数（通常等于 GPU 数）
```

例如 4 卡、每卡 batch 2、累积 8 次：有效 batch size 为 `2 × 8 × 4 = 64`。

### 4.2 精度与显存

| 参数 | 默认值 | 典型值/限制 |
|---|---:|---|
| `bf16` | `False` | 支持 BF16 的 GPU 优先使用 |
| `fp16` | `False` | 不支持 BF16 的旧 GPU 可使用 |
| `pure_bf16` | `False` | 属于 `FinetuningArguments`；不用 AMP，部分低秩优化场景使用 |
| `fp8` | `False` | 需要相应 GPU 和软件栈；不能与 4/8-bit 模型量化同时使用 |
| `deepspeed` | `None` | 多卡全量训练或 ZeRO 优化时填写 DeepSpeed JSON 路径 |

建议不要同时开启 `bf16` 和 `fp16`。未开启任何混合精度时，代码会给出显存/效率警告。

### 4.3 日志、验证与保存

| 参数 | 常见默认值 | 典型值/说明 |
|---|---:|---|
| `output_dir` | 版本相关 | 每个实验使用独立目录 |
| `logging_steps` | `500` | 小实验常用 `5`～`20`；大实验可增大 |
| `save_strategy` | `steps` | 可按 `steps` 或 `epoch` 保存 |
| `save_steps` | `500` | 与总更新步数匹配，避免一次检查点都存不到 |
| `save_total_limit` | `None` | 磁盘有限时常设 `2`～`5` |
| `save_only_model` | `False` | 要断点续训必须保持 `False` |
| `eval_strategy` | `no` | 有验证集时设 `steps` 或 `epoch` |
| `eval_steps` | 继承 `logging_steps` | 使用 steps 验证时按成本调整 |
| `load_best_model_at_end` | `False` | 需要自动保留最佳模型时开启 |
| `metric_for_best_model` | `None` | 如 `eval_loss`；必须对应实际产生的指标名 |
| `greater_is_better` | 自动推断 | loss 类指标应为 `False` |
| `report_to` | 版本相关 | `none`、`wandb`、`tensorboard`、`swanlab`、`mlflow` |
| `plot_loss` | `False` | 属于 `FinetuningArguments`；需要训练结束后输出 loss 图时开启 |

开启 `load_best_model_at_end` 时，通常要求保存策略与验证策略一致；steps 模式下还应让 `save_steps` 与 `eval_steps` 合理对齐。

### 4.4 数据加载与断点续训

| 参数 | 默认值 | 典型值/说明 |
|---|---:|---|
| `dataloader_num_workers` | `0` | 常设 `2`～`8`；CPU/内存不足时降低 |
| `dataloader_pin_memory` | `True` | GPU 训练通常保持开启 |
| `resume_from_checkpoint` | `None` | 指定 `checkpoint-N`；若输出目录已有检查点，代码可能自动选最后一个 |
| `overwrite_output_dir` | `False` | 明确开始全新实验且允许使用已有目录时设 `True` |
| `ddp_timeout` | 常见 `1800` 秒 | 多卡加载模型很慢或节点启动不同步时增大 |

如果输出目录已有模型文件、没有可恢复检查点，且 `overwrite_output_dir=False`，程序会拒绝覆盖。

---

## 5. FinetuningArguments：训练阶段和参数高效微调

### 5.1 训练阶段

| 参数 | 默认值 | 可选值 | 使用场景 |
|---|---:|---|---|
| `stage` | `sft` | `pt` | 继续预训练/领域自回归语料 |
|  |  | `sft` | 指令微调、对话微调 |
|  |  | `rm` | 奖励模型训练 |
|  |  | `dpo` | 成对偏好训练 |
|  |  | `kto` | 非成对 chosen/rejected 反馈训练 |
|  |  | `ppo` | 在线 RLHF，需要奖励模型，配置和资源成本较高 |

### 5.2 微调方式

| 参数 | 默认值 | 典型选择 |
|---|---:|---|
| `finetuning_type` | `lora` | 单卡/少卡优先 LoRA；资源充足并追求全参数适配时用 `full`；`freeze` 用于只训练部分层；`oft` 仅在明确需要 OFT 时使用 |

#### LoRA 参数

| 参数 | 默认值 | 典型值/影响 |
|---|---:|---|
| `lora_rank` | `8` | 常用 `8`、`16`、`32`、`64`；越大容量和显存占用越高 |
| `lora_alpha` | `None` | 未指定时自动设为 `2 × lora_rank` |
| `lora_dropout` | `0.0` | 小数据或过拟合时尝试 `0.05`～`0.1` |
| `lora_target` | `all` | 通常保持 `all`；需严格控制参数量时指定 `q_proj,v_proj` 等模块 |
| `additional_target` | `None` | 需要额外训练并保存 embedding、lm_head 等非 LoRA 模块时指定 |
| `use_rslora` | `False` | 较高 rank 下可尝试稳定缩放 |
| `use_dora` | `False` | 明确评估 DoRA 时开启，计算和兼容性成本更高 |
| `create_new_adapter` | `False` | 已加载 adapter，但希望在其上另建新 adapter 时使用；量化模型禁止这种组合 |

`lora_target` 和 `additional_target` 构造后会从逗号字符串转成列表。

#### Freeze 参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `freeze_trainable_layers` | `2` | 正数训练最后 N 层；负数训练最前 N 层 |
| `freeze_trainable_modules` | `all` | 指定这些层中哪些模块可训练 |
| `freeze_extra_modules` | `None` | 额外训练 embedding、lm_head 等模块 |

### 5.3 多模态冻结策略

| 参数 | 默认值 | 使用场景 |
|---|---:|---|
| `freeze_vision_tower` | `True` | 默认不更新视觉编码器；视觉域差异大时可关闭 |
| `freeze_multi_modal_projector` | `True` | 需要适配视觉-语言对齐层时设 `False` |
| `freeze_language_model` | `False` | 只训练视觉侧或 projector 时设 `True` |

### 5.4 DPO/KTO 常用参数

| 参数 | 默认值 | 典型值/说明 |
|---|---:|---|
| `pref_beta` | `0.1` | 偏好强度/KL 约束尺度，常从 `0.05`～`0.5` 搜索 |
| `pref_loss` | `sigmoid` | 常规 DPO 用 `sigmoid`；也支持 `hinge`、`ipo`、`kto_pair`、`orpo`、`simpo` |
| `dpo_label_smoothing` | `0.0` | cDPO 可用小正数；仅对 `sigmoid` loss 有效 |
| `pref_ftx` | `0.0` | DPO 中混入 SFT loss 的系数，防止能力漂移时使用 |
| `ref_model` | `None` | DPO 参考模型；不指定时流程可能使用当前策略模型副本，评测时建议显式指定 |
| `ref_model_quantization_bit` | `None` | 显存紧张时用 `4` 或 `8` |
| `kto_chosen_weight` | `1.0` | KTO chosen 样本损失权重 |
| `kto_rejected_weight` | `1.0` | KTO rejected 样本损失权重 |

ORPO、SimPO 不需要 reference model；其他 DPO loss 通常需要。

### 5.5 PPO 最小必要参数

只有明确进行 PPO 时才需要：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `reward_model` | `None` | PPO 必填 |
| `reward_model_type` | `lora` | 奖励模型加载方式 |
| `reward_model_adapters` | `None` | 奖励模型 LoRA 路径 |
| `reward_model_quantization_bit` | `None` | 可设 `4` 或 `8` 降低显存 |
| `ppo_epochs` | `4` | 每批经验的优化轮数 |
| `ppo_target` | `6.0` | 自适应 KL 控制目标 |

PPO 只允许 WandB、TensorBoard、Trackio 或 none 日志后端；不支持只做 evaluation。

---

## 6. GeneratingArguments：生成式验证和推理参数

这些参数主要影响 CLI/API 推理以及 `predict_with_generate=True` 的生成式验证，不影响普通 teacher-forcing SFT loss。

| 参数 | 默认值 | 典型值/说明 |
|---|---:|---|
| `do_sample` | `True` | 可复现评测通常设 `False`；开放式生成设 `True` |
| `temperature` | `0.95` | 事实/代码任务常用 `0.1`～`0.7`；创意任务可提高 |
| `top_p` | `0.7` | 常用 `0.7`～`0.95` |
| `top_k` | `50` | 常用 `20`～`50`；不需要时按模型生成配置调整 |
| `num_beams` | `1` | 确定性序列任务可尝试 `2`～`4`，但速度和显存成本增加 |
| `max_new_tokens` | `1024` | 按目标回答长度和显存调整 |
| `max_length` | `1024` | 只有 `max_new_tokens <= 0` 时才生效 |
| `repetition_penalty` | `1.0` | 输出重复时尝试 `1.05`～`1.2` |
| `length_penalty` | `1.0` | 主要用于 beam search |

`max_new_tokens > 0` 时，转换为生成配置会删除 `max_length`，因此默认真正控制生成长度的是 `max_new_tokens`。

训练中的生成式预测还涉及 `TrainingArguments.predict_with_generate`；它只能用于 SFT，并且不能与 `compute_accuracy=True` 或 DeepSpeed ZeRO-3 同时使用。

---

## 7. 五类参数之间的关键限制

### 7.1 训练阶段矩阵

| 配置 | PT | SFT | RM | DPO/KTO | PPO |
|---|---:|---:|---:|---:|---:|
| `packing` | 默认开启 | 可选 | 谨慎/通常关闭 | 谨慎/通常关闭 | 通常关闭 |
| `neat_packing` | 不允许 | 允许 | 不允许 | 不允许 | 不允许 |
| `train_on_prompt` | 不允许 | 允许 | 不允许 | 不允许 | 不允许 |
| `mask_history` | 不允许 | 允许 | 不允许 | 不允许 | 不允许 |
| `predict_with_generate` | 不允许 | 允许 | 不允许 | 不允许 | 不允许 |
| `load_best_model_at_end` | 可用 | 可用 | 不支持 | 可用 | 不支持 |

### 7.2 量化、微调方式和后端

| 组合 | 是否允许 | 原因/处理 |
|---|---|---|
| 4/8-bit + LoRA | 允许 | 典型 QLoRA |
| 4/8-bit + OFT | 允许 | 源码允许 |
| 4/8-bit + Full/Freeze | 不允许 | 在线量化训练仅支持 LoRA/OFT |
| 量化 + `resize_vocab` | 不允许 | 不能调整量化 embedding 层 |
| 量化 + 多个 adapter | 不允许 | 先离线合并 adapter |
| 量化 + `create_new_adapter` 且已加载 adapter | 不允许 | 量化模型上禁止这种叠加 |
| FP8 + 4/8-bit 量化 | 不允许 | 两种低精度机制冲突 |
| LoRA + GaLore/APOLLO/BAdam | 不允许 | 优化方式冲突 |
| 训练 + `infer_backend=vllm/sglang` | 不允许 | 训练仅使用 HF backend |

### 7.3 DeepSpeed/分布式

- DeepSpeed 必须通过 `llamafactory-cli`/`torchrun` 以分布式方式启动；单卡也可用 `FORCE_TORCHRUN=1`。
- DeepSpeed ZeRO-3 不兼容 `predict_with_generate`、`pure_bf16`、PiSSA 初始化、Unsloth、KTransformers。
- 普通 DDP + LoRA 且用户未指定时，代码自动设置 `ddp_find_unused_parameters=False`。
- RM/PPO 的 full/freeze 训练不能从普通 Trainer checkpoint 恢复，解析器会清除 `resume_from_checkpoint`。

### 7.4 验证与预测

- `do_train=True` 必须提供 `dataset`。
- `do_eval=True`、`do_predict=True` 或 `predict_with_generate=True` 时，必须提供 `eval_dataset` 或正数 `val_size`。
- SFT 中 `do_predict=True` 时必须同时设置 `predict_with_generate=True`，否则无法保存生成结果。
- `predict_with_generate=True` 与 token-level `compute_accuracy=True` 互斥。

---

## 8. 三种常见配置起点

以下只给参数关系，不代替针对模型和数据的实验搜索。

### 8.1 单卡 LoRA SFT

```yaml
model_name_or_path: Qwen/Qwen3-4B-Instruct-2507
trust_remote_code: true

stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all

dataset: your_dataset
template: qwen3_nothink
cutoff_len: 2048
preprocessing_num_workers: 8

output_dir: saves/qwen3-4b/lora/sft
overwrite_output_dir: true
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
logging_steps: 10
save_steps: 500
report_to: none
```

### 8.2 显存紧张的 QLoRA SFT

```yaml
model_name_or_path: your/model
quantization_method: bnb
quantization_bit: 4
quantization_type: nf4
double_quantization: true
upcast_layernorm: true

stage: sft
do_train: true
finetuning_type: lora
lora_rank: 16
lora_target: all

dataset: your_dataset
template: model_template
cutoff_len: 2048

output_dir: saves/model/qlora/sft
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 1.0e-4
num_train_epochs: 3.0
bf16: true
```

### 8.3 带验证集和最佳模型选择的 SFT

```yaml
stage: sft
do_train: true
do_eval: true

dataset: your_train_dataset
eval_dataset: your_eval_dataset

eval_strategy: steps
eval_steps: 100
save_strategy: steps
save_steps: 100
load_best_model_at_end: true
metric_for_best_model: eval_loss
greater_is_better: false
save_total_limit: 3
```

---

## 9. 配置前的最小检查清单

1. `model_name_or_path` 与 `template` 是否匹配。
2. `stage` 是否匹配数据格式：SFT、偏好对还是无配对反馈。
3. `finetuning_type` 是否符合显存预算；量化训练是否只配 LoRA/OFT。
4. `cutoff_len` 是否覆盖主要样本，同时不会造成 OOM。
5. 有效 batch size 是否按 GPU 数和梯度累积正确计算。
6. 学习率是否与 LoRA/全量/偏好训练类型匹配。
7. 需要验证时，是否提供 `eval_dataset` 或 `val_size`，并对齐 eval/save 策略。
8. 需要断点续训时，是否保留优化器状态，即不要设置 `save_only_model=True`。
9. BF16/FP16、4-bit、FP8、DeepSpeed 组合是否满足上述限制。
10. 正式训练前先用较小 `max_samples` 跑通数据、loss、保存和推理流程。

# Evo-1 当前修改总结（2026-03-29）

## 1. 总体目标
本轮修改围绕以下优先级推进：
- 训练稳定性优先
- 训练速度其次
- 显存占用再次
- 数据缓存避免大规模碎片化文件
- 训练流程与 `ms-swift` 常见工程实践对齐（尤其是分布式/Deepspeed 可配置性）

## 2. 核心改动

### 2.1 训练稳定性与日志增强
涉及文件：
- `scripts/train.py`

主要改动：
- 新增全局随机种子控制（按 `seed + process_index`）。
- 新增非有限值（NaN/Inf）检测、诊断文件落盘、连续异常步数中止机制。
- `loss` 使用更稳的 `float32` 计算路径。
- 增加梯度累积控制（`--gradient_accumulation_steps`）。
- 训练日志新增吞吐与显存指标。
- 指标上报支持 `swanlab_project`。
- 新增可选 `fused AdamW`，支持自动能力探测与回退。
- 分布式下 `loss` 聚合为全局均值再用于 best-loss 判断与 checkpoint 元数据。

新增/重要参数：
- `--seed`
- `--gradient_accumulation_steps`
- `--non_finite_max_streak`
- `--fused_adamw` / `--no_fused_adamw`
- `--swanlab_project`
- `--cache_format`
- `--cache_shard_size_mb`
- `--gradient_checkpointing_use_reentrant`

### 2.2 训练过程中展示 s/it 与 it/s
涉及文件：
- `scripts/train.py`

主要改动：
- 在 step 日志中输出：`s/it`、`it/s`、`samples/s`。
- SwanLab/W&B payload 同步新增 `it_per_sec`。

日志示例（实际运行）：
- `Speed: 1.8650 s/it | 0.54 it/s | 1.07 samples/s`

### 2.3 数据缓存从碎片化文件改为分片索引格式
涉及文件：
- `dataset/exporter.py`
- `dataset/indexed_cached_dataset.py`（新增）
- `dataset/build.py`
- `dataset/__init__.py`
- `dataset/lerobot_dataset_pretrain_mp.py`

主要改动：
- 新增 `indexed_v2` 缓存格式：
  - `data-xxxxx.bin` 分片
  - `index.json` 索引
  - `manifest.json` 元信息
- 读取端新增内存映射读取，避免海量小文件 I/O。
- `build_dataset` 根据 `manifest schema_version` 自动选择 legacy 或 indexed_v2 数据集。
- 导出阶段新增 `episode_cache_mode=index`，避免生成大量 `episodes/*.pkl` 临时碎片文件。

### 2.4 分布式/Deepspeed 脚本能力增强（参考 ms-swift 习惯）
涉及文件：
- `scripts/train_lerobotv21_pipeline.sh`
- `config/deepspeed/zero2.json`（新增）
- `config/deepspeed/zero3.json`（新增）
- `config/deepspeed/zero2_offload.json`（新增）
- `config/deepspeed/zero3_offload.json`（新增）

主要改动：
- 支持 `DEEPSPEED_PRESET` 快速切换（`zero2/zero3/zero2_offload/zero3_offload`）。
- 支持多机参数透传：`MACHINE_RANK`、`MAIN_PROCESS_IP`、`MAIN_PROCESS_PORT`、`RDZV_*`。
- 支持 `STAGE2_TARGET_GLOBAL_BATCH` 自动换算梯度累积。
- 导出 NCCL 相关稳态环境变量（默认启用）。
- stage2 支持随机初始化和从 stage1 恢复两种模式。
- 日志中打印关键运行配置，便于复现实验。

### 2.5 模型配置与 checkpointing 选项打通
涉及文件：
- `model/config.py`
- `model/evo1_model.py`
- `model/internvl3/internvl3_embedder.py`

主要改动：
- 新增并打通 `gradient_checkpointing_use_reentrant` 配置。
- 在可控路径中显式传递 checkpoint kwargs（支持则使用，不支持则回退）。

### 2.6 按 README 风格且双阶段 batch=16 的训练脚本
涉及文件：
- `scripts/train_readme_batch16.sh`（新增）

主要特性：
- 固定 stage1/stage2 batch 都为 16。
- 实验名默认：`version_1.0_stage_12`。
- stage2 默认按 README 逻辑从 stage1 恢复（`STAGE2_INIT_FROM_SCRATCH=0`）。
- 自动检测 `dataset/config.yaml` 中路径是否可用。
- 若默认配置路径不可用，自动回退到 `dataset/config_lerobotv21_click_alarmclock.yaml`。
- 若显式传入 `DATASET_CONFIG` 且不可用，直接报清晰错误。

## 3. 最近验证结果

### 3.1 单元测试
命令：
- `python3 -m unittest discover -s tests -p 'test_*.py'`

结果：
- `Ran 11 tests`
- `OK (skipped=5)`

### 3.2 stage2 实测（40 steps）
配置摘要：
- `USE_DEEPSPEED=1`
- `STAGE2_STEPS=40`
- `STAGE2_BATCH_SIZE=2`
- `STAGE2_GRAD_ACCUM_STEPS=1`
- `STAGE2_INIT_FROM_SCRATCH=1`
- `CACHE_FORMAT=indexed_v2`

关键日志（`/root/evo1_runs/checkpoints/stage2/train_log_20260329_193154.log`）：
- Step 10: `0.3297 sec/step`, `6.07 samples/s`, `7.21 GiB`
- Step 20: `0.3256 sec/step`, `6.14 samples/s`, `7.21 GiB`
- Step 30: `0.3306 sec/step`, `6.05 samples/s`, `7.21 GiB`
- Step 40: `0.3321 sec/step`, `6.02 samples/s`, `7.21 GiB`

稳态结论（step10-40）：
- 约 `0.33 sec/step`
- 约 `6.0 ~ 6.1 samples/s`
- 峰值显存约 `7.21 GiB`

SwanLab run：
- `https://swanlab.cn/@mrblue0632/evo1-pro/runs/vfj8x5wxx4bx7k02khf86`

### 3.3 s/it 与 it/s 输出验证
配置摘要：
- `STAGE2_STEPS=5`
- `DISABLE_TQDM=0`

关键日志（`/root/evo1_runs/checkpoints/stage2/train_log_20260329_194231.log`）：
- `Speed: 1.8650 s/it | 0.54 it/s | 1.07 samples/s`

SwanLab run：
- `https://swanlab.cn/@mrblue0632/evo1-pro/runs/gsnh5drlm9nymecxq3sl3`

## 4. 已知事项
- `accelerate launch` 打印的 `--dynamo_backend was set to 'no'` 是提示，不是训练失败原因。
- 真实失败主因通常是数据配置路径不可用（例如 `dataset/config.yaml` 内遗留旧路径）。
- 仍可能出现来自上游依赖内部的 checkpoint `use_reentrant` 警告，不影响本项目已接管的训练主路径。

## 5. 当前推荐入口
- 双阶段 batch=16（README风格）：
  - `./scripts/train_readme_batch16.sh all`
- 仅验证环境和配置：
  - `SKIP_VIDEO_CHECK=1 ./scripts/train_readme_batch16.sh validate`
- 观察实时 `s/it`：
  - 设置 `DISABLE_TQDM=0`

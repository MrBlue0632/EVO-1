# Evo-1 修改说明

## 这次主要改了什么

### 1. model 工程化整理
- 新增 `Evo_1/model/config.py`，把模型配置的默认值补全、字段归一化、合法性校验集中管理。
- 新增 `Evo_1/model/evo1_model.py`，把 `EVO1` 的构建、推理、冻结逻辑从脚本式写法整理成稳定模块。
- 调整 `Evo_1/model/__init__.py` 和 `Evo_1/__init__.py`，改成懒加载导出，避免只导入配置时就拉起重依赖。
- 保留 `Evo_1/scripts/Evo1.py` 作为兼容入口，现有调用方式不需要改。

### 2. 数据集加载管线整理
- 新增 `Evo_1/dataset/build.py`，统一 dataset 和 dataloader 的构建入口。
- 新增 `Evo_1/dataset/utils.py`，抽出 padding、归一化、collate、cache 路径等通用逻辑。
- 重构 `Evo_1/dataset/lerobot_dataset_pretrain_mp.py`：
  - 去掉硬编码缓存路径
  - 将 stats 计算结果写入缓存目录，而不是源数据目录
  - 使用稳定 cache key 和原子写入
  - 去掉 `__getitem__` 中异常后随机换样本的行为，改成显式报错
  - 保持 batch key 与原训练流程兼容

### 3. 训练入口整理
- 重构 `Evo_1/scripts/train.py`，改为通过 `build_model`、`build_dataset`、`build_dataloader` 调用新模块。
- 保留原有 CLI 参数和训练主流程，不改模型参数语义。
- 让训练脚本和 model/dataset 的职责边界更清晰，后续更容易维护。

### 4. 现有文件上的顺手修正
- 保留并兼容了你工作区里 `flow_matching.py` 和 `internvl3_embedder.py` 已有的安全性/日志改动。
- 这些改动主要是：更规范的日志输出、更严格的 shape 检查、减少调试残留打印。

## 新增文件
- `Evo_1/__init__.py`
- `Evo_1/model/config.py`
- `Evo_1/model/evo1_model.py`
- `Evo_1/model/__init__.py`
- `Evo_1/dataset/build.py`
- `Evo_1/dataset/utils.py`
- `tests/test_model_config.py`
- `tests/test_dataset_utils.py`
- `engineering_update_report.md`

## 验证情况
- 已通过 `py_compile` 语法检查。
- 已运行 `python3 -m unittest discover -s tests -p 'test_*.py'`。
- 当前环境缺少 `torch`，因此依赖 `torch` 的 3 个 dataset utility 测试被跳过。

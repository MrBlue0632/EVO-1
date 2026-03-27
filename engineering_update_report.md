# Evo-1 工程化更新报告

## Git 历史背景
从最近的仓库历史看，项目的提交主要集中在 README 更新、一次 `transformers` 兼容性修复，以及 Aloha 支持的加入。这说明项目此前更偏向功能推进和文档补充，而训练、模型、数据加载这几块核心代码的工程化收口相对较少。

相关历史信号：
- `bdf044c fix transformers version`
- `613b56f Add support for Aloha`
- 之后多次 `update readme`

## 本次更新的重点
这次更新只关注工程质量，不涉及模型能力增强，也不调整超参数语义。

## 参考 ms-swift 的内容
这次改造参考了 ms-swift 的一些工程化思路，但没有照搬其完整框架。主要借鉴点如下：

- **配置归一化与构建边界**
  参考 ms-swift 将参数整理、默认值补全、合法性校验放到独立层的做法，把 Evo-1 的模型配置处理从模型主体里拆出来，放到 `Evo_1/model/config.py` 中集中管理。

- **模块职责拆分**
  参考 ms-swift 将 model、dataset、dataloader、train pipeline 分层组织的方式，对 Evo-1 做了最小化拆分：
  - `model` 负责模型定义和构建
  - `dataset` 负责数据集构建、cache 与 collate
  - `train.py` 只保留训练流程控制

- **稳定入口与包导出**
  参考 ms-swift 统一入口、降低脚本和内部实现耦合的做法，为 Evo-1 增加了 `build_model`、`build_dataset`、`build_dataloader` 这类稳定边界，并对包导出做了整理。

- **数据集管线的可维护性**
  参考 ms-swift 在 dataset 侧强调可配置、可复用、少副作用的思路，对 Evo-1 的 LeRobot 数据管线进行了收口，包括 cache 目录管理、stats 读取与缓存、collate 抽离、异常处理显式化等。

- **测试与验证思路**
  参考 ms-swift 会为工具函数、配置层、训练相关边界补最小测试覆盖的思路，这次给 Evo-1 增加了配置层和 dataset utility 的基础测试。

需要说明的是，本次**没有**参考或迁移 ms-swift 的以下内容：
- 没有引入 ms-swift 的完整训练框架
- 没有引入其参数体系、CLI 框架或模型注册全量机制
- 没有修改 Evo-1 现有模型超参数语义

换句话说，这次参考 ms-swift 的重点是“工程组织方式”，不是“框架级迁移”。

### 1. 模型代码规范化
- 新增 `Evo_1/model/config.py`，集中处理模型配置的默认值补全、字段归一化和合法性校验。
- 新增 `Evo_1/model/evo1_model.py`，将 Evo-1 的模型构建、推理和冻结逻辑从脚本式组织整理为稳定模块。
- 保留现有配置语义，同时移除了模型构造阶段对外部 `config` 的原地污染式修改。
- 保留 `Evo_1/scripts/Evo1.py` 作为兼容入口，避免现有调用方式失效。
- 将包导出改为懒加载，避免仅导入配置模块时就触发重依赖加载。

### 2. 数据集加载管线加固
- 新增 `Evo_1/dataset/build.py`，统一 dataset 与 dataloader 的构建入口。
- 新增 `Evo_1/dataset/utils.py`，沉淀 padding、归一化、collate、cache 路径等通用逻辑。
- 重构 `Evo_1/dataset/lerobot_dataset_pretrain_mp.py`，重点包括：
  - 去除硬编码缓存路径
  - 将统计信息写入缓存目录，而不是回写源数据目录
  - 使用稳定 cache key，降低旧缓存污染风险
  - 使用原子写入，降低中断时缓存损坏概率
  - 对坏 cache、坏视频、缺失 state/action 的样本改为显式失败，不再随机跳样本
  - 保持与原训练流程兼容的 batch key 输出

### 3. 训练入口收口
- 重构 `Evo_1/scripts/train.py`，改为通过 `build_model`、`build_dataset`、`build_dataloader` 调用新边界。
- 降低训练脚本与模型、数据集实现细节的直接耦合。
- 保留原有 CLI 参数和整体训练流程，避免破坏现有使用方式。

## 兼容性与风险
- 本次没有有意修改模型超参数语义。
- 现有 `train.py` 的 CLI 参数仍然保留。
- 现有数据集配置结构继续有效，仅新增了可选的 cache 控制能力。
- 本次最明显的行为变化是：数据集异常处理更严格。以前坏样本可能被静默跳过并替换成随机样本，现在会直接报错，便于定位数据问题，但也意味着脏数据会更早暴露出来。

## 验证情况
- `python3 -m py_compile` 已通过，说明重构后的包和脚本文件语法正常。
- `python3 -m unittest discover -s tests -p test_*.py` 已运行通过。
- 当前系统 Python 环境没有安装 `torch`，因此依赖 `torch` 的 dataset utility 测试被显式跳过，没有在本机完成这部分运行时验证。

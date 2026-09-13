# QCal Agent

QCal Agent 是面向超导量子单比特自动校准的完整第一版工程。一个仓库同时包含：

- 可审计的 Agent 状态机、严格工具协议、预算和回滚；
- 独立的实验采集/分析/绘图层，可由模拟后端切换到实验室适配器；
- S21、Power2D、ZPA2D、能谱、Rabi、Ramsey、T1、Echo、IQraw 和单比特 XEB；
- 数据集生成、泄漏审计、Nanbeige4.2-3B LoRA 微调、离线评测和闭环评测；
- 内容寻址原始数据、拟合质量门控和留出 IQ shot 验收。

本仓库从产品层面直接以 `1.0.0` 发布，不要求用户理解历史实验版本。旧数据和旧模型只用于内部对照，不进入第一版命名、默认配置或发布说明。

## 安全与科研边界

默认运行的是物理启发模拟器，不会连接真实仪器。语言模型只选择注册实验和有界参数，不拟合原始数组、不执行任意 Python/Shell，也不能绕过控制器判定成功。Agent 真机接入通过 `RegisteredHardwareBackend` 显式注册每个采集函数，并要求逐次操作确认；部署时仍须补齐设备通道映射、硬件限幅、斜率限制、急停、独占锁和监督干跑证据。

单比特 XEB 是随机线路保真度衰减实验，不等价于 Google 多比特随机线路采样。当前没有实现耦合器、`generate_coupler` 或双比特门；ZPA2D 是单比特 Z/磁通偏置与读出频率二维扫描，不依赖耦合器。

## 本地运行

```bash
conda env create -f environment.yml -p ./.conda-env
conda activate ./.conda-env
python -m pip install -e . --no-build-isolation
pytest -q
qm-agent run --policy rule --backend physical --output-dir runs/rule-demo
qcal full --output runs/measurement-demo
python scripts/build_dataset.py --devices 64 --output dataset
python scripts/audit_dataset.py dataset
```

训练与发布必须在服务器上执行，并保留基座模型标识、数据哈希、预训练基线、训练配置、完整评测和 LoRA 文件哈希。仓库不分发 Nanbeige 基座权重。

## 目录

- `src/qmagent/`：Agent、协议、控制器、物理模拟和报告。
- `src/qcal/`：实验采集、分析、绘图和真机后端接口。
- `training/`：LoRA 训练与冻结上下文评测。
- `scripts/`：数据构建、审计、服务器流水线和发布检查。
- `tests/`：Agent、测控和训练目标测试。
- `docs/`：第一版架构、真机接入和评测声明。

# Phase-Path Network for Multivariate Time Series Classification

基于阶段路径验证的多变量时间序列分类方法实现。

## 核心思想

将时间序列分类从"这个样本包含哪些局部模式"升级为"这些局部阶段是否按照该类别允许的路径演化"。

## 项目结构

```
.
├── data/                      # 数据处理模块
│   ├── windowing.py          # 重叠窗口切分
│   ├── counterfactual.py     # 反事实阶段路径生成
│   └── dataloader.py         # 数据加载器
├── models/                    # 模型模块
│   ├── backbones/            # 编码器（InceptionTime、ResNet等）
│   ├── phase_prototypes.py   # 低秩阶段原型
│   ├── phase_assignment.py   # 阶段分配
│   ├── phase_graph.py        # 类别阶段图
│   ├── path_forward.py       # 路径递推
│   ├── uncertainty.py        # 不确定性估计
│   ├── confirmed_memory.py   # 延迟确认记忆
│   ├── tf_branch.py          # 时频分支
│   └── phase_path_net.py     # 完整网络
├── losses/                    # 损失函数
│   ├── classification.py     # 分类损失
│   ├── counterfactual.py     # 反事实损失
│   ├── transition.py         # 转移损失
│   ├── memory.py             # 记忆损失
│   └── tf_consistency.py     # 时频一致性损失
├── diagnostics/               # 诊断可视化
│   ├── phase_graph_vis.py    # 阶段图可视化
│   ├── path_vis.py           # 路径可视化
│   └── memory_drift_vis.py   # 记忆漂移可视化
├── configs/                   # 配置文件
├── utils/                     # 工具函数
├── train.py                   # 训练脚本
├── test.py                    # 测试脚本
└── requirements.txt           # 依赖包

```

## 数据集

数据集路径：`D:\Xjnproject\XJNproject\1uka\SPINNET\dataset`

### 核心阶段路径数据集（7个）
- Handwriting - 字符笔画顺序
- UWaveGestureLibrary - 手势动作轨迹
- SpokenArabicDigits - 语音音素演化
- PEMS-SF - 交通流量阶段
- Heartbeat - 心动周期转移
- SelfRegulationSCP1/SCP2 - 脑电阶段

### 非平稳验证数据集（3个）
- HHAR - 设备异质性
- PAMAP2 - 跨被试非平稳性（需下载）
- USC-HAD - 动作风格变化（需下载）

## 环境设置

```bash
# 激活虚拟环境
# Windows
TFproject\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

## 使用方法

```bash
# 训练
python train.py --config configs/handwriting.yaml

# 测试
python test.py --checkpoint checkpoints/best_model.pth --dataset Handwriting
```

## 主要创新点

1. **类别条件阶段路径验证** - 核心创新
2. **反事实阶段路径学习** - 关键创新
3. **延迟确认阶段记忆** - 增强创新
4. **低秩阶段原型** - 结构创新
5. **阶段扰动评估协议** - 辅助贡献

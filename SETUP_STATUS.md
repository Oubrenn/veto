# Phase-Path Network 项目搭建完成

## ✅ 已完成的工作

### 1. 项目结构
```
D:\AAAIproject\
├── data/                      # 数据处理模块 ✓
│   ├── windowing.py          # 重叠窗口切分
│   ├── counterfactual.py     # 反事实阶段路径生成
│   └── dataloader.py         # 数据加载器（支持.ts/.pt/.npy格式）
├── models/                    # 模型模块 ✓
│   ├── backbones/            # 编码器
│   │   ├── inception_time.py # InceptionTime
│   │   ├── resnet.py         # ResNet1D
│   │   └── fcn.py            # FCN
│   ├── phase_prototypes.py   # 低秩阶段原型
│   ├── phase_assignment.py   # 阶段分配
│   ├── phase_graph.py        # 类别阶段图
│   ├── path_forward.py       # 路径递推（HMM-like forward）
│   ├── uncertainty.py        # 不确定性估计
│   ├── confirmed_memory.py   # 延迟确认记忆
│   ├── tf_branch.py          # 时频分支
│   └── phase_path_net.py     # 完整网络
├── losses/                    # 损失函数 ✓
│   ├── classification.py     # 分类损失
│   ├── counterfactual.py     # 反事实损失
│   ├── transition.py         # 转移损失
│   ├── memory.py             # 记忆损失
│   └── tf_consistency.py     # 时频一致性损失
├── utils/                     # 工具函数 ✓
├── configs/                   # 配置文件 ✓
├── train.py                   # 训练脚本 ✓
├── test.py                    # 测试脚本 ✓
├── requirements.txt           # 依赖包 ✓
└── README.md                  # 项目说明 ✓
```

### 2. 数据集
- **路径**: `D:\Xjnproject\XJNproject\1uka\SPINNET\dataset`
- **可用数据集**: Handwriting, HHAR, Heartbeat, PEMS-SF, SelfRegulationSCP1, SelfRegulationSCP2, SpokenArabicDigits, UWaveGestureLibrary, 等
- **格式支持**: .ts (sktime), .pt (PyTorch缓存), .npy

### 3. 虚拟环境
- **名称**: TFproject
- **位置**: `D:\AAAIproject\TFproject`
- **Python版本**: 3.12.4

## ⚠️ 当前问题

### PyTorch DLL加载错误
```
OSError: Error loading "torch\lib\c10.dll" or one of its dependencies
```

**原因**: Windows系统缺少Visual C++ Redistributable或PyTorch版本不兼容

**解决方案**:

### 方案1: 安装Visual C++ Redistributable（推荐）
1. 下载并安装 [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)
2. 重启命令行
3. 重新测试

### 方案2: 重新安装PyTorch
```bash
# 激活虚拟环境
TFproject\Scripts\activate

# 卸载当前PyTorch
pip uninstall torch torchvision torchaudio

# 安装CPU版本（稳定）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 或安装CUDA版本（如果有NVIDIA GPU）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 方案3: 使用Conda环境（最稳定）
```bash
# 创建新环境
conda create -n phasepath python=3.10
conda activate phasepath

# 安装PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 安装其他依赖
pip install scikit-learn pandas matplotlib seaborn tqdm tensorboard pyyaml h5py
```

## 🚀 快速开始

### 1. 修复PyTorch后验证数据加载
```bash
# 激活环境
TFproject\Scripts\activate

# 测试数据加载
python test_dataloader.py
```

### 2. 训练模型
```bash
# 在Handwriting数据集上训练
python train.py --dataset Handwriting --epochs 50 --batch_size 16

# 使用配置文件
python train.py --config configs/handwriting.yaml
```

### 3. 测试模型
```bash
python test.py --checkpoint checkpoints/Handwriting_best.pth --dataset Handwriting
```

## 📊 核心创新点

1. **类别条件阶段路径验证** - 不仅识别"出现了什么阶段"，还验证"阶段演化顺序是否符合该类别"
2. **反事实阶段路径学习** - 保留局部合理性，只破坏演化顺序，强制模型学习路径信息
3. **延迟确认阶段记忆** - 分离候选缓存和长期记忆，避免瞬时噪声污染
4. **低秩阶段原型** - 用子空间表示阶段，容纳变化
5. **可微路径递推** - HMM-like forward算法计算路径概率

## 📝 论文实验建议

### 核心数据集（阶段路径验证）
- Handwriting ⭐⭐⭐⭐⭐
- UWaveGestureLibrary ⭐⭐⭐⭐⭐
- SpokenArabicDigits ⭐⭐⭐⭐⭐
- PEMS-SF ⭐⭐⭐⭐
- Heartbeat ⭐⭐⭐⭐

### 非平稳验证（记忆机制）
- HHAR（设备异质性）
- PAMAP2（跨被试）- 需下载
- USC-HAD（动作风格）- 需下载

### 标准主表（18个UEA数据集）
- 当前可用：Handwriting, UWaveGestureLibrary, SpokenArabicDigits, PEMS-SF, Heartbeat, SelfRegulationSCP1/2
- 需补充：ArticularyWordRecognition, CharacterTrajectories, NATOPS, Cricket, BasicMotions, Epilepsy, Libras, RacketSports

## 🔧 后续工作

1. **修复PyTorch环境** - 按照上述方案解决DLL加载问题
2. **验证数据加载** - 确保所有数据集正确加载
3. **小规模训练测试** - 在一个数据集上跑10个epoch验证代码正确性
4. **完整训练** - 在核心7个数据集上完整训练
5. **补充数据集** - 下载缺失的UEA数据集
6. **可视化诊断** - 实现阶段图、路径、记忆漂移可视化
7. **消融实验** - 验证各模块贡献

## 📚 参考资源

- 数据集来源: [UCR/UEA Archive](https://www.timeseriesclassification.com/)
- PyTorch安装: [官方指南](https://pytorch.org/get-started/locally/)
- 项目架构基于初稿PDF中的方法设计

---

**项目状态**: 🟡 代码完成，等待环境修复

**下一步**: 安装Visual C++ Redistributable或重新安装PyTorch

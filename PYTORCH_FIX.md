# PyTorch环境修复指南

## 问题诊断

当前错误：
```
OSError: [WinError 1114] Error loading "D:\AAAIproject\TFproject\Lib\site-packages\torch\lib\c10.dll"
```

这是Windows上PyTorch最常见的问题，原因是缺少必要的系统库。

## 解决方案（按优先级排序）

### ✅ 方案1: 安装Visual C++ Redistributable（最简单）

1. **下载安装包**
   - 链接: https://aka.ms/vs/17/release/vc_redist.x64.exe
   - 或访问: https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist

2. **运行安装**
   - 双击下载的 `vc_redist.x64.exe`
   - 按照提示完成安装
   - 如果提示"已安装"，选择"修复"

3. **验证**
   ```bash
   # 重启命令行
   cd D:\AAAIproject
   TFproject\Scripts\activate
   python -c "import torch; print(torch.__version__)"
   ```

### ✅ 方案2: 重新安装PyTorch（如果方案1无效）

#### 2.1 卸载现有PyTorch
```bash
TFproject\Scripts\activate
pip uninstall torch torchvision torchaudio -y
```

#### 2.2 安装CPU版本（稳定，推荐开始使用）
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### 2.3 或安装CUDA版本（如果有NVIDIA显卡）

**检查CUDA版本**:
```bash
nvidia-smi
```

**安装对应版本**:
```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### ✅ 方案3: 使用Conda（最稳定，适合长期开发）

#### 3.1 安装Miniconda
- 下载: https://docs.conda.io/en/latest/miniconda.html
- 安装后重启命令行

#### 3.2 创建新环境
```bash
# 创建环境
conda create -n phasepath python=3.10 -y
conda activate phasepath

# 安装PyTorch（CPU版本）
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# 或安装CUDA版本
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 安装其他依赖
pip install scikit-learn pandas matplotlib seaborn tqdm tensorboard pyyaml h5py

# 验证
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

#### 3.3 切换到新环境
```bash
# 以后每次使用
conda activate phasepath
cd D:\AAAIproject

# 运行训练
python train.py --dataset Handwriting --epochs 10
```

## 验证安装成功

运行以下命令验证：

```bash
# 激活环境
TFproject\Scripts\activate  # 或 conda activate phasepath

# 测试PyTorch
python -c "import torch; print('PyTorch version:', torch.__version__)"

# 测试CUDA（如果安装了GPU版本）
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# 测试数据加载
cd D:\AAAIproject
python test_dataloader.py

# 如果成功，会看到：
# Loading from .pt format: Handwriting_TRAIN_pad0.cache.pt
# Loaded Handwriting train: XXX samples, shape=(XXX, XXX, XXX), classes=XX
```

## 快速测试训练

环境修复后，运行快速测试：

```bash
cd D:\AAAIproject
TFproject\Scripts\activate  # 或 conda activate phasepath

# 10个epoch快速测试
python train.py --dataset Handwriting --epochs 10 --batch_size 16

# 查看训练日志
# 输出会保存到: logs/Handwriting/
# 模型会保存到: checkpoints/Handwriting_best.pth
```

## 常见问题

### Q1: 提示"找不到指定的模块"
**A**: 确保安装了Visual C++ Redistributable（方案1）

### Q2: 训练很慢
**A**: 
- 检查是否使用了GPU版本：`python -c "import torch; print(torch.cuda.is_available())"`
- 如果返回False，安装CUDA版本的PyTorch
- 或使用CPU版本但减小batch_size

### Q3: 内存不足
**A**: 
- 减小batch_size: `--batch_size 8`
- 减小窗口数量（修改configs中的window_size）

### Q4: 数据加载失败
**A**: 
- 检查数据集路径是否正确
- 运行 `python check_datasets.py` 查看可用数据集
- 确保数据集格式为.ts或.pt

## 推荐配置

**开发/调试**:
- Python 3.10 + PyTorch CPU版本
- 小数据集 + 小batch_size测试

**正式训练**:
- Python 3.10 + PyTorch CUDA版本
- 完整数据集 + 合理batch_size

**发布/部署**:
- 使用Conda环境导出: `conda env export > environment.yml`
- 或使用pip: `pip freeze > requirements_frozen.txt`

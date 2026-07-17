#!/bin/bash
# 项目验证脚本

echo "======================================"
echo "Phase-Path Network 项目验证"
echo "======================================"
echo ""

# 检查项目结构
echo "1. 检查项目结构..."
folders=("data" "models" "models/backbones" "losses" "utils" "configs")
for folder in "${folders[@]}"; do
    if [ -d "$folder" ]; then
        echo "  ✓ $folder/"
    else
        echo "  ✗ $folder/ (missing)"
    fi
done
echo ""

# 检查核心模块
echo "2. 检查核心模块..."
modules=(
    "data/windowing.py"
    "data/counterfactual.py"
    "data/dataloader.py"
    "models/phase_prototypes.py"
    "models/phase_assignment.py"
    "models/phase_graph.py"
    "models/path_forward.py"
    "models/uncertainty.py"
    "models/confirmed_memory.py"
    "models/phase_path_net.py"
    "models/backbones/inception_time.py"
    "losses/classification.py"
    "losses/counterfactual.py"
    "train.py"
    "test.py"
)

for module in "${modules[@]}"; do
    if [ -f "$module" ]; then
        echo "  ✓ $module"
    else
        echo "  ✗ $module (missing)"
    fi
done
echo ""

# 统计代码量
echo "3. 代码统计..."
total_lines=$(find . -name "*.py" -type f -exec wc -l {} + | tail -1 | awk '{print $1}')
total_files=$(find . -name "*.py" -type f | wc -l)
echo "  总文件数: $total_files"
echo "  总代码行: $total_lines"
echo ""

# 检查配置文件
echo "4. 检查配置文件..."
configs=("configs/default.yaml" "configs/handwriting.yaml")
for config in "${configs[@]}"; do
    if [ -f "$config" ]; then
        echo "  ✓ $config"
    else
        echo "  ✗ $config (missing)"
    fi
done
echo ""

# 检查文档
echo "5. 检查文档..."
docs=("README.md" "SETUP_STATUS.md" "PYTORCH_FIX.md" "requirements.txt")
for doc in "${docs[@]}"; do
    if [ -f "$doc" ]; then
        echo "  ✓ $doc"
    else
        echo "  ✗ $doc (missing)"
    fi
done
echo ""

echo "======================================"
echo "项目验证完成！"
echo "======================================"
echo ""
echo "下一步:"
echo "1. 修复PyTorch环境（参考 PYTORCH_FIX.md）"
echo "2. 运行: python test_dataloader.py"
echo "3. 开始训练: python train.py --dataset Handwriting --epochs 10"

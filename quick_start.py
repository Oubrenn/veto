"""快速开始脚本 - 测试数据集加载"""
import sys
sys.path.append('.')

from data import get_dataset_info

# 数据集路径
DATA_PATH = "D:/Xjnproject/XJNproject/1uka/SPINNET/dataset"

print("正在扫描数据集...")
info = get_dataset_info(DATA_PATH)

print(f"\n找到 {len(info)} 个数据集:\n")
print(f"{'Dataset':<30} {'Classes':<10} {'Channels':<10} {'Length':<10}")
print("-" * 60)

for name, meta in sorted(info.items()):
    print(f"{name:<30} {meta['n_classes']:<10} {meta['n_channels']:<10} {meta['seq_length']:<10}")

print("\n核心阶段路径数据集（推荐用于开发）:")
core_datasets = [
    'Handwriting',
    'UWaveGestureLibrary',
    'SpokenArabicDigits',
    'PEMS-SF',
    'Heartbeat',
    'SelfRegulationSCP1',
    'SelfRegulationSCP2'
]

available_core = [d for d in core_datasets if d in info]
print(f"  可用: {', '.join(available_core)}")
print(f"  缺失: {', '.join([d for d in core_datasets if d not in info])}")

print("\n非平稳验证数据集:")
har_datasets = ['HHAR', 'DSADS']
available_har = [d for d in har_datasets if d in info]
print(f"  可用: {', '.join(available_har)}")

print("\n快速测试训练命令:")
if available_core:
    test_dataset = available_core[0]
    print(f"  python train.py --dataset {test_dataset} --epochs 10 --batch_size 16")

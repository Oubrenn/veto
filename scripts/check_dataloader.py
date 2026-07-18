"""测试数据加载"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from data.dataloader import PhasePathDataset

    # 数据集路径
    DATA_PATH = "D:/Xjnproject/XJNproject/1uka/SPINNET/dataset"

    print("测试加载Handwriting数据集...")
    print("-" * 60)

    # 加载训练集
    try:
        train_dataset = PhasePathDataset(
            data_path=DATA_PATH,
            dataset_name='Handwriting',
            split='train',
            normalize=True
        )

        print(f"\n训练集信息:")
        print(f"  样本数: {len(train_dataset)}")
        print(f"  类别数: {train_dataset.n_classes}")
        print(f"  通道数: {train_dataset.n_channels}")
        print(f"  序列长度: {train_dataset.seq_length}")

        # 测试获取一个样本
        x, y = train_dataset[0]
        print(f"\n样本形状:")
        print(f"  X: {x.shape}")
        print(f"  y: {y.item()}")

    except Exception as e:
        print(f"加载训练集失败: {e}")
        import traceback
        traceback.print_exc()

    # 加载测试集
    try:
        test_dataset = PhasePathDataset(
            data_path=DATA_PATH,
            dataset_name='Handwriting',
            split='test',
            normalize=True
        )

        print(f"\n测试集信息:")
        print(f"  样本数: {len(test_dataset)}")

    except Exception as e:
        print(f"加载测试集失败: {e}")

    print("\n" + "=" * 60)
    print("数据加载测试完成！")

except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保已安装PyTorch: pip install torch")

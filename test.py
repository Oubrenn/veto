"""测试脚本"""
import os
import argparse
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    sns = None

from data import get_dataloader
from models import PhasePathNet


def test(model, test_loader, device):
    """测试模型"""
    model.eval()

    all_preds = []
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for x, labels in test_loader:
            x = x.to(device)
            labels = labels.to(device)

            output = model(x)
            logits = output['logits']
            pred = torch.argmax(logits, dim=-1)

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_logits.append(logits.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_logits = np.concatenate(all_logits, axis=0)

    return all_preds, all_labels, all_logits


def compute_metrics(preds, labels, n_classes):
    """计算评估指标"""
    acc = accuracy_score(labels, preds)

    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, average='macro', zero_division=0
    )

    # 每类指标
    per_class_metrics = precision_recall_fscore_support(
        labels, preds, average=None, zero_division=0
    )

    metrics = {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'per_class_precision': per_class_metrics[0],
        'per_class_recall': per_class_metrics[1],
        'per_class_f1': per_class_metrics[2],
        'per_class_support': per_class_metrics[3]
    }

    return metrics


def plot_confusion_matrix(labels, preds, save_path):
    """绘制混淆矩阵"""
    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(10, 8))
    if sns is not None:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    else:
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar()
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, int(cm[i, j]), ha='center', va='center')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f'Confusion matrix saved to {save_path}')


def main():
    parser = argparse.ArgumentParser(description='测试Phase-Path网络')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型checkpoint路径')
    parser.add_argument('--dataset', type=str, default='Handwriting', help='数据集名称')
    parser.add_argument('--data_path', type=str, default='D:/Xjnproject/XJNproject/1uka/SPINNET/dataset',
                       help='数据集路径')
    parser.add_argument('--batch_size', type=int, default=32, help='批大小')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', type=str, default='results', help='结果保存目录')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载数据
    print(f'Loading dataset: {args.dataset}')
    test_loader = get_dataloader(
        args.data_path,
        args.dataset,
        batch_size=args.batch_size,
        split='test',
        shuffle=False
    )

    dataset = test_loader.dataset
    n_classes = dataset.n_classes
    n_channels = dataset.n_channels
    seq_length = dataset.seq_length

    print(f'Classes: {n_classes}, Channels: {n_channels}, Length: {seq_length}')

    # 加载模型
    print(f'Loading model from {args.checkpoint}')
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)

    ckpt_args = checkpoint.get('args', {})
    model = PhasePathNet(
        n_classes=n_classes,
        n_channels=n_channels,
        seq_length=seq_length,
        n_phases=ckpt_args.get('n_phases', 5),
        embed_dim=ckpt_args.get('embed_dim', 128),
        window_size=ckpt_args.get('window_size', None),
        stride=ckpt_args.get('stride', None),
        backbone=ckpt_args.get('backbone', 'inception'),
        use_tf_branch=False,
        use_memory=not ckpt_args.get('no_memory', False),
        transition_mode=ckpt_args.get('transition_mode', 'free'),
        prototype_mode=ckpt_args.get('prototype_mode', 'class'),
        head_mode=ckpt_args.get('head_mode', 'veto'),
        use_uncertainty=not ckpt_args.get('no_uncertainty', False),
    ).to(args.device)

    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, best acc: {checkpoint['best_acc']:.4f}")

    # 测试
    print('Testing...')
    preds, labels, logits = test(model, test_loader, args.device)

    # 计算指标
    metrics = compute_metrics(preds, labels, n_classes)

    print('\n=== Test Results ===')
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-score: {metrics['f1']:.4f}")

    print('\n=== Per-class Metrics ===')
    for i in range(n_classes):
        print(f"Class {i}: P={metrics['per_class_precision'][i]:.4f}, "
              f"R={metrics['per_class_recall'][i]:.4f}, "
              f"F1={metrics['per_class_f1'][i]:.4f}, "
              f"N={metrics['per_class_support'][i]}")

    # 保存结果
    results_file = os.path.join(args.output_dir, f'{args.dataset}_results.txt')
    with open(results_file, 'w') as f:
        f.write(f'Dataset: {args.dataset}\n')
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1-score: {metrics['f1']:.4f}\n")

    print(f'\nResults saved to {results_file}')

    # 绘制混淆矩阵
    cm_path = os.path.join(args.output_dir, f'{args.dataset}_confusion_matrix.png')
    plot_confusion_matrix(labels, preds, cm_path)


if __name__ == '__main__':
    main()

"""工具函数"""
import torch
import numpy as np
import random
import os


def set_seed(seed: int = 42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> int:
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(state: dict, filepath: str, is_best: bool = False):
    """保存checkpoint"""
    torch.save(state, filepath)
    if is_best:
        best_path = filepath.replace('.pth', '_best.pth')
        torch.save(state, best_path)


def load_checkpoint(filepath: str, model: torch.nn.Module, optimizer=None):
    """加载checkpoint"""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint.get('epoch', 0), checkpoint.get('best_acc', 0.0)


class AverageMeter:
    """平均值计算器"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience: int = 10, delta: float = 0.0):
        """
        Args:
            patience: 容忍轮数
            delta: 最小改进量
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_metric: float) -> bool:
        """
        Args:
            val_metric: 验证指标（越大越好）

        Returns:
            是否应该停止训练
        """
        score = val_metric

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.early_stop


def get_device(device_str: str = 'auto') -> torch.device:
    """获取设备"""
    if device_str == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        return torch.device(device_str)


def format_time(seconds: float) -> str:
    """格式化时间"""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f'{h:02d}:{m:02d}:{s:02d}'

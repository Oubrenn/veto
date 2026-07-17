"""Training entry point and reusable training utilities."""
import argparse
import os
from typing import Dict, Optional

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data import CounterfactualGenerator, get_dataloader
from losses import ClassificationLoss, CounterfactualLoss, MemoryLoss, TransitionLoss
from models import PhasePathNet, build_baseline_model
from utils.common import set_seed


BASELINE_MODEL_NAMES = [
    "rocket",
    "multirocket",
    "inceptiontime",
    "timesnet",
    "tapnet",
    "pdftime",
    "hc2_lite",
    "mts2graph",
    "simtsc",
    "tma_gat",
    "temporal_cnn",
]


def reconstruct_from_windows(
    windows: torch.Tensor,
    seq_length: int,
    stride: int,
) -> torch.Tensor:
    """Approximate overlap-add reconstruction from partitioned windows."""
    if windows.ndim != 4:
        raise ValueError(f"Expected (B, N, L, C) windows, got {tuple(windows.shape)}")

    B, N, L, C = windows.shape
    total_length = (N - 1) * stride + L
    x = windows.new_zeros(B, total_length, C)
    counts = windows.new_zeros(B, total_length, C)
    for idx in range(N):
        start = idx * stride
        end = start + L
        x[:, start:end, :] += windows[:, idx, :, :]
        counts[:, start:end, :] += 1

    x = x / counts.clamp_min(1)
    if total_length < seq_length:
        x = F.pad(x, (0, 0, 0, seq_length - total_length), mode="replicate")
    return x[:, :seq_length, :]


def train_epoch(
    model,
    train_loader,
    optimizer,
    losses,
    cf_generator,
    device,
    epoch,
    loss_weights: Optional[Dict[str, float]] = None,
    cf_start_epoch: int = 5,
):
    """Train one epoch and return mean loss/accuracy."""
    model.train()
    if loss_weights is None:
        loss_weights = {"cls": 1.0, "cf": 0.5, "trans": 0.1, "mem": 0.1, "tf": 0.1}

    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    disable_tqdm = os.environ.get("DISABLE_TQDM", "0") == "1"
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=disable_tqdm)
    for batch in pbar:
        if len(batch) == 2:
            x, labels = batch
            valid_lengths = None
        elif len(batch) == 3:
            x, labels, valid_lengths = batch
        else:
            raise ValueError("Expected training batches of (x, label) or (x, label, length)")
        x = x.to(device)
        labels = labels.to(device)
        if valid_lengths is not None:
            valid_lengths = valid_lengths.to(device)

        optimizer.zero_grad()
        output = model(x, labels, valid_lengths=valid_lengths)
        logits = output["logits"]

        loss_cls = losses["cls"](logits, labels)

        is_phase_path = hasattr(model, "window_partitioner")

        if is_phase_path and "cf" in losses and cf_generator is not None and epoch >= cf_start_epoch:
            windows = model.window_partitioner.partition(x)
            cf_windows, _ = cf_generator.generate(windows)
            cf_x = reconstruct_from_windows(
                cf_windows,
                seq_length=x.shape[1],
                stride=model.window_partitioner.stride,
            )
            cf_output = model(cf_x, labels, valid_lengths=valid_lengths)
            loss_cf = losses["cf"](logits, cf_output["logits"], labels)
        else:
            loss_cf = torch.tensor(0.0, device=device)

        if is_phase_path and "trans" in losses:
            loss_trans = losses["trans"](
                output["phase_assignment"],
                output["transition_matrices"],
                labels,
                window_mask=output.get("window_mask"),
            )
        else:
            loss_trans = torch.tensor(0.0, device=device)

        if is_phase_path and "mem" in losses and "confirmed_memory" in output:
            loss_mem = losses["mem"](
                output["embeddings"],
                output["phase_assignment"],
                output["confirmed_memory"],
                output["reliability"],
                labels,
            )
        else:
            loss_mem = torch.tensor(0.0, device=device)

        loss_tf = torch.tensor(0.0, device=device)
        loss = (
            loss_weights.get("cls", 1.0) * loss_cls
            + loss_weights.get("cf", 0.0) * loss_cf
            + loss_weights.get("trans", 0.0) * loss_trans
            + loss_weights.get("mem", 0.0) * loss_mem
            + loss_weights.get("tf", 0.0) * loss_tf
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        pred = torch.argmax(logits, dim=-1)
        acc = (pred == labels).float().mean().item()

        total_loss += loss.item()
        total_acc += acc
        n_batches += 1
        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "acc": f"{acc:.4f}",
                "cls": f"{loss_cls.item():.4f}",
                "cf": f"{loss_cf.item():.4f}",
            }
        )

    return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1)


def evaluate(model, test_loader, device):
    """Evaluate a model and return accuracy, predictions, labels."""
    model.eval()
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        disable_tqdm = os.environ.get("DISABLE_TQDM", "0") == "1"
        for batch in tqdm(test_loader, desc="Evaluating", disable=disable_tqdm):
            if len(batch) == 2:
                x, labels = batch
                valid_lengths = None
            elif len(batch) == 3:
                x, labels, valid_lengths = batch
            else:
                raise ValueError("Expected evaluation batches of (x, label) or (x, label, length)")
            x = x.to(device)
            labels = labels.to(device)
            if valid_lengths is not None:
                valid_lengths = valid_lengths.to(device)
            logits = model(x, valid_lengths=valid_lengths)["logits"]
            pred = torch.argmax(logits, dim=-1)
            total_correct += (pred == labels).sum().item()
            total_samples += labels.size(0)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = total_correct / max(total_samples, 1)
    return acc, np.array(all_preds), np.array(all_labels)


def stratified_train_val_indices(y, val_fraction: float, seed: int):
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    train_indices = []
    val_indices = []
    for label in np.unique(y):
        label_indices = np.where(y == label)[0]
        rng.shuffle(label_indices)
        if len(label_indices) <= 1:
            train_indices.extend(label_indices.tolist())
            continue
        n_val = int(round(len(label_indices) * val_fraction))
        n_val = min(len(label_indices) - 1, max(1, n_val))
        val_indices.extend(label_indices[:n_val].tolist())
        train_indices.extend(label_indices[n_val:].tolist())
    if not train_indices or not val_indices:
        raise ValueError("Stratified train/validation split produced an empty split")
    return sorted(train_indices), sorted(val_indices)


def make_subset_loader(dataset, indices, batch_size, shuffle, drop_last):
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last and len(indices) >= batch_size,
    )


def build_model(args, dataset_train):
    if args.model != "veto":
        return build_baseline_model(
            args.model,
            n_channels=dataset_train.n_channels,
            n_classes=dataset_train.n_classes,
            hidden_dim=args.embed_dim,
        )

    return PhasePathNet(
        n_classes=dataset_train.n_classes,
        n_channels=dataset_train.n_channels,
        seq_length=dataset_train.seq_length,
        n_phases=args.n_phases,
        embed_dim=args.embed_dim,
        window_size=args.window_size,
        stride=args.stride,
        backbone=args.backbone,
        use_tf_branch=False,
        use_memory=not args.no_memory,
        transition_mode=args.transition_mode,
        prototype_mode=args.prototype_mode,
        head_mode=args.head_mode,
        path_score_mode=args.path_score_mode,
        path_weight_override=args.path_weight_override,
        use_uncertainty=not args.no_uncertainty,
        memory_update_mode=args.memory_update_mode,
    )


def main():
    parser = argparse.ArgumentParser(description="Train Phase-Path network")
    parser.add_argument("--dataset", type=str, default="Handwriting")
    parser.add_argument(
        "--data_path",
        type=str,
        default="D:/Xjnproject/XJNproject/1uka/SPINNET/dataset",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--model", choices=["veto", *BASELINE_MODEL_NAMES], default="veto")
    parser.add_argument("--backbone", choices=["inception", "resnet", "fcn"], default="inception")
    parser.add_argument("--n_phases", type=int, default=5)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--window_size", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--transition_mode", choices=["uniform", "free", "class_independent", "neural", "attention"], default="neural")
    parser.add_argument("--prototype_mode", choices=["class", "shared", "full"], default="class")
    parser.add_argument("--head_mode", choices=["backbone", "orderless", "prototype", "hmm", "veto"], default="veto")
    parser.add_argument("--path_score_mode", choices=["gain", "raw"], default="gain")
    parser.add_argument("--path_weight_override", type=float, default=None)
    parser.add_argument("--no_memory", action="store_true")
    parser.add_argument("--memory_update_mode", choices=["confirmed", "direct_ema"], default="confirmed")
    parser.add_argument("--no_cf", action="store_true")
    parser.add_argument("--no_trans", action="store_true")
    parser.add_argument("--no_uncertainty", action="store_true")
    parser.add_argument("--cf_start_epoch", type=int, default=5)
    args = parser.parse_args()
    set_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    writer = SummaryWriter(f"logs/{args.dataset}")

    full_train_loader = get_dataloader(
        args.data_path,
        args.dataset,
        batch_size=args.batch_size,
        split="train",
        shuffle=False,
        drop_last=False,
    )
    train_indices, val_indices = stratified_train_val_indices(
        full_train_loader.dataset.y,
        args.val_fraction,
        args.seed,
    )
    train_loader = make_subset_loader(
        full_train_loader.dataset,
        train_indices,
        args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = make_subset_loader(
        full_train_loader.dataset,
        val_indices,
        args.batch_size,
        shuffle=False,
        drop_last=False,
    )
    test_loader = get_dataloader(
        args.data_path,
        args.dataset,
        batch_size=args.batch_size,
        split="test",
        shuffle=False,
        drop_last=False,
    )

    model = build_model(args, full_train_loader.dataset).to(args.device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    losses = {"cls": ClassificationLoss(label_smoothing=0.1)}
    if not args.no_cf:
        losses["cf"] = CounterfactualLoss(margin=1.0)
    if not args.no_trans:
        losses["trans"] = TransitionLoss(loss_type="mse")
    if not args.no_memory:
        losses["mem"] = MemoryLoss(margin=0.8)
    cf_generator = None if args.no_cf else CounterfactualGenerator(min_phase_length=3)
    loss_weights = {"cls": 1.0, "cf": 0.5, "trans": 0.1, "mem": 0.1, "tf": 0.0}

    best_val_f1 = float("-inf")
    best_val_acc = 0.0
    best_path = os.path.join(args.save_dir, f"{args.dataset}_best_val.pth")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            losses,
            cf_generator,
            args.device,
            epoch,
            loss_weights=loss_weights,
            cf_start_epoch=args.cf_start_epoch,
        )
        val_acc, val_preds, val_labels = evaluate(model, val_loader, args.device)
        val_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
        scheduler.step()

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)
        writer.add_scalar("Acc/val", val_acc, epoch)
        writer.add_scalar("MacroF1/val", val_f1, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, "
            f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, val_f1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1 or (np.isclose(val_f1, best_val_f1) and val_acc > best_val_acc):
            best_val_f1 = val_f1
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_acc": best_val_acc,
                    "best_val_macro_f1": best_val_f1,
                    "selection_metric": "val_macro_f1",
                    "args": vars(args),
                },
                best_path,
            )
            print(f"Best validation model saved. Macro-F1: {best_val_f1:.4f}")

    writer.close()
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location=args.device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
    test_acc, test_preds, test_labels = evaluate(model, test_loader, args.device)
    test_f1 = f1_score(test_labels, test_preds, average="macro", zero_division=0)
    print(
        f"Training finished. Best validation Macro-F1: {best_val_f1:.4f}. "
        f"Selected checkpoint test_acc={test_acc:.4f}, test_macro_f1={test_f1:.4f}"
    )


if __name__ == "__main__":
    main()

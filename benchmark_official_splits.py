"""Run official TRAIN/TEST split experiments with configurable ablations."""
import argparse
import csv
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Subset

from data import CounterfactualGenerator, get_dataloader
from losses import ClassificationLoss, CounterfactualLoss, MemoryLoss, TransitionLoss
from models import PhasePathNet, build_baseline_model
from train import evaluate, train_epoch
from utils.common import count_parameters, set_seed


DEFAULT_DATA_PATH = "D:/Xjnproject/XJNproject/1uka/SPINNET/dataset"
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

CSV_FIELDS = [
    "experiment",
    "dataset",
    "seed",
    "status",
    "error",
    "train_file",
    "test_file",
    "train_samples",
    "val_samples",
    "test_samples",
    "train_fraction",
    "val_fraction",
    "selection_metric",
    "n_classes",
    "n_channels",
    "seq_length",
    "model_params",
    "epochs_requested",
    "epochs_completed",
    "batch_size",
    "lr",
    "label_smoothing",
    "weight_decay",
    "device",
    "model",
    "backbone",
    "normalize",
    "window_size",
    "stride",
    "head_mode",
    "prototype_mode",
    "transition_mode",
    "path_score_mode",
    "path_weight_override",
    "use_cf",
    "use_trans",
    "use_memory",
    "memory_update_mode",
    "use_uncertainty",
    "drop_last",
    "final_train_loss",
    "final_train_acc",
    "final_val_acc",
    "final_val_macro_f1",
    "best_val_acc",
    "best_val_macro_f1",
    "selected_test_acc",
    "selected_test_macro_f1",
    "final_test_acc",
    "final_macro_f1",
    "best_test_acc",
    "best_macro_f1",
    "diagnostic_ppa",
    "diagnostic_delta_g",
    "diagnostic_auroc",
    "prototype_drift",
    "memory_drift",
    "best_epoch",
    "train_seconds",
    "eval_seconds",
    "total_seconds",
    "single_sample_latency_ms",
    "throughput_samples_per_sec",
    "peak_gpu_memory_mb",
]


def find_split_file(dataset_dir: Path, dataset_name: str, split: str) -> str:
    split_upper = split.upper()
    candidates = [
        dataset_dir / f"{dataset_name}_{split_upper}_pad0.cache.pt",
        dataset_dir / f"{dataset_name}_{split_upper}.pt",
        dataset_dir / f"{dataset_name}_{split_upper}.ts",
        dataset_dir / f"{dataset_name}_{split_upper}.npy",
        dataset_dir / f"X_{split}.npy",
        dataset_dir / f"{split}.npy",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.name
    return ""


def discover_datasets(data_path: Path) -> list[str]:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {data_path}")
    return sorted(item.name for item in data_path.iterdir() if item.is_dir())


def make_losses(args) -> dict:
    losses = {"cls": ClassificationLoss(label_smoothing=args.label_smoothing)}
    if args.model != "veto":
        return losses
    if not args.no_cf:
        losses["cf"] = CounterfactualLoss(margin=1.0)
    if not args.no_trans:
        losses["trans"] = TransitionLoss(loss_type="mse")
    if not args.no_memory:
        losses["mem"] = MemoryLoss(margin=0.8)
    return losses


def make_model(args, dataset_train):
    if args.model != "veto":
        return build_baseline_model(
            args.model,
            n_channels=dataset_train.n_channels,
            n_classes=dataset_train.n_classes,
            hidden_dim=args.embed_dim,
        ).to(args.device)

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
    ).to(args.device)


def empty_result(args, dataset_name: str, seed: int, status: str, error: str = "") -> dict:
    return {
        "experiment": args.experiment,
        "dataset": dataset_name,
        "seed": seed,
        "status": status,
        "error": error,
        "train_file": "",
        "test_file": "",
        "train_samples": "",
        "val_samples": "",
        "test_samples": "",
        "train_fraction": args.train_fraction,
        "val_fraction": args.val_fraction,
        "selection_metric": "val_macro_f1",
        "n_classes": "",
        "n_channels": "",
        "seq_length": "",
        "model_params": "",
        "epochs_requested": args.epochs,
        "epochs_completed": 0,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "label_smoothing": args.label_smoothing,
        "weight_decay": args.weight_decay,
        "device": args.device,
        "model": args.model,
        "backbone": args.backbone,
        "normalize": not args.no_normalize,
        "window_size": args.window_size or "",
        "stride": args.stride or "",
        "head_mode": args.head_mode,
        "prototype_mode": args.prototype_mode,
        "transition_mode": args.transition_mode,
        "path_score_mode": args.path_score_mode,
        "path_weight_override": args.path_weight_override if args.path_weight_override is not None else "",
        "use_cf": not args.no_cf,
        "use_trans": not args.no_trans,
        "use_memory": not args.no_memory,
        "memory_update_mode": args.memory_update_mode,
        "use_uncertainty": not args.no_uncertainty,
        "drop_last": not args.no_drop_last,
        "final_train_loss": "",
        "final_train_acc": "",
        "final_val_acc": "",
        "final_val_macro_f1": "",
        "best_val_acc": "",
        "best_val_macro_f1": "",
        "selected_test_acc": "",
        "selected_test_macro_f1": "",
        "final_test_acc": "",
        "final_macro_f1": "",
        "best_test_acc": "",
        "best_macro_f1": "",
        "diagnostic_ppa": "",
        "diagnostic_delta_g": "",
        "diagnostic_auroc": "",
        "prototype_drift": "",
        "memory_drift": "",
        "best_epoch": "",
        "train_seconds": "",
        "eval_seconds": "",
        "total_seconds": "",
        "single_sample_latency_ms": "",
        "throughput_samples_per_sec": "",
        "peak_gpu_memory_mb": "",
    }


def format_cell(value):
    if value == "" or value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def state_dict_to_cpu(model) -> dict:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def write_results(results: list[dict], csv_path: Path, json_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for result in results:
            writer.writerow({field: format_cell(result.get(field, "")) for field in CSV_FIELDS})

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)


def stratified_train_val_indices(y, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
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

    if not train_indices:
        raise ValueError("Stratified split produced an empty training set")
    if not val_indices:
        raise ValueError("Stratified split produced an empty validation set")
    return sorted(train_indices), sorted(val_indices)


def apply_train_fraction(indices: list[int], y, fraction: float, seed: int) -> list[int]:
    if fraction >= 1.0:
        return sorted(indices)
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    selected = []
    indices_by_label = {}
    for idx in indices:
        indices_by_label.setdefault(y[idx], []).append(idx)
    for label_indices in indices_by_label.values():
        label_indices = np.asarray(label_indices)
        n_keep = max(1, int(round(len(label_indices) * fraction)))
        n_keep = min(n_keep, len(label_indices))
        selected.extend(rng.choice(label_indices, size=n_keep, replace=False).tolist())
    return sorted(selected)


def make_subset_loader(dataset, indices: list[int], batch_size: int, shuffle: bool, num_workers: int, drop_last: bool):
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last and len(indices) >= batch_size,
    )


def subset_loader(loader, fraction: float, seed: int, batch_size: int, drop_last: bool):
    if fraction >= 1.0:
        return loader
    dataset = loader.dataset
    rng = np.random.default_rng(seed)
    y = np.asarray(dataset.y)
    selected = []
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        n_keep = max(1, int(round(len(idx) * fraction)))
        selected.extend(rng.choice(idx, size=n_keep, replace=False).tolist())
    selected = sorted(selected)
    return DataLoader(
        Subset(dataset, selected),
        batch_size=batch_size,
        shuffle=True,
        num_workers=loader.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last and len(selected) >= batch_size,
    )


def measure_inference(model, loader, device, max_batches: int = 5) -> tuple[float, float]:
    model.eval()
    n_samples = 0
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(loader):
            if batch_idx >= max_batches:
                break
            x = x.to(device)
            _ = model(x)["logits"]
            n_samples += x.shape[0]
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    throughput = n_samples / elapsed if elapsed > 0 else 0.0
    latency = (elapsed / n_samples) * 1000 if n_samples > 0 else 0.0
    return latency, throughput


def reconstruct_corrupted_windows(model, x):
    if not hasattr(model, "window_partitioner"):
        return x
    windows = model.window_partitioner.partition(x)
    if windows.shape[1] >= 4:
        corrupted = windows.clone()
        corrupted[:, 1, :, :], corrupted[:, -2, :, :] = (
            corrupted[:, -2, :, :].clone(),
            corrupted[:, 1, :, :].clone(),
        )
    else:
        corrupted = windows.flip(dims=[1])
    from train import reconstruct_from_windows

    return reconstruct_from_windows(
        corrupted,
        seq_length=x.shape[1],
        stride=model.window_partitioner.stride,
    )


def compute_order_diagnostics(model, loader, device, max_batches: int):
    if max_batches <= 0 or not hasattr(model, "window_partitioner"):
        return "", "", ""
    model.eval()
    valid_scores = []
    corrupt_scores = []
    with torch.no_grad():
        for batch_idx, (x, labels) in enumerate(loader):
            if batch_idx >= max_batches:
                break
            x = x.to(device)
            labels = labels.to(device)
            windows = model.window_partitioner.partition(x)
            embeddings = model.encoder(windows)
            valid = model.forward_from_embeddings(embeddings)
            shuffled_embeddings = embeddings.clone()
            if embeddings.shape[1] > 1:
                for sample_idx in range(embeddings.shape[0]):
                    order = torch.randperm(embeddings.shape[1], device=device)
                    shuffled_embeddings[sample_idx] = embeddings[sample_idx, order]
            corrupt = model.forward_from_embeddings(shuffled_embeddings)
            idx = torch.arange(labels.shape[0], device=device)
            norm = max(embeddings.shape[1] - 1, 1)
            valid_scores.extend((valid["transition_gain"][idx, labels] / norm).detach().cpu().numpy())
            corrupt_scores.extend((corrupt["transition_gain"][idx, labels] / norm).detach().cpu().numpy())

    if not valid_scores:
        return "", "", ""
    valid_scores = np.asarray(valid_scores)
    corrupt_scores = np.asarray(corrupt_scores)
    delta = valid_scores - corrupt_scores
    ppa = float(np.mean(delta > 0))
    delta_g = float(np.mean(delta))
    try:
        from sklearn.metrics import roc_auc_score

        labels_auc = np.concatenate([np.ones_like(valid_scores), np.zeros_like(corrupt_scores)])
        scores_auc = np.concatenate([valid_scores, corrupt_scores])
        auroc = float(roc_auc_score(labels_auc, scores_auc))
    except ValueError:
        auroc = ""
    return ppa, delta_g, auroc


def clone_prototype_state(model):
    if not hasattr(model, "phase_prototypes"):
        return {}
    return {
        "U": model.phase_prototypes.U.detach().clone(),
        "V": model.phase_prototypes.V.detach().clone(),
    }


def normalized_state_drift(current_tensors, initial_tensors):
    if not initial_tensors:
        return ""
    numerator = 0.0
    denominator = 0.0
    for key, initial in initial_tensors.items():
        current = current_tensors[key].detach()
        numerator += torch.norm(current - initial.to(current.device)).item() ** 2
        denominator += torch.norm(initial).item() ** 2
    return float((numerator ** 0.5) / ((denominator ** 0.5) + 1e-8))


def run_one_dataset_seed(args, dataset_name: str, seed: int) -> dict:
    set_seed(seed)
    started = time.perf_counter()
    data_path = Path(args.data_path)
    dataset_dir = data_path / dataset_name
    result = empty_result(args, dataset_name, seed, status="running")
    result["train_file"] = find_split_file(dataset_dir, dataset_name, "train")
    result["test_file"] = find_split_file(dataset_dir, dataset_name, "test")

    full_train_loader = get_dataloader(
        args.data_path,
        dataset_name,
        batch_size=args.batch_size,
        split="train",
        normalize=not args.no_normalize,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    train_dataset_full = full_train_loader.dataset
    train_indices, val_indices = stratified_train_val_indices(train_dataset_full.y, args.val_fraction, seed)
    train_indices = apply_train_fraction(train_indices, train_dataset_full.y, args.train_fraction, seed)
    train_loader = make_subset_loader(
        train_dataset_full,
        train_indices,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=not args.no_drop_last,
    )
    val_loader = make_subset_loader(
        train_dataset_full,
        val_indices,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    test_loader = get_dataloader(
        args.data_path,
        dataset_name,
        batch_size=args.batch_size,
        split="test",
        normalize=not args.no_normalize,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    dataset_train = train_dataset_full
    result.update(
        {
            "train_samples": len(train_loader.dataset),
            "val_samples": len(val_loader.dataset),
            "test_samples": len(test_loader.dataset),
            "n_classes": dataset_train.n_classes,
            "n_channels": dataset_train.n_channels,
            "seq_length": dataset_train.seq_length,
        }
    )

    model = make_model(args, dataset_train)
    result["model_params"] = count_parameters(model)
    initial_prototypes = clone_prototype_state(model)
    initial_memory = (
        model.confirmed_memory.confirmed_memory.detach().clone()
        if hasattr(model, "confirmed_memory")
        else None
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    losses = make_losses(args)
    cf_generator = None if args.no_cf or args.model != "veto" else CounterfactualGenerator(min_phase_length=3)
    loss_weights = {
        "cls": 1.0,
        "cf": 0.0 if args.no_cf else args.cf_weight,
        "trans": 0.0 if args.no_trans else args.trans_weight,
        "mem": 0.0 if args.no_memory else args.mem_weight,
        "tf": 0.0,
    }

    best_val_macro_f1 = float("-inf")
    best_val_acc = 0.0
    best_state = None
    best_epoch = 0
    train_seconds = 0.0
    eval_seconds = 0.0
    final_train_loss = ""
    final_train_acc = ""
    final_val_acc = ""
    final_val_macro_f1 = ""
    final_test_acc = ""
    final_macro_f1 = ""

    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, args.epochs + 1):
        train_start = time.perf_counter()
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
        train_seconds += time.perf_counter() - train_start

        should_eval = epoch == args.epochs or epoch % args.eval_every == 0
        if should_eval:
            eval_start = time.perf_counter()
            val_acc, preds, labels = evaluate(model, val_loader, args.device)
            eval_seconds += time.perf_counter() - eval_start
            macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
            final_val_acc = val_acc
            final_val_macro_f1 = macro_f1
            if macro_f1 > best_val_macro_f1 or (
                np.isclose(macro_f1, best_val_macro_f1) and val_acc > best_val_acc
            ):
                best_val_macro_f1 = macro_f1
                best_val_acc = val_acc
                best_epoch = epoch
                best_state = state_dict_to_cpu(model)
        else:
            val_acc = final_val_acc
            macro_f1 = final_val_macro_f1
        scheduler.step()

        final_train_loss = train_loss
        final_train_acc = train_acc

        result.update(
            {
                "epochs_completed": epoch,
                "final_train_loss": final_train_loss,
                "final_train_acc": final_train_acc,
                "final_val_acc": final_val_acc,
                "final_val_macro_f1": final_val_macro_f1,
                "best_val_acc": best_val_acc if best_val_macro_f1 > float("-inf") else "",
                "best_val_macro_f1": best_val_macro_f1 if best_val_macro_f1 > float("-inf") else "",
                "final_test_acc": final_test_acc,
                "final_macro_f1": final_macro_f1,
                "selected_test_acc": final_test_acc,
                "selected_test_macro_f1": final_macro_f1,
                "best_test_acc": "",
                "best_macro_f1": "",
                "best_epoch": best_epoch,
                "train_seconds": train_seconds,
                "eval_seconds": eval_seconds,
                "total_seconds": time.perf_counter() - started,
            }
        )
        eval_text = "skip" if not should_eval else f"{val_acc:.4f}"
        f1_text = "skip" if not should_eval else f"{macro_f1:.4f}"
        best_val_text = f"{best_val_macro_f1:.4f}" if np.isfinite(best_val_macro_f1) else "n/a"
        print(
            f"[{dataset_name} seed={seed}] epoch {epoch}/{args.epochs}: "
            f"val_acc={eval_text}, val_f1={f1_text}, best_val_f1={best_val_text}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    eval_start = time.perf_counter()
    test_acc, preds, labels = evaluate(model, test_loader, args.device)
    eval_seconds += time.perf_counter() - eval_start
    test_macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    final_test_acc = test_acc
    final_macro_f1 = test_macro_f1
    result.update(
        {
            "selected_test_acc": final_test_acc,
            "selected_test_macro_f1": final_macro_f1,
            "final_test_acc": final_test_acc,
            "final_macro_f1": final_macro_f1,
            "best_test_acc": "",
            "best_macro_f1": "",
            "eval_seconds": eval_seconds,
            "total_seconds": time.perf_counter() - started,
        }
    )
    latency_ms, throughput = measure_inference(model, test_loader, args.device)
    diagnostic_ppa, diagnostic_delta_g, diagnostic_auroc = compute_order_diagnostics(
        model,
        test_loader,
        args.device,
        args.diagnostic_batches,
    )
    if hasattr(model, "phase_prototypes"):
        prototype_drift = normalized_state_drift(
            {"U": model.phase_prototypes.U, "V": model.phase_prototypes.V},
            initial_prototypes,
        )
    else:
        prototype_drift = ""
    if initial_memory is not None:
        current_memory = model.confirmed_memory.confirmed_memory.detach()
        memory_drift = float(
            torch.norm(current_memory - initial_memory.to(current_memory.device)).item()
            / (torch.norm(initial_memory).item() + 1e-8)
        )
    else:
        memory_drift = ""
    peak_gpu = (
        torch.cuda.max_memory_allocated() / (1024**2)
        if args.device.startswith("cuda") and torch.cuda.is_available()
        else 0.0
    )
    result.update(
        {
            "status": "ok",
            "diagnostic_ppa": diagnostic_ppa,
            "diagnostic_delta_g": diagnostic_delta_g,
            "diagnostic_auroc": diagnostic_auroc,
            "prototype_drift": prototype_drift,
            "memory_drift": memory_drift,
            "single_sample_latency_ms": latency_ms,
            "throughput_samples_per_sec": throughput,
            "peak_gpu_memory_mb": peak_gpu,
            "total_seconds": time.perf_counter() - started,
        }
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark official TRAIN/TEST splits.")
    parser.add_argument("--experiment", type=str, default="main")
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=[42])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model", choices=["veto", *BASELINE_MODEL_NAMES], default="veto")
    parser.add_argument("--backbone", choices=["inception", "resnet", "fcn"], default="inception")
    parser.add_argument("--n_phases", type=int, default=5)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--window_size", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--head_mode", choices=["backbone", "orderless", "prototype", "hmm", "veto"], default="veto")
    parser.add_argument("--prototype_mode", choices=["class", "shared", "full"], default="class")
    parser.add_argument("--transition_mode", choices=["uniform", "free", "class_independent", "neural", "attention"], default="neural")
    parser.add_argument("--train_fraction", type=float, default=1.0)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--path_score_mode", choices=["gain", "raw"], default="gain")
    parser.add_argument("--path_weight_override", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--no_normalize", action="store_true")
    parser.add_argument("--no_cf", action="store_true")
    parser.add_argument("--no_trans", action="store_true")
    parser.add_argument("--no_memory", action="store_true")
    parser.add_argument("--memory_update_mode", choices=["confirmed", "direct_ema"], default="confirmed")
    parser.add_argument("--no_uncertainty", action="store_true")
    parser.add_argument("--no_drop_last", action="store_true")
    parser.add_argument("--cf_weight", type=float, default=0.5)
    parser.add_argument("--trans_weight", type=float, default=0.1)
    parser.add_argument("--mem_weight", type=float, default=0.1)
    parser.add_argument("--cf_start_epoch", type=int, default=5)
    parser.add_argument("--diagnostic_batches", type=int, default=3)
    parser.add_argument("--output", type=str, default="diagnostics/official_split_benchmark.csv")
    parser.add_argument("--json_output", type=str, default="diagnostics/official_split_benchmark.json")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    if not (0 < args.train_fraction <= 1):
        raise ValueError("--train_fraction must be in (0, 1]")
    if not (0 < args.val_fraction < 1):
        raise ValueError("--val_fraction must be in (0, 1)")

    csv_path = Path(args.output)
    json_path = Path(args.json_output)
    dataset_names = args.datasets or discover_datasets(Path(args.data_path))

    results = []
    print(f"Benchmarking {len(dataset_names)} datasets x {len(args.seeds)} seeds")
    for dataset_idx, dataset_name in enumerate(dataset_names, start=1):
        for seed in args.seeds:
            print(f"\n=== [{dataset_idx}/{len(dataset_names)}] {dataset_name} seed={seed} ===")
            try:
                result = run_one_dataset_seed(args, dataset_name, seed)
            except Exception as exc:
                error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
                print(f"[{dataset_name} seed={seed}] failed: {error}")
                result = empty_result(args, dataset_name, seed, "load_or_train_failed", error)
                dataset_dir = Path(args.data_path) / dataset_name
                result["train_file"] = find_split_file(dataset_dir, dataset_name, "train")
                result["test_file"] = find_split_file(dataset_dir, dataset_name, "test")
            results.append(result)
            write_results(results, csv_path, json_path)

    print(f"\nDone. Wrote {csv_path} and {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

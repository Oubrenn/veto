"""Real-data order corruption diagnostics for trained PhasePathNet models."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import CounterfactualGenerator, PhasePathDataset
from models import PhasePathNet
from train import reconstruct_from_windows


def build_model_from_checkpoint(checkpoint, dataset, args):
    ckpt_args = checkpoint.get("args", {})
    state_dict = checkpoint["model_state_dict"]
    transition_mode = ckpt_args.get("transition_mode")
    if transition_mode is None:
        if "phase_graph.transition_matrix" in state_dict:
            transition_mode = "uniform"
        elif "phase_graph.transition_logits" in state_dict:
            transition_mode = (
                "class_independent"
                if state_dict["phase_graph.transition_logits"].ndim == 2
                else "free"
            )
        elif "phase_graph.phase_query" in state_dict:
            transition_mode = "attention"
        elif "phase_graph.transition_generator.0.weight" in state_dict:
            transition_mode = "neural"
        else:
            transition_mode = args.transition_mode
    model = PhasePathNet(
        n_classes=dataset.n_classes,
        n_channels=dataset.n_channels,
        seq_length=dataset.seq_length,
        n_phases=int(ckpt_args.get("n_phases", args.n_phases)),
        embed_dim=int(ckpt_args.get("embed_dim", args.embed_dim)),
        window_size=ckpt_args.get("window_size", args.window_size),
        stride=ckpt_args.get("stride", args.stride),
        backbone=ckpt_args.get("backbone", args.backbone),
        use_tf_branch=False,
        use_memory=not bool(ckpt_args.get("no_memory", False)),
        transition_mode=transition_mode,
        prototype_mode=ckpt_args.get("prototype_mode", args.prototype_mode),
        head_mode=ckpt_args.get("head_mode", args.head_mode),
        path_score_mode=ckpt_args.get("path_score_mode", args.path_score_mode),
        use_uncertainty=not bool(ckpt_args.get("no_uncertainty", False)),
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(
            "Warning: loaded checkpoint with non-strict keys; "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
    return model


def corrupt_time_blocks(x, strategy: str, severity: int) -> torch.Tensor:
    x = x.clone()
    B, T, C = x.shape
    n_blocks = max(4, severity + 2)
    block_size = max(1, T // n_blocks)
    blocks = [x[:, i * block_size : min((i + 1) * block_size, T), :] for i in range(n_blocks)]
    if n_blocks * block_size < T:
        blocks.append(x[:, n_blocks * block_size :, :])

    if strategy == "block_swap":
        i, j = 1, min(len(blocks) - 2, 1 + severity)
        blocks[i], blocks[j] = blocks[j], blocks[i]
    elif strategy == "segment_reversal":
        end = min(len(blocks) - 1, 1 + severity)
        blocks[1:end] = list(reversed(blocks[1:end]))
    elif strategy == "non_adjacent_exchange":
        i = 1
        j = min(len(blocks) - 2, 2 + severity)
        blocks[i], blocks[j] = blocks[j], blocks[i]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return torch.cat(blocks, dim=1)[:, :T, :]


def corrupt_latent_windows(model, x, strategy: str, severity: int) -> torch.Tensor:
    windows = model.window_partitioner.partition(x)
    generator = CounterfactualGenerator(min_phase_length=max(1, severity))
    if strategy == "block_swap":
        cf_windows, _ = generator.generate(windows, strategy="swap")
    elif strategy == "segment_reversal":
        cf_windows = windows.flip(dims=[1])
    elif strategy == "non_adjacent_exchange":
        cf_windows = windows.clone()
        if cf_windows.shape[1] >= 4:
            cf_windows[:, 1, :, :], cf_windows[:, -2, :, :] = (
                cf_windows[:, -2, :, :].clone(),
                cf_windows[:, 1, :, :].clone(),
            )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return reconstruct_from_windows(
        cf_windows,
        seq_length=x.shape[1],
        stride=model.window_partitioner.stride,
    )


def batch_scores(model, x):
    out = model(x)
    probs = torch.softmax(out["logits"], dim=-1)
    pred = probs.argmax(dim=-1)
    conf = probs.max(dim=-1).values
    return {
        "logits": out["logits"],
        "path": out["path_log_probs"],
        "iid": out["iid_log_probs"],
        "gain": out["transition_gain"],
        "proto": out["proto_score"],
        "pred": pred,
        "conf": conf,
    }


def latent_scores(model, embeddings):
    out = model.forward_from_embeddings(embeddings)
    probs = torch.softmax(out["logits"], dim=-1)
    return {
        "logits": out["logits"],
        "path": out["path_log_probs"],
        "iid": out["iid_log_probs"],
        "gain": out["transition_gain"],
        "proto": out["proto_score"],
        "pred": probs.argmax(dim=-1),
        "conf": probs.max(dim=-1).values,
    }


def permute_latent_embeddings(embeddings, strategy: str, severity: int, generator):
    if embeddings.shape[1] <= 1:
        return embeddings.clone()
    n_steps = embeddings.shape[1]
    orders = []
    for _ in range(embeddings.shape[0]):
        order = torch.arange(n_steps, device=embeddings.device)
        if strategy == "block_swap":
            i, j = 1, min(n_steps - 1, 1 + severity)
            order[i], order[j] = order[j].clone(), order[i].clone()
        elif strategy == "segment_reversal":
            end = min(n_steps, 1 + severity + 1)
            order[1:end] = torch.flip(order[1:end], dims=[0])
        elif strategy == "non_adjacent_exchange":
            i = 1
            j = min(n_steps - 1, 2 + severity)
            order[i], order[j] = order[j].clone(), order[i].clone()
        elif strategy == "shuffle":
            order = torch.randperm(n_steps, device=embeddings.device, generator=generator)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        orders.append(order)
    return torch.stack([embeddings[idx, order] for idx, order in enumerate(orders)], dim=0)


def gather_true_class(stacked_scores, labels):
    labels = labels.view(1, -1, 1).expand(stacked_scores.shape[0], -1, 1)
    return stacked_scores.gather(dim=2, index=labels).squeeze(2)


def diagnose(args):
    dataset = PhasePathDataset(args.data_path, args.dataset, split=args.split, normalize=not args.no_normalize)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = build_model_from_checkpoint(checkpoint, dataset, args).to(args.device)
    model.eval()

    rows = []
    strategies = args.strategies
    with torch.no_grad():
        for level in args.levels:
            for strategy in strategies:
                severity_deltas = []
                for severity in args.severities:
                    valid_scores = []
                    corrupt_scores = []
                    valid_iid_scores = []
                    corrupt_iid_scores = []
                    valid_proto_scores = []
                    corrupt_proto_scores = []
                    valid_gain_scores = []
                    corrupt_gain_scores = []
                    valid_conf = []
                    corrupt_conf = []
                    correct_valid = []
                    correct_corrupt = []
                    flip_flags = []

                    for batch_idx, (x, labels) in enumerate(loader):
                        x = x.to(args.device)
                        labels = labels.to(args.device)
                        if args.max_batches and batch_idx >= args.max_batches:
                            break
                        if level == "latent":
                            windows = model.window_partitioner.partition(x)
                            embeddings = model.encoder(windows)
                            valid = latent_scores(model, embeddings)
                            repeated = []
                            for repeat_idx in range(args.n_permutations):
                                generator = torch.Generator(device=args.device)
                                generator.manual_seed(args.seed + repeat_idx)
                                permuted = permute_latent_embeddings(
                                    embeddings,
                                    strategy,
                                    severity,
                                    generator,
                                )
                                repeated.append(latent_scores(model, permuted))
                            corrupt = {
                                key: torch.stack([item[key] for item in repeated], dim=0)
                                for key in ["path", "iid", "gain", "proto", "logits", "conf", "pred"]
                            }
                        elif level == "raw":
                            valid = batch_scores(model, x)
                            x_corrupt = corrupt_time_blocks(x, strategy, severity)
                            corrupt = batch_scores(model, x_corrupt)
                        else:
                            raise ValueError(f"Unknown level: {level}")
                        idx = torch.arange(labels.shape[0], device=args.device)
                        valid_scores.extend(valid["path"][idx, labels].detach().cpu().numpy())
                        valid_iid_scores.extend(valid["iid"][idx, labels].detach().cpu().numpy())
                        valid_gain_scores.extend(valid["gain"][idx, labels].detach().cpu().numpy())
                        valid_proto_scores.extend(valid["proto"][idx, labels].detach().cpu().numpy())
                        valid_conf.extend(valid["conf"].detach().cpu().numpy())
                        correct_valid.extend((valid["pred"] == labels).detach().cpu().numpy())

                        if level == "latent":
                            corrupt_scores.extend(gather_true_class(corrupt["path"], labels).reshape(-1).detach().cpu().numpy())
                            corrupt_iid_scores.extend(gather_true_class(corrupt["iid"], labels).reshape(-1).detach().cpu().numpy())
                            corrupt_gain_scores.extend(gather_true_class(corrupt["gain"], labels).reshape(-1).detach().cpu().numpy())
                            corrupt_proto_scores.extend(gather_true_class(corrupt["proto"], labels).reshape(-1).detach().cpu().numpy())
                            corrupt_conf.extend(corrupt["conf"].reshape(-1).detach().cpu().numpy())
                            correct_corrupt.extend((corrupt["pred"] == labels.unsqueeze(0)).reshape(-1).detach().cpu().numpy())
                            flip_flags.extend((corrupt["pred"] != valid["pred"].unsqueeze(0)).reshape(-1).detach().cpu().numpy())
                        else:
                            corrupt_scores.extend(corrupt["path"][idx, labels].detach().cpu().numpy())
                            corrupt_iid_scores.extend(corrupt["iid"][idx, labels].detach().cpu().numpy())
                            corrupt_gain_scores.extend(corrupt["gain"][idx, labels].detach().cpu().numpy())
                            corrupt_proto_scores.extend(corrupt["proto"][idx, labels].detach().cpu().numpy())
                            corrupt_conf.extend(corrupt["conf"].detach().cpu().numpy())
                            correct_corrupt.extend((corrupt["pred"] == labels).detach().cpu().numpy())
                            flip_flags.extend((corrupt["pred"] != valid["pred"]).detach().cpu().numpy())

                    valid_scores = np.asarray(valid_scores)
                    corrupt_scores = np.asarray(corrupt_scores)
                    valid_iid_scores = np.asarray(valid_iid_scores)
                    corrupt_iid_scores = np.asarray(corrupt_iid_scores)
                    valid_gain_scores = np.asarray(valid_gain_scores)
                    corrupt_gain_scores = np.asarray(corrupt_gain_scores)
                    valid_proto_scores = np.asarray(valid_proto_scores)
                    corrupt_proto_scores = np.asarray(corrupt_proto_scores)
                    valid_conf = np.asarray(valid_conf)
                    corrupt_conf = np.asarray(corrupt_conf)
                    if level == "latent" and args.n_permutations > 1:
                        valid_scores_for_delta = np.repeat(valid_scores, args.n_permutations)
                        valid_iid_for_delta = np.repeat(valid_iid_scores, args.n_permutations)
                        valid_gain_for_delta = np.repeat(valid_gain_scores, args.n_permutations)
                        valid_proto_for_delta = np.repeat(valid_proto_scores, args.n_permutations)
                    else:
                        valid_scores_for_delta = valid_scores
                        valid_iid_for_delta = valid_iid_scores
                        valid_gain_for_delta = valid_gain_scores
                        valid_proto_for_delta = valid_proto_scores
                    delta_path = valid_scores_for_delta - corrupt_scores
                    delta_gain = valid_gain_for_delta - corrupt_gain_scores
                    ppa = float(np.mean(delta_path > 0))
                    labels_auc = np.concatenate([np.ones_like(valid_gain_for_delta), np.zeros_like(corrupt_gain_scores)])
                    scores_auc = np.concatenate([valid_gain_for_delta, corrupt_gain_scores])
                    auroc = float(roc_auc_score(labels_auc, scores_auc))
                    conf_drop = float(np.mean(np.repeat(valid_conf, args.n_permutations) - corrupt_conf)) if level == "latent" and args.n_permutations > 1 else float(np.mean(valid_conf - corrupt_conf))
                    acc_valid = float(np.mean(correct_valid))
                    acc_corrupt = float(np.mean(correct_corrupt))
                    occurrence_invariance_error = float(np.mean(np.abs(valid_proto_for_delta - corrupt_proto_scores)))
                    iid_reference_invariance_error = float(np.mean(np.abs(valid_iid_for_delta - corrupt_iid_scores)))
                    transition_gain_drop = float(np.mean(delta_gain))
                    prediction_flip_rate = float(np.mean(flip_flags))
                    severity_deltas.append(transition_gain_drop)
                    rows.append(
                        {
                            "dataset": args.dataset,
                            "checkpoint": args.checkpoint,
                            "level": level,
                            "strategy": strategy,
                            "severity": severity,
                            "ppa": ppa,
                            "delta_g": transition_gain_drop,
                            "score_margin": float(np.mean(delta_path)),
                            "valid_corrupt_auroc": auroc,
                            "occurrence_invariance_error": occurrence_invariance_error,
                            "iid_reference_invariance_error": iid_reference_invariance_error,
                            "transition_gain_drop": transition_gain_drop,
                            "confidence_drop": conf_drop,
                            "valid_accuracy": acc_valid,
                            "corrupt_accuracy": acc_corrupt,
                            "accuracy_drop": acc_valid - acc_corrupt,
                            "prediction_flip_rate": prediction_flip_rate,
                            "n_permutations": args.n_permutations if level == "latent" else 1,
                        }
                    )
                if len(args.severities) > 1:
                    auc = float(np.trapz(severity_deltas, args.severities))
                    rows.append(
                        {
                            "dataset": args.dataset,
                            "checkpoint": args.checkpoint,
                            "level": level,
                            "strategy": strategy,
                            "severity": "severity_auc",
                            "ppa": "",
                            "delta_g": auc,
                            "score_margin": auc,
                            "valid_corrupt_auroc": "",
                            "occurrence_invariance_error": "",
                            "iid_reference_invariance_error": "",
                            "transition_gain_drop": auc,
                            "confidence_drop": "",
                            "valid_accuracy": "",
                            "corrupt_accuracy": "",
                            "accuracy_drop": "",
                            "prediction_flip_rate": "",
                            "n_permutations": args.n_permutations if level == "latent" else 1,
                        }
                    )
    return rows


def write_rows(rows, output, json_output):
    output = Path(output)
    json_output = Path(json_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with json_output.open("w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Order corruption diagnostics")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_path", default="D:/Xjnproject/XJNproject/1uka/SPINNET/dataset")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--levels", nargs="*", default=["latent", "raw"])
    parser.add_argument("--strategies", nargs="*", default=["block_swap", "segment_reversal", "non_adjacent_exchange"])
    parser.add_argument("--severities", nargs="*", type=int, default=[1, 2, 3])
    parser.add_argument("--n_permutations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--no_normalize", action="store_true")
    parser.add_argument("--n_phases", type=int, default=5)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--window_size", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--backbone", choices=["inception", "resnet", "fcn"], default="inception")
    parser.add_argument("--transition_mode", choices=["uniform", "free", "class_independent", "neural", "attention"], default="neural")
    parser.add_argument("--prototype_mode", choices=["class", "shared", "full"], default="class")
    parser.add_argument("--head_mode", choices=["backbone", "orderless", "prototype", "hmm", "veto"], default="veto")
    parser.add_argument("--path_score_mode", choices=["gain", "raw"], default="gain")
    parser.add_argument("--output", default="diagnostics/order_corruption.csv")
    parser.add_argument("--json_output", default="diagnostics/order_corruption.json")
    args = parser.parse_args()

    rows = diagnose(args)
    write_rows(rows, args.output, args.json_output)
    for row in rows:
        if row["severity"] != "severity_auc":
            print(
                f"{row['level']} {row['strategy']} severity={row['severity']}: "
                f"PPA={row['ppa']:.3f}, dG={row['delta_g']:.3f}, "
                f"AUROC={row['valid_corrupt_auroc']:.3f}"
            )


if __name__ == "__main__":
    main()

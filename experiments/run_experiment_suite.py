"""Generate or run the recommended VETO experiment suite commands."""
import argparse
import shlex
import subprocess
from pathlib import Path


CORE_ABLATIONS = {
    "backbone_only": ["--head_mode", "backbone", "--no_cf", "--no_trans", "--no_memory"],
    "local_only": ["--head_mode", "veto", "--path_weight_override", "0.0", "--trans_weight", "0.0", "--cf_weight", "0.0"],
    "raw_transition": ["--path_score_mode", "raw"],
    "class_independent_transition": ["--transition_mode", "class_independent"],
    "class_specific_prototypes": ["--head_mode", "prototype", "--prototype_mode", "class", "--transition_mode", "uniform", "--no_cf", "--no_trans", "--no_memory"],
    "shared_dictionary_only": ["--head_mode", "prototype", "--prototype_mode", "shared", "--transition_mode", "uniform", "--no_cf", "--no_trans", "--no_memory"],
    "uniform_transition": ["--transition_mode", "uniform"],
    "free_transition_matrix": ["--transition_mode", "free"],
    "neural_transition_generator": ["--transition_mode", "neural"],
    "wo_counterfactual": ["--no_cf"],
    "full_rank_prototypes": ["--prototype_mode", "full"],
    "wo_confirmed_memory": ["--no_memory"],
    "direct_ema_memory": ["--memory_update_mode", "direct_ema"],
    "no_counterfactual": ["--no_cf"],
    "full_veto": ["--transition_mode", "neural", "--prototype_mode", "class", "--head_mode", "veto"],
}

GENERATOR_VARIANTS = {
    "free_transition_matrix": ["--transition_mode", "free"],
    "parameter_matched_mlp": ["--transition_mode", "attention"],
    "neural_transition_generator": ["--transition_mode", "neural"],
}

BACKBONES = ["inception", "resnet", "fcn"]
DEFAULT_MAIN_DATASETS = [
    "ArticularyWordRecognition",
    "AtrialFibrillation",
    "BasicMotions",
    "CharacterTrajectories",
    "Cricket",
    "DuckDuckGeese",
    "EigenWorms",
    "Epilepsy",
    "EthanolConcentration",
    "ERing",
    "FaceDetection",
    "FingerMovements",
    "HandMovementDirection",
    "Handwriting",
    "Heartbeat",
    "JapaneseVowels",
    "Libras",
    "LSST",
    "MotorImagery",
    "NATOPS",
    "PEMS-SF",
    "PenDigits",
    "RacketSports",
    "SelfRegulationSCP1",
    "SelfRegulationSCP2",
    "UWaveGestureLibrary",
]


def base_command(args, experiment, extra):
    output_dir = Path(args.output_dir)
    csv_path = output_dir / f"{experiment}.csv"
    json_path = output_dir / f"{experiment}.json"
    cmd = [
        args.python,
        "benchmark_official_splits.py",
        "--experiment",
        experiment,
        "--data_path",
        args.data_path,
        "--datasets",
        *args.datasets,
        "--seeds",
        *[str(seed) for seed in args.seeds],
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--device",
        args.device,
        "--output",
        str(csv_path),
        "--json_output",
        str(json_path),
    ]
    cmd.extend(extra)
    return cmd


def commands_for_suite(args):
    commands = []
    if args.suite in {"main", "all"}:
        commands.append(base_command(args, "full_veto", ["--transition_mode", "neural", "--prototype_mode", "class", "--head_mode", "veto"]))
        commands.append(base_command(args, "local_only", ["--head_mode", "veto", "--path_weight_override", "0.0", "--no_cf", "--no_trans", "--no_memory"]))
        commands.append(base_command(args, "raw_transition", ["--path_score_mode", "raw"]))
        commands.append(base_command(args, "class_independent_transition", ["--transition_mode", "class_independent"]))
    if args.suite in {"ablations", "all"}:
        for name, extra in CORE_ABLATIONS.items():
            commands.append(base_command(args, f"ablation_{name}", extra))
    if args.suite in {"generator", "all"}:
        for fraction in args.train_fractions:
            for name, extra in GENERATOR_VARIANTS.items():
                commands.append(
                    base_command(
                        args,
                        f"generator_{name}_{int(fraction * 100)}pct",
                        extra + ["--train_fraction", str(fraction)],
                    )
                )
    if args.suite in {"backbone", "all"}:
        for backbone in BACKBONES:
            commands.append(
                base_command(
                    args,
                    f"backbone_{backbone}_baseline",
                    ["--backbone", backbone, "--head_mode", "backbone", "--no_cf", "--no_trans", "--no_memory"],
                )
            )
            commands.append(
                base_command(
                    args,
                    f"backbone_{backbone}_veto",
                    ["--backbone", backbone, "--head_mode", "veto", "--transition_mode", "neural"],
                )
            )
    if args.suite in {"mechanism", "all"}:
        commands.append([
            args.python,
            "experiments\\synthetic_path_control.py",
            "--output",
            str(Path(args.output_dir) / "synthetic_path_control.csv"),
            "--json_output",
            str(Path(args.output_dir) / "synthetic_path_control.json"),
        ])
        commands.append([
            args.python,
            "experiments\\memory_pollution_stress.py",
            "--pollutions",
            "0.1",
            "0.2",
            "0.4",
            "--pollution_type",
            "mislabeled_windows",
            "--output",
            str(Path(args.output_dir) / "memory_pollution_label_noise.csv"),
            "--json_output",
            str(Path(args.output_dir) / "memory_pollution_label_noise.json"),
        ])
        commands.append([
            args.python,
            "experiments\\memory_pollution_stress.py",
            "--pollutions",
            "0.1",
            "0.2",
            "0.4",
            "--pollution_type",
            "burst_noise",
            "--output",
            str(Path(args.output_dir) / "memory_pollution_window_noise.csv"),
            "--json_output",
            str(Path(args.output_dir) / "memory_pollution_window_noise.json"),
        ])
    if args.suite in {"efficiency", "all"}:
        commands.append([
            args.python,
            "experiments\\efficiency_scaling.py",
            "--device",
            args.device,
            "--output",
            str(Path(args.output_dir) / "efficiency_scaling.csv"),
            "--json_output",
            str(Path(args.output_dir) / "efficiency_scaling.json"),
        ])
    return commands


def main():
    parser = argparse.ArgumentParser(description="Generate/run experiment suite commands")
    parser.add_argument("--suite", choices=["main", "ablations", "generator", "backbone", "mechanism", "efficiency", "all"], default="ablations")
    parser.add_argument("--python", default="D:/anaconda3/envs/TFproject/python.exe")
    parser.add_argument("--data_path", default="D:/Xjnproject/XJNproject/1uka/SPINNET/dataset")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_MAIN_DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--train_fractions", nargs="+", type=float, default=[0.1, 0.25, 0.5, 1.0])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="diagnostics/suite")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    commands = commands_for_suite(args)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    script_path = Path(args.output_dir) / f"{args.suite}_commands.ps1"
    with script_path.open("w", encoding="utf-8") as file:
        for cmd in commands:
            file.write(" ".join(shlex.quote(part) for part in cmd) + "\n")

    print(f"Wrote {len(commands)} commands to {script_path}")
    for cmd in commands:
        print(" ".join(shlex.quote(part) for part in cmd))
        if args.run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

"""Check available datasets without importing PyTorch."""
from collections import Counter
from pathlib import Path


DATA_PATH = "D:/Xjnproject/XJNproject/1uka/SPINNET/dataset"
SUPPORTED_SUFFIXES = {".npy", ".pt", ".ts"}


def find_dataset_files(dataset_dir: Path) -> list[Path]:
    return [
        item
        for item in dataset_dir.iterdir()
        if item.is_file() and item.suffix.lower() in SUPPORTED_SUFFIXES
    ]


def extension_counts(files: list[Path]) -> str:
    counts = Counter(file.suffix.lower() for file in files)
    return ", ".join(f"{suffix}:{counts[suffix]}" for suffix in sorted(counts))


def main() -> int:
    print("Scanning dataset directory...")
    data_path = Path(DATA_PATH)

    if not data_path.exists():
        print(f"Error: dataset path does not exist: {DATA_PATH}")
        return 1

    datasets = []
    for item in data_path.iterdir():
        if item.is_dir():
            files = find_dataset_files(item)
            if files:
                datasets.append((item.name, files))

    print(f"\nFound {len(datasets)} datasets:\n")
    print(f"{'Dataset':<30} {'Files':<10} {'Formats'}")
    print("-" * 60)

    for name, files in sorted(datasets):
        print(f"{name:<30} {len(files):<10} {extension_counts(files)}")

    core_datasets = [
        "Handwriting",
        "UWaveGestureLibrary",
        "SpokenArabicDigits",
        "PEMS-SF",
        "Heartbeat",
        "SelfRegulationSCP1",
        "SelfRegulationSCP2",
    ]
    dataset_names = [name for name, _ in datasets]
    available_core = [name for name in core_datasets if name in dataset_names]
    missing_core = [name for name in core_datasets if name not in dataset_names]

    print("\nCore datasets:")
    print(f"  Available: {', '.join(available_core) if available_core else 'None'}")
    print(f"  Missing: {', '.join(missing_core) if missing_core else 'None'}")

    har_datasets = ["HHAR", "DSADS"]
    available_har = [name for name in har_datasets if name in dataset_names]
    print("\nNon-stationary validation datasets:")
    print(f"  Available: {', '.join(available_har) if available_har else 'None'}")

    print("\nProject status:")
    print("  Virtual Env: TFproject")
    print(f"  Data Path: {DATA_PATH}")
    print(f"  Datasets Found: {len(datasets)}")

    if available_core:
        test_dataset = available_core[0]
        print("\nSuggested smoke train command:")
        print("  Activate: TFproject\\Scripts\\activate")
        print(f"  Train: python train.py --dataset {test_dataset} --epochs 10 --batch_size 16")
    else:
        print("\nWarning: no core datasets found. Check dataset path and file formats.")

    if datasets:
        first_dataset, _ = sorted(datasets)[0]
        first_path = data_path / first_dataset
        print(f"\nExample files for '{first_dataset}':")
        for file in sorted(find_dataset_files(first_path))[:5]:
            print(f"  - {file.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

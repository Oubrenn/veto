"""Check that the UEA datasets used in the main table load with official splits."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import get_dataloader


DATA_PATH = "D:/Xjnproject/XJNproject/1uka/SPINNET/dataset"

TABLE_DATASETS = [
    "ArticularyWordRecognition",
    "AtrialFibrillation",
    "BasicMotions",
    "CharacterTrajectories",
    "Cricket",
    "DuckDuckGeese",
    "EigenWorms",
    "Epilepsy",
    "ERing",
    "EthanolConcentration",
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
    "PenDigits",
    "PEMS-SF",
    "PhonemeSpectra",
    "RacketSports",
    "SelfRegulationSCP1",
    "SelfRegulationSCP2",
]


def main() -> int:
    print("dataset,train,test,n_classes,n_channels,seq_length,status")
    failed = 0
    for dataset in TABLE_DATASETS:
        try:
            train_loader = get_dataloader(
                DATA_PATH,
                dataset,
                batch_size=4,
                split="train",
                shuffle=False,
                num_workers=0,
                drop_last=False,
            )
            test_loader = get_dataloader(
                DATA_PATH,
                dataset,
                batch_size=4,
                split="test",
                shuffle=False,
                num_workers=0,
                drop_last=False,
            )
            ds = train_loader.dataset
            print(
                f"{dataset},{len(train_loader.dataset)},{len(test_loader.dataset)},"
                f"{ds.n_classes},{ds.n_channels},{ds.seq_length},ok"
            )
        except Exception as exc:  # pragma: no cover - command-line diagnostic
            failed += 1
            print(f"{dataset},,,,,,ERROR: {type(exc).__name__}: {exc}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

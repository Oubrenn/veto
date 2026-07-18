# VETO

VETO is a PyTorch research codebase for multivariate time-series classification with phase-path modeling, memory modules, uncertainty estimation, and experiment utilities.

## Repository Structure

```text
.
|-- configs/              YAML configuration files
|-- experiments/          Benchmark, ablation, diagnostic, and artifact scripts
|-- losses/               Training objectives
|-- models/               Model components and neural backbones
|-- scripts/              Utility scripts for evaluation, checks, and table generation
|-- tests/                Pytest test suite
|-- utils/                Shared utilities
|-- train.py              Main training entry point
|-- test.py               Checkpoint evaluation entry point
|-- test_dataloader.py    Dataloader smoke test
|-- requirements.txt      Python dependencies
|-- README.md             Project overview
```

## Quick Start

```bash
pip install -r requirements.txt
python train.py --dataset Handwriting --epochs 10 --batch_size 16
python test.py --checkpoint checkpoints/Handwriting_best_val.pth --dataset Handwriting
python -m pytest tests -q
```

## Notes

- Provide datasets locally before running training or evaluation.
- Keep checkpoints and generated outputs outside version control.
- Use the scripts in `experiments/` and `scripts/` for reproducibility workflows.

"""Generate all four key figures for the paper.

Figure 4: Critical Difference Diagram
Figure 5: Learned Phase-Path Evidence
Figure 6: Counterfactual Order Verification
Figure 7: Sensitivity and Memory Stability
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats
from scipy.stats import friedmanchisquare
from tqdm import tqdm

# Import project modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import get_dataloader
from models import PhasePathNet


# ============================================================================
# Figure 4: Critical Difference Diagram
# ============================================================================

def compute_average_ranks(results_df: pd.DataFrame, datasets: List[str], methods: List[str]) -> Dict[str, float]:
    """Compute average rank for each method across datasets."""
    ranks = {method: [] for method in methods}

    for dataset in datasets:
        dataset_results = {}
        for method in methods:
            acc = results_df[(results_df['method'] == method) &
                           (results_df['dataset'] == dataset)]['accuracy'].values
            if len(acc) > 0:
                dataset_results[method] = acc[0]
            else:
                dataset_results[method] = np.nan

        # Rank methods for this dataset (higher accuracy = lower rank number)
        valid_methods = [m for m in methods if not np.isnan(dataset_results[m])]
        if len(valid_methods) > 0:
            sorted_methods = sorted(valid_methods, key=lambda m: dataset_results[m], reverse=True)
            for rank, method in enumerate(sorted_methods, start=1):
                ranks[method].append(rank)

    # Compute average ranks
    avg_ranks = {}
    for method in methods:
        if len(ranks[method]) > 0:
            avg_ranks[method] = np.mean(ranks[method])
        else:
            avg_ranks[method] = np.nan

    return avg_ranks


def nemenyi_critical_distance(num_methods: int, num_datasets: int, alpha: float = 0.05) -> float:
    """Calculate Nemenyi critical distance."""
    # Critical values for alpha=0.05 (approximate)
    q_alpha_values = {
        5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102,
        10: 3.164, 11: 3.219, 12: 3.268, 15: 3.389, 20: 3.578
    }

    if num_methods in q_alpha_values:
        q_alpha = q_alpha_values[num_methods]
    else:
        # Linear interpolation for missing values
        q_alpha = 2.5 + (num_methods - 5) * 0.05

    cd = q_alpha * np.sqrt(num_methods * (num_methods + 1) / (6.0 * num_datasets))
    return cd


def plot_critical_difference_diagram(avg_ranks: Dict[str, float],
                                    cd_value: float,
                                    output_path: Path,
                                    highlight_method: str = 'VETO'):
    """Plot critical difference diagram."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Sort methods by average rank
    sorted_methods = sorted(avg_ranks.items(), key=lambda x: x[1])
    methods = [m[0] for m in sorted_methods]
    ranks = [m[1] for m in sorted_methods]

    n_methods = len(methods)
    y_positions = np.arange(n_methods)

    # Plot ranks
    for i, (method, rank) in enumerate(zip(methods, ranks)):
        color = 'darkred' if method == highlight_method else 'black'
        weight = 'bold' if method == highlight_method else 'normal'
        ax.plot(rank, y_positions[i], 'o', color=color, markersize=10)
        ax.text(rank + 0.1, y_positions[i], method,
               fontsize=10, va='center', color=color, weight=weight)

    # Draw CD bars for methods that are not significantly different
    cd_groups = []
    for i in range(n_methods):
        for j in range(i + 1, n_methods):
            if abs(ranks[i] - ranks[j]) <= cd_value:
                cd_groups.append((i, j))

    # Draw horizontal lines for non-significant differences
    for i, j in cd_groups:
        y_mid = (y_positions[i] + y_positions[j]) / 2
        ax.plot([ranks[i], ranks[j]], [y_mid, y_mid],
               'k-', linewidth=2, alpha=0.3)

    ax.set_xlabel('Average Rank (lower is better)', fontsize=12)
    ax.set_yticks([])
    ax.set_title(f'Critical Difference Diagram (CD={cd_value:.3f})', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_xaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 4 to {output_path}")


# ============================================================================
# Figure 5: Learned Phase-Path Evidence
# ============================================================================

def extract_phase_path_data(model: PhasePathNet,
                           dataloader,
                           device: torch.device,
                           class_labels: List[int] = [0, 1],
                           n_samples_per_class: int = 1) -> Dict:
    """Extract responsibility heatmap and transition matrices."""
    model.eval()

    samples_data = {c: [] for c in class_labels}

    with torch.no_grad():
        for x, labels in dataloader:
            x = x.to(device)
            labels = labels.to(device)

            # Get model outputs
            output = model(x, labels)

            # Get phase assignments (responsibility)
            windows = model.window_partitioner.partition(x)
            embeddings = model.encoder(windows)
            proto_output = model.phase_prototypes(embeddings)
            assign_output = model.phase_assignment(
                proto_output['template_dist'],
                proto_output['subspace_residual']
            )

            phase_assignment = assign_output['phase_assignment']  # [B, N, C, K]

            # Get transition matrices
            graph_output = model.phase_graph()
            transition_matrices = graph_output['transition_matrices']  # [C, K, K]

            # Collect samples
            for i in range(x.size(0)):
                label = labels[i].item()
                if label in class_labels and len(samples_data[label]) < n_samples_per_class:
                    # Get correct class prediction confidence
                    pred_class = output['logits'][i].argmax().item()
                    if pred_class == label:
                        samples_data[label].append({
                            'x': x[i].cpu().numpy(),
                            'responsibility': phase_assignment[i, :, label, :].cpu().numpy(),
                            'transition_matrix': transition_matrices[label].cpu().numpy(),
                            'label': label
                        })

            # Check if we have enough samples
            if all(len(samples_data[c]) >= n_samples_per_class for c in class_labels):
                break

    return samples_data


def plot_phase_path_evidence(samples_data: Dict,
                            output_path: Path,
                            dataset_name: str = "MotorImagery"):
    """Plot phase-path evidence visualization."""
    n_classes = len(samples_data)
    fig = plt.figure(figsize=(18, 6 * n_classes))
    gs = GridSpec(n_classes, 3, figure=fig, hspace=0.3, wspace=0.3)

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for row, (class_idx, class_samples) in enumerate(samples_data.items()):
        if len(class_samples) == 0:
            continue

        sample = class_samples[0]  # Use first sample
        x_data = sample['x']  # [T, C]
        responsibility = sample['responsibility']  # [N, K]
        transition_matrix = sample['transition_matrix']  # [K, K]

        # (a) Time series
        ax_ts = fig.add_subplot(gs[row, 0])
        # Plot first 5 channels only to avoid clutter
        n_channels_to_plot = min(5, x_data.shape[1])
        for c in range(n_channels_to_plot):
            ax_ts.plot(x_data[:, c], alpha=0.7, label=f'Ch{c+1}')
        ax_ts.set_xlabel('Time')
        ax_ts.set_ylabel('Amplitude')
        ax_ts.set_title(f'Class {class_idx} Time Series', fontsize=12, weight='bold')
        ax_ts.legend(loc='upper right', fontsize=8)
        ax_ts.grid(True, alpha=0.3)

        # (b) Responsibility heatmap
        ax_heat = fig.add_subplot(gs[row, 1])
        im = ax_heat.imshow(responsibility.T, aspect='auto', cmap='YlOrRd',
                           interpolation='nearest')
        ax_heat.set_xlabel('Temporal Windows')
        ax_heat.set_ylabel('Phase Prototypes')
        ax_heat.set_title(f'Class {class_idx} Responsibility Heatmap',
                         fontsize=12, weight='bold')
        plt.colorbar(im, ax=ax_heat, label='Responsibility')

        # Add top-1 phase path
        top1_path = responsibility.argmax(axis=1)
        ax_path = ax_heat.twiny()
        ax_path.set_xlim(ax_heat.get_xlim())
        for t in range(len(top1_path)):
            ax_path.scatter(t, -0.5, c=[colors[top1_path[t]]],
                          marker='s', s=100, edgecolors='black')
        ax_path.set_xticks([])

        # (c) Transition matrix
        ax_trans = fig.add_subplot(gs[row, 2])
        im_trans = ax_trans.imshow(transition_matrix, cmap='Blues',
                                   vmin=0, vmax=1, aspect='auto')
        ax_trans.set_xlabel('To Phase')
        ax_trans.set_ylabel('From Phase')
        ax_trans.set_title(f'Class {class_idx} Transition Matrix',
                          fontsize=12, weight='bold')
        plt.colorbar(im_trans, ax=ax_trans, label='Probability')

        # Annotate strong transitions
        K = transition_matrix.shape[0]
        for i in range(K):
            for j in range(K):
                if transition_matrix[i, j] > 0.2:
                    ax_trans.text(j, i, f'{transition_matrix[i, j]:.2f}',
                                ha='center', va='center', fontsize=8)

    plt.suptitle(f'Phase-Path Evidence: {dataset_name}', fontsize=16, weight='bold')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 5 to {output_path}")


# ============================================================================
# Figure 6: Counterfactual Order Verification
# ============================================================================

def compute_order_gap(model: PhasePathNet,
                     dataloader,
                     device: torch.device,
                     n_shuffles: int = 100) -> Dict:
    """Compute order gap by shuffling temporal windows."""
    model.eval()

    real_gains = []
    shuffled_gains_all = []

    with torch.no_grad():
        for x, labels in tqdm(dataloader, desc="Computing order gaps"):
            x = x.to(device)
            labels = labels.to(device)

            # Encode windows
            windows = model.window_partitioner.partition(x)
            embeddings = model.encoder(windows)  # [B, N, D]

            # Get real path gain
            real_output = model.forward_from_embeddings(embeddings, labels)
            if 'path_scores' in real_output:
                for i in range(x.size(0)):
                    label = labels[i].item()
                    real_gain = real_output['path_scores'][i, label].item()
                    real_gains.append(real_gain)

                    # Generate shuffled paths
                    shuffled_gains_sample = []
                    for _ in range(n_shuffles):
                        # Shuffle temporal dimension
                        perm = torch.randperm(embeddings.size(1))
                        shuffled_emb = embeddings[i:i+1, perm, :]

                        shuffled_output = model.forward_from_embeddings(
                            shuffled_emb,
                            labels[i:i+1]
                        )
                        shuffled_gain = shuffled_output['path_scores'][0, label].item()
                        shuffled_gains_sample.append(shuffled_gain)

                    shuffled_gains_all.append(shuffled_gains_sample)

    order_gaps = []
    for real, shuffled_list in zip(real_gains, shuffled_gains_all):
        order_gap = real - np.mean(shuffled_list)
        order_gaps.append(order_gap)

    return {
        'real_gains': real_gains,
        'shuffled_gains': shuffled_gains_all,
        'order_gaps': order_gaps,
        'mean_order_gap': np.mean(order_gaps),
        'std_order_gap': np.std(order_gaps)
    }


def plot_order_verification(order_data: Dict[str, Dict],
                           output_path: Path,
                           datasets: List[str]):
    """Plot counterfactual order verification."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) Shuffled gain distribution for 2 representative datasets
    ax1 = axes[0]
    representative_datasets = datasets[:2]

    positions = []
    for i, dataset in enumerate(representative_datasets):
        data = order_data[dataset]
        real_gains = data['real_gains']
        shuffled_gains = data['shuffled_gains']

        # Flatten shuffled gains
        all_shuffled = [g for sample_list in shuffled_gains for g in sample_list]

        pos = i * 2
        positions.append(pos)

        # Violin plot for shuffled distribution
        parts = ax1.violinplot([all_shuffled], positions=[pos],
                              widths=0.8, showmeans=False, showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor('lightblue')
            pc.set_alpha(0.7)

        # Real gain as scatter
        ax1.scatter([pos] * len(real_gains), real_gains,
                   color='darkred', s=30, alpha=0.6, label='Real' if i == 0 else '')

        # Mean shuffled as dashed line
        mean_shuffled = np.mean(all_shuffled)
        ax1.hlines(mean_shuffled, pos - 0.4, pos + 0.4,
                  colors='blue', linestyles='dashed', linewidth=2,
                  label='Shuffled mean' if i == 0 else '')

    ax1.set_xticks(positions)
    ax1.set_xticklabels(representative_datasets)
    ax1.set_ylabel('Transition Gain')
    ax1.set_title('Shuffled Gain Distribution', fontsize=12, weight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # (b) Order gap across all datasets
    ax2 = axes[1]
    x_pos = np.arange(len(datasets))
    mean_gaps = [order_data[ds]['mean_order_gap'] for ds in datasets]
    std_gaps = [order_data[ds]['std_order_gap'] for ds in datasets]

    ax2.bar(x_pos, mean_gaps, yerr=std_gaps, capsize=5,
           color='steelblue', alpha=0.8, edgecolor='black')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(datasets, rotation=45, ha='right')
    ax2.set_ylabel('Order Gap Δ_ord')
    ax2.set_title('Order Gap Across Datasets', fontsize=12, weight='bold')
    ax2.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 6 to {output_path}")


# ============================================================================
# Figure 7: Sensitivity and Memory Stability
# ============================================================================

def run_sensitivity_experiments(base_config: Dict,
                               datasets: List[str],
                               param_name: str,
                               param_values: List,
                               device: torch.device) -> pd.DataFrame:
    """Run sensitivity experiments for a parameter."""
    results = []

    for dataset in datasets:
        for param_value in param_values:
            config = base_config.copy()
            config[param_name] = param_value

            # Load data
            train_loader = get_dataloader(dataset, split='train',
                                        batch_size=config['batch_size'])
            val_loader = get_dataloader(dataset, split='val',
                                       batch_size=config['batch_size'])

            # Create model
            model = PhasePathNet(
                n_classes=config['n_classes'],
                n_channels=config['n_channels'],
                seq_length=config['seq_length'],
                n_phases=config.get('n_phases', 5),
                embed_dim=config['embed_dim']
            ).to(device)

            # Quick training (reduced epochs for speed)
            # ... training code ...

            # Evaluate
            val_acc = evaluate_model(model, val_loader, device)

            results.append({
                'dataset': dataset,
                'param': param_name,
                'value': param_value,
                'accuracy': val_acc
            })

    return pd.DataFrame(results)


def run_memory_stability_experiment(base_config: Dict,
                                   dataset: str,
                                   corruption_ratios: List[float],
                                   device: torch.device) -> pd.DataFrame:
    """Run memory stability experiment with candidate corruption."""
    results = []

    for corruption_ratio in corruption_ratios:
        # Train with direct EMA
        config_ema = base_config.copy()
        config_ema['memory_update_mode'] = 'direct_ema'
        # ... training with corruption ...

        # Train with confirmed memory
        config_confirmed = base_config.copy()
        config_confirmed['memory_update_mode'] = 'confirmed'
        # ... training with corruption ...

        results.append({
            'corruption_ratio': corruption_ratio,
            'method': 'Direct EMA',
            'prototype_drift': 0.0  # compute actual drift
        })
        results.append({
            'corruption_ratio': corruption_ratio,
            'method': 'Confirmed Memory',
            'prototype_drift': 0.0  # compute actual drift
        })

    return pd.DataFrame(results)


def evaluate_model(model, dataloader, device):
    """Quick evaluation helper."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, labels in dataloader:
            x, labels = x.to(device), labels.to(device)
            output = model(x, labels)
            pred = output['logits'].argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

    return correct / total if total > 0 else 0.0


def plot_sensitivity_and_stability(sensitivity_df: pd.DataFrame,
                                  stability_df: pd.DataFrame,
                                  output_path: Path):
    """Plot sensitivity and memory stability."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Sensitivity to K
    ax1 = axes[0, 0]
    k_data = sensitivity_df[sensitivity_df['param'] == 'n_phases']
    for dataset in k_data['dataset'].unique():
        ds_data = k_data[k_data['dataset'] == dataset]
        ax1.plot(ds_data['value'], ds_data['accuracy'],
                marker='o', label=dataset)
    ax1.set_xlabel('Number of Phases (K)')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Sensitivity to Prototype Number', fontsize=12, weight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # (b) Sensitivity to lambda_g initialization
    ax2 = axes[0, 1]
    lambda_data = sensitivity_df[sensitivity_df['param'] == 'lambda_g_init']
    for dataset in lambda_data['dataset'].unique():
        ds_data = lambda_data[lambda_data['dataset'] == dataset]
        ax2.plot(ds_data['value'], ds_data['accuracy'],
                marker='s', label=dataset)
    ax2.set_xlabel('λ_g Initial Value')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Sensitivity to Transition Weight Init', fontsize=12, weight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # (c) Sensitivity to threshold
    ax3 = axes[1, 0]
    thresh_data = sensitivity_df[sensitivity_df['param'] == 'threshold']
    for dataset in thresh_data['dataset'].unique():
        ds_data = thresh_data[thresh_data['dataset'] == dataset]
        ax3.plot(ds_data['value'], ds_data['accuracy'],
                marker='^', label=dataset)
    ax3.set_xlabel('Confirmation Threshold τ_r')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Sensitivity to Confirmation Threshold', fontsize=12, weight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # (d) Memory stability
    ax4 = axes[1, 1]
    for method in stability_df['method'].unique():
        method_data = stability_df[stability_df['method'] == method]
        ax4.plot(method_data['corruption_ratio'],
                method_data['prototype_drift'],
                marker='o', label=method, linewidth=2)
    ax4.set_xlabel('Candidate Corruption Ratio ρ')
    ax4.set_ylabel('Prototype Drift')
    ax4.set_title('Memory Stability Under Noise', fontsize=12, weight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 7 to {output_path}")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    raise RuntimeError(
        "Retired manuscript entry point. Use experiments/redraw_paper_figures.py, which "
        "requires traceable source data and emits SVG/PDF/PNG consistently."
    )
    parser = argparse.ArgumentParser(description='Generate paper figures')
    parser.add_argument('--output_dir', type=str, default='diagnostics/paper_figures',
                       help='Output directory for figures')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Data directory')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--figures', type=str, default='all',
                       help='Comma-separated list of figures to generate (4,5,6,7) or "all"')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    figures_to_generate = [4, 5, 6, 7] if args.figures == 'all' else [int(f) for f in args.figures.split(',')]

    print(f"Generating figures: {figures_to_generate}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")

    # Figure 4: Critical Difference Diagram
    if 4 in figures_to_generate:
        print("\n" + "="*80)
        print("Generating Figure 4: Critical Difference Diagram")
        print("="*80)

        # Load main results
        main_results_path = Path("diagnostics/paper_artifacts_main_11ds/tables/table_main_paper_run_main_11ds.csv")
        if main_results_path.exists():
            results_df = pd.read_csv(main_results_path)

            # Prepare data for CD diagram
            # TODO: Add baseline methods from external_baselines
            methods = ['VETO', 'HC2', 'ROCKET', 'InceptionTime', 'TimesNet',
                      'TapNet', 'PDFTime', 'MTS2Graph', 'SimTSC', 'TMA-GAT', 'MultiRocket']
            datasets = results_df['dataset'].unique().tolist()

            # Mock data for demonstration (replace with actual results)
            avg_ranks = compute_average_ranks(results_df, datasets, methods)
            cd = nemenyi_critical_distance(len(methods), len(datasets))

            plot_critical_difference_diagram(
                avg_ranks, cd,
                output_dir / 'fig4_critical_difference.pdf',
                highlight_method='VETO'
            )
        else:
            print(f"Warning: Main results file not found at {main_results_path}")

    # Figure 5: Phase-Path Evidence
    if 5 in figures_to_generate:
        print("\n" + "="*80)
        print("Generating Figure 5: Phase-Path Evidence")
        print("="*80)

        # Load trained model and extract phase-path data
        dataset_name = 'MotorImagery'
        # TODO: Load actual trained model checkpoint
        print(f"Note: Requires trained model checkpoint for {dataset_name}")
        print("Skipping Figure 5 - requires model checkpoint")

    # Figure 6: Order Verification
    if 6 in figures_to_generate:
        print("\n" + "="*80)
        print("Generating Figure 6: Counterfactual Order Verification")
        print("="*80)

        # Load order corruption data
        order_data_path = Path("diagnostics/paper_run_fast/order_corruption.csv")
        if order_data_path.exists():
            order_df = pd.read_csv(order_data_path)

            # Process and plot
            datasets = ['DuckDuckGeese', 'Handwriting', 'LSST',
                       'MotorImagery', 'SelfRegulationSCP1', 'SelfRegulationSCP2']

            # TODO: Extract actual order gap data
            print("Note: Using existing order corruption data")
            print("For full recomputation, use --recompute flag")
        else:
            print(f"Warning: Order corruption data not found at {order_data_path}")

    # Figure 7: Sensitivity and Stability
    if 7 in figures_to_generate:
        print("\n" + "="*80)
        print("Generating Figure 7: Sensitivity and Memory Stability")
        print("="*80)

        # Check for existing sensitivity data
        sensitivity_dir = Path("diagnostics/sensitivity_smoke")
        if sensitivity_dir.exists():
            print("Note: Using existing sensitivity data")
            # TODO: Load and plot sensitivity results
        else:
            print("Note: Sensitivity experiments not yet run")
            print("Run with --run_sensitivity flag to generate data")

    print("\n" + "="*80)
    print("Figure generation complete!")
    print(f"Outputs saved to: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()

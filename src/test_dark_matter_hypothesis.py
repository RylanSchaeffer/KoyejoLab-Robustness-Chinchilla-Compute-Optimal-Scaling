#!/usr/bin/env python3
"""
Test the "dark matter" hypothesis for neural scaling laws using the Chinchilla dataset.

This script loads the existing Chinchilla training data and tests whether model loss
follows a two-term additive power law:
    Loss = C1/Compute^α1 + C2/Compute^α2 + noise

Usage:
    python src/test_dark_matter_hypothesis.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Import the existing data loading function and our new dark matter analysis
import src.analyze as analyze
from src.dark_matter_scaling import DarkMatterScalingAnalysis


def load_chinchilla_data():
    """Load the Chinchilla training data."""
    return analyze.load_epoch_research_svg_extracted_data_csv()


def test_dark_matter_with_different_compute_metrics(save_results=True, results_dir="results"):
    """
    Test the dark matter hypothesis using different compute metrics.
    
    Args:
        save_results: Whether to save results to disk
        results_dir: Directory to save results
    """
    # Create results directory
    if save_results:
        Path(results_dir).mkdir(exist_ok=True, parents=True)
    
    # Load data
    print("Loading Chinchilla training data...")
    data = load_chinchilla_data()
    print(f"Loaded {len(data)} training runs")
    print(f"Data columns: {list(data.columns)}")
    
    # Define different compute metrics to test
    compute_metrics = {
        'Model Size (Parameters)': {
            'column': 'Model Size',
            'normalization': 1e9,  # Normalize by 1B parameters
            'description': 'Number of model parameters'
        },
        'Training FLOPs': {
            'column': 'Training FLOP', 
            'normalization': 1e18,  # Normalize by 1e18 FLOPs
            'description': 'Total training compute in FLOPs'
        },
        'Training Tokens': {
            'column': 'Training Tokens',
            'normalization': 1e9,   # Normalize by 1B tokens
            'description': 'Number of training tokens'
        }
    }
    
    # Store results for all metrics
    all_results = {}
    
    print("\n" + "=" * 80)
    print("TESTING DARK MATTER HYPOTHESIS WITH DIFFERENT COMPUTE METRICS")
    print("=" * 80)
    
    for metric_name, metric_info in compute_metrics.items():
        print(f"\n{'='*20} {metric_name} {'='*20}")
        print(f"Description: {metric_info['description']}")
        
        # Extract compute and loss data
        compute = data[metric_info['column']].values
        losses = data['loss'].values
        
        # Remove any NaN values
        valid_idx = ~(np.isnan(compute) | np.isnan(losses))
        compute = compute[valid_idx]
        losses = losses[valid_idx]
        
        print(f"Valid data points: {len(compute)}")
        print(f"Compute range: {compute.min():.2e} to {compute.max():.2e}")
        print(f"Loss range: {losses.min():.4f} to {losses.max():.4f}")
        
        # Create analyzer with appropriate normalization
        analyzer = DarkMatterScalingAnalysis(
            normalize_compute=True, 
            normalization_factor=metric_info['normalization']
        )
        
        # Run analysis
        try:
            metric_results_dir = os.path.join(results_dir, metric_name.replace(' ', '_').replace('(', '').replace(')', ''))
            if save_results:
                Path(metric_results_dir).mkdir(exist_ok=True, parents=True)
            
            results = analyzer.analyze_dark_matter_hypothesis(
                compute, losses, 
                save_results=save_results, 
                plot_dir=metric_results_dir if save_results else "."
            )
            
            all_results[metric_name] = results
            
        except Exception as e:
            print(f"Error analyzing {metric_name}: {e}")
            continue
    
    # Create summary comparison plot
    if len(all_results) > 1:
        create_summary_comparison_plot(all_results, save_results, results_dir)
    
    # Save summary results
    if save_results:
        save_summary_results(all_results, results_dir)
    
    return all_results


def create_summary_comparison_plot(all_results, save_results=True, results_dir="results"):
    """Create a summary plot comparing results across different compute metrics."""
    
    metrics = list(all_results.keys())
    n_metrics = len(metrics)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Dark Matter Hypothesis: Summary Across Compute Metrics', fontsize=16)
    
    # Improvement percentages
    ax1 = axes[0, 0]
    rmse_improvements = [all_results[m]['improvement_rmse_pct'] for m in metrics]
    loss_improvements = [all_results[m]['improvement_loss_pct'] for m in metrics]
    
    x = np.arange(n_metrics)
    width = 0.35
    
    ax1.bar(x - width/2, rmse_improvements, width, label='RMSE Improvement', alpha=0.7)
    ax1.bar(x + width/2, loss_improvements, width, label='Fitting Loss Improvement', alpha=0.7)
    ax1.set_xlabel('Compute Metric')
    ax1.set_ylabel('Improvement (%)')
    ax1.set_title('Two-Term Model Improvements')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.replace(' ', '\n') for m in metrics], rotation=0, ha='center')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Absolute errors
    ax2 = axes[0, 1]
    single_rmse = [all_results[m]['single_rmse'] for m in metrics]
    two_term_rmse = [all_results[m]['two_term_rmse'] for m in metrics]
    
    ax2.bar(x - width/2, single_rmse, width, label='Single Power Law', alpha=0.7)
    ax2.bar(x + width/2, two_term_rmse, width, label='Two-Term Power Law', alpha=0.7)
    ax2.set_xlabel('Compute Metric')
    ax2.set_ylabel('RMSE')
    ax2.set_title('Model RMSE Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.replace(' ', '\n') for m in metrics], rotation=0, ha='center')
    ax2.legend()
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    # Fitted exponents comparison
    ax3 = axes[1, 0]
    single_alphas = []
    two_term_alpha1s = []
    two_term_alpha2s = []
    
    for metric in metrics:
        single_alphas.append(all_results[metric]['single_params'][1])
        two_term_alpha1s.append(all_results[metric]['two_term_params'][1])
        two_term_alpha2s.append(all_results[metric]['two_term_params'][3])
    
    ax3.scatter(x, single_alphas, label='Single α', s=100, alpha=0.7, marker='o')
    ax3.scatter(x, two_term_alpha1s, label='Two-term α₁', s=100, alpha=0.7, marker='^')
    ax3.scatter(x, two_term_alpha2s, label='Two-term α₂', s=100, alpha=0.7, marker='v')
    ax3.set_xlabel('Compute Metric')
    ax3.set_ylabel('Scaling Exponent')
    ax3.set_title('Fitted Scaling Exponents')
    ax3.set_xticks(x)
    ax3.set_xticklabels([m.replace(' ', '\n') for m in metrics], rotation=0, ha='center')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Model selection criteria (AIC-like)
    ax4 = axes[1, 1]
    # Compute approximate AIC: 2k + n*ln(RSS/n) where k is number of parameters
    n_points = [len(all_results[m]['single_params']) for m in metrics]  # This is a placeholder
    single_aic_approx = [2*3 + all_results[m]['single_loss'] for m in metrics]  # 3 params
    two_term_aic_approx = [2*5 + all_results[m]['two_term_loss'] for m in metrics]  # 5 params
    
    ax4.bar(x - width/2, single_aic_approx, width, label='Single Power Law', alpha=0.7)
    ax4.bar(x + width/2, two_term_aic_approx, width, label='Two-Term Power Law', alpha=0.7)
    ax4.set_xlabel('Compute Metric')
    ax4.set_ylabel('Penalized Loss (lower is better)')
    ax4.set_title('Model Selection (with complexity penalty)')
    ax4.set_xticks(x)
    ax4.set_xticklabels([m.replace(' ', '\n') for m in metrics], rotation=0, ha='center')
    ax4.legend()
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_results:
        plt.savefig(f"{results_dir}/dark_matter_summary_comparison.png", dpi=300, bbox_inches='tight')
        print(f"Summary plot saved to {results_dir}/dark_matter_summary_comparison.png")
    
    plt.show()


def save_summary_results(all_results, results_dir="results"):
    """Save a summary of all results to CSV."""
    
    summary_data = []
    
    for metric_name, results in all_results.items():
        # Single power law parameters
        single_params = results['single_params']
        single_row = {
            'Compute_Metric': metric_name,
            'Model_Type': 'Single Power Law',
            'C1': np.exp(single_params[0]),
            'alpha1': single_params[1],
            'C2': np.nan,
            'alpha2': np.nan,
            'E': np.exp(single_params[2]),
            'RMSE': results['single_rmse'],
            'MAE': results['single_mae'],
            'Fitting_Loss': results['single_loss']
        }
        summary_data.append(single_row)
        
        # Two-term power law parameters
        two_term_params = results['two_term_params']
        two_term_row = {
            'Compute_Metric': metric_name,
            'Model_Type': 'Two-Term Power Law',
            'C1': np.exp(two_term_params[0]),
            'alpha1': two_term_params[1],
            'C2': np.exp(two_term_params[2]),
            'alpha2': two_term_params[3],
            'E': np.exp(two_term_params[4]),
            'RMSE': results['two_term_rmse'],
            'MAE': results['two_term_mae'],
            'Fitting_Loss': results['two_term_loss']
        }
        summary_data.append(two_term_row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(results_dir, "dark_matter_summary_results.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary results saved to {summary_file}")
    
    # Also save improvement statistics
    improvement_data = []
    for metric_name, results in all_results.items():
        improvement_row = {
            'Compute_Metric': metric_name,
            'RMSE_Improvement_Pct': results['improvement_rmse_pct'],
            'Loss_Improvement_Pct': results['improvement_loss_pct'],
            'Single_RMSE': results['single_rmse'],
            'TwoTerm_RMSE': results['two_term_rmse'],
            'Single_Loss': results['single_loss'],
            'TwoTerm_Loss': results['two_term_loss']
        }
        improvement_data.append(improvement_row)
    
    improvement_df = pd.DataFrame(improvement_data)
    improvement_file = os.path.join(results_dir, "dark_matter_improvements.csv")
    improvement_df.to_csv(improvement_file, index=False)
    print(f"Improvement statistics saved to {improvement_file}")


def print_dark_matter_conclusions(all_results):
    """Print conclusions about the dark matter hypothesis."""
    print("\n" + "=" * 80)
    print("DARK MATTER HYPOTHESIS CONCLUSIONS")
    print("=" * 80)
    
    best_metric = None
    best_improvement = -np.inf
    
    for metric_name, results in all_results.items():
        improvement = results['improvement_rmse_pct']
        print(f"\n{metric_name}:")
        print(f"  RMSE improvement: {improvement:.1f}%")
        print(f"  Loss improvement: {results['improvement_loss_pct']:.1f}%")
        
        # Check if two-term model shows meaningful improvement
        if improvement > 5:  # 5% threshold
            print(f"  → Strong evidence for dark matter with {metric_name}")
        elif improvement > 1:
            print(f"  → Weak evidence for dark matter with {metric_name}")
        else:
            print(f"  → No evidence for dark matter with {metric_name}")
        
        if improvement > best_improvement:
            best_improvement = improvement
            best_metric = metric_name
    
    print(f"\nBest evidence for dark matter: {best_metric} ({best_improvement:.1f}% improvement)")
    
    # Check consistency of exponents across metrics
    alpha1_values = [all_results[m]['two_term_params'][1] for m in all_results.keys()]
    alpha2_values = [all_results[m]['two_term_params'][3] for m in all_results.keys()]
    
    print(f"\nConsistency check:")
    print(f"  α₁ range: {min(alpha1_values):.3f} to {max(alpha1_values):.3f}")
    print(f"  α₂ range: {min(alpha2_values):.3f} to {max(alpha2_values):.3f}")
    
    if max(alpha1_values) - min(alpha1_values) < 0.1 and max(alpha2_values) - min(alpha2_values) < 0.1:
        print("  → High consistency across compute metrics (strong evidence)")
    elif max(alpha1_values) - min(alpha1_values) < 0.3 and max(alpha2_values) - min(alpha2_values) < 0.3:
        print("  → Moderate consistency across compute metrics")
    else:
        print("  → Low consistency across compute metrics (weak evidence)")


if __name__ == "__main__":
    print("Testing Dark Matter Hypothesis for Neural Scaling Laws")
    print("=" * 60)
    print("This script tests whether model loss follows a two-term power law:")
    print("Loss = C₁/Compute^α₁ + C₂/Compute^α₂ + noise")
    print("=" * 60)
    
    # Run the analysis
    try:
        all_results = test_dark_matter_with_different_compute_metrics(
            save_results=True, 
            results_dir="results/dark_matter_analysis"
        )
        
        # Print conclusions
        print_dark_matter_conclusions(all_results)
        
    except Exception as e:
        print(f"Error running dark matter analysis: {e}")
        import traceback
        traceback.print_exc() 
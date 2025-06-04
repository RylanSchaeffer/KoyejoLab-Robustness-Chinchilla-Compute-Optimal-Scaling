import autograd.numpy as np
from autograd import grad
from autograd.scipy.stats import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import Tuple, Optional, Dict, List
import warnings

# Set style for better plots
try:
    plt.style.use('seaborn-v0_8')
except:
    try:
        plt.style.use('seaborn')
    except:
        plt.style.use('default')
        
sns.set_palette("husl")

class DarkMatterScalingAnalysis:
    """
    Class for fitting and analyzing neural scaling laws, including support for 
    two-term additive power laws to test the "dark matter" hypothesis.
    
    Dark matter hypothesis: Loss = C1/Compute^α1 + C2/Compute^α2 + noise
    """
    
    def __init__(self, normalize_compute: bool = True, normalization_factor: float = 1e18):
        """
        Initialize the scaling analysis.
        
        Args:
            normalize_compute: Whether to normalize compute values
            normalization_factor: Factor to divide compute by (default 1e18)
        """
        self.normalize_compute = normalize_compute
        self.normalization_factor = normalization_factor
        
    def _normalize_compute_values(self, compute: np.ndarray) -> np.ndarray:
        """Normalize compute values for numerical stability."""
        if self.normalize_compute:
            return compute / self.normalization_factor
        return compute
    
    def single_power_law(self, compute: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        Single-term power law: Loss = C/Compute^α + E
        
        Args:
            compute: Compute values (FLOPs, parameters, etc.)
            params: [log_C, alpha, log_E] where C and E are in log space
            
        Returns:
            Predicted loss values
        """
        log_C, alpha, log_E = params
        compute_norm = self._normalize_compute_values(compute)
        return np.exp(log_C) / (compute_norm ** alpha) + np.exp(log_E)
    
    def two_term_power_law(self, compute: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        Two-term additive power law: Loss = C1/Compute^α1 + C2/Compute^α2 + E
        
        Args:
            compute: Compute values
            params: [log_C1, alpha1, log_C2, alpha2, log_E] where C1, C2, E are in log space
            
        Returns:
            Predicted loss values
        """
        log_C1, alpha1, log_C2, alpha2, log_E = params
        compute_norm = self._normalize_compute_values(compute)
        
        term1 = np.exp(log_C1) / (compute_norm ** alpha1)
        term2 = np.exp(log_C2) / (compute_norm ** alpha2)
        constant = np.exp(log_E)
        
        return term1 + term2 + constant
    
    def huber_loss(self, y_true: np.ndarray, y_pred: np.ndarray, delta: float = 1e-3) -> float:
        """
        Robust Huber loss function for fitting.
        
        Args:
            y_true: True values
            y_pred: Predicted values  
            delta: Huber loss threshold
            
        Returns:
            Huber loss value
        """
        diff = y_true - y_pred
        cond = np.abs(diff) <= delta
        loss = np.where(cond, 0.5 * diff**2, delta * (np.abs(diff) - 0.5 * delta))
        return np.sum(loss)
    
    def objective_single(self, params: np.ndarray, compute: np.ndarray, losses: np.ndarray) -> float:
        """Objective function for single power law fitting."""
        predictions = self.single_power_law(compute, params)
        # Use log space for better numerical stability
        return self.huber_loss(np.log(losses), np.log(predictions))
    
    def objective_two_term(self, params: np.ndarray, compute: np.ndarray, losses: np.ndarray) -> float:
        """Objective function for two-term power law fitting."""
        predictions = self.two_term_power_law(compute, params)
        # Use log space for better numerical stability
        return self.huber_loss(np.log(losses), np.log(predictions))
    
    def fit_single_power_law(self, compute: np.ndarray, losses: np.ndarray, 
                           initial_params: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
        """
        Fit single-term power law to data.
        
        Args:
            compute: Compute values (FLOPs, parameters, etc.)
            losses: Loss values
            initial_params: Initial parameter guess [log_C, alpha, log_E]
            
        Returns:
            Tuple of (fitted_params, final_loss)
        """
        if initial_params is None:
            # Reasonable initial guesses
            initial_params = np.array([np.log(100), 0.3, np.log(0.1)])
        
        # Try multiple initializations to avoid local minima
        best_result = None
        best_loss = np.inf
        
        for i in range(5):
            # Add some random noise to initial params
            init_noise = initial_params + np.random.normal(0, 0.1, len(initial_params))
            
            try:
                result = minimize(
                    self.objective_single,
                    init_noise,
                    args=(compute, losses),
                    method='BFGS',
                    jac=grad(self.objective_single)
                )
                
                if result.fun < best_loss:
                    best_loss = result.fun
                    best_result = result
                    
            except Exception as e:
                warnings.warn(f"Optimization attempt {i+1} failed: {e}")
                continue
        
        if best_result is None:
            raise RuntimeError("All optimization attempts failed")
            
        return best_result.x, best_result.fun
    
    def fit_two_term_power_law(self, compute: np.ndarray, losses: np.ndarray,
                             initial_params: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
        """
        Fit two-term additive power law to data.
        
        Args:
            compute: Compute values
            losses: Loss values  
            initial_params: Initial parameter guess [log_C1, alpha1, log_C2, alpha2, log_E]
            
        Returns:
            Tuple of (fitted_params, final_loss)
        """
        if initial_params is None:
            # Reasonable initial guesses - make sure α1 ≠ α2
            initial_params = np.array([np.log(200), 0.4, np.log(50), 0.1, np.log(0.1)])
        
        # Try multiple initializations
        best_result = None
        best_loss = np.inf
        
        for i in range(10):  # More attempts for more complex model
            # Add random noise and ensure α1 ≠ α2
            init_noise = initial_params + np.random.normal(0, 0.1, len(initial_params))
            
            # Ensure different exponents
            if abs(init_noise[1] - init_noise[3]) < 0.05:
                init_noise[3] += 0.2 * (1 if np.random.random() > 0.5 else -1)
            
            try:
                result = minimize(
                    self.objective_two_term,
                    init_noise,
                    args=(compute, losses),
                    method='BFGS',
                    jac=grad(self.objective_two_term)
                )
                
                if result.fun < best_loss:
                    best_loss = result.fun
                    best_result = result
                    
            except Exception as e:
                warnings.warn(f"Optimization attempt {i+1} failed: {e}")
                continue
        
        if best_result is None:
            raise RuntimeError("All optimization attempts failed")
            
        return best_result.x, best_result.fun
    
    def compute_residuals(self, compute: np.ndarray, losses: np.ndarray, 
                         params: np.ndarray, model_type: str) -> np.ndarray:
        """
        Compute residuals for a fitted model.
        
        Args:
            compute: Compute values
            losses: True loss values
            params: Fitted parameters
            model_type: 'single' or 'two_term'
            
        Returns:
            Residuals (true - predicted)
        """
        if model_type == 'single':
            predictions = self.single_power_law(compute, params)
        elif model_type == 'two_term':
            predictions = self.two_term_power_law(compute, params)
        else:
            raise ValueError("model_type must be 'single' or 'two_term'")
            
        return losses - predictions
    
    def print_fitted_parameters(self, single_params: np.ndarray, two_term_params: np.ndarray):
        """Print fitted parameters in a readable format."""
        print("=" * 60)
        print("FITTED SCALING LAW PARAMETERS")
        print("=" * 60)
        
        print("\nSingle Power Law: Loss = C/Compute^α + E")
        print("-" * 40)
        C_single = np.exp(single_params[0])
        alpha_single = single_params[1]
        E_single = np.exp(single_params[2])
        
        if self.normalize_compute:
            print(f"C = {C_single:.6f} (×{self.normalization_factor:.0e} for unnormalized)")
        else:
            print(f"C = {C_single:.6f}")
        print(f"α = {alpha_single:.6f}")
        print(f"E = {E_single:.6f}")
        
        print("\nTwo-Term Power Law: Loss = C₁/Compute^α₁ + C₂/Compute^α₂ + E")
        print("-" * 60)
        C1 = np.exp(two_term_params[0])
        alpha1 = two_term_params[1]
        C2 = np.exp(two_term_params[2])
        alpha2 = two_term_params[3]
        E_two = np.exp(two_term_params[4])
        
        if self.normalize_compute:
            print(f"C₁ = {C1:.6f} (×{self.normalization_factor:.0e} for unnormalized)")
            print(f"C₂ = {C2:.6f} (×{self.normalization_factor:.0e} for unnormalized)")
        else:
            print(f"C₁ = {C1:.6f}")
            print(f"C₂ = {C2:.6f}")
        print(f"α₁ = {alpha1:.6f}")
        print(f"α₂ = {alpha2:.6f}")
        print(f"E = {E_two:.6f}")
        
        print("\n" + "=" * 60)
    
    def save_fitted_parameters(self, single_params: np.ndarray, two_term_params: np.ndarray, 
                             filename: str = "fitted_parameters.csv"):
        """Save fitted parameters to CSV file."""
        data = {
            'Model': ['Single Power Law', 'Two-Term Power Law'],
            'C1': [np.exp(single_params[0]), np.exp(two_term_params[0])],
            'alpha1': [single_params[1], two_term_params[1]],
            'C2': [np.nan, np.exp(two_term_params[2])],
            'alpha2': [np.nan, two_term_params[3]],
            'E': [np.exp(single_params[2]), np.exp(two_term_params[4])],
            'normalization_factor': [self.normalization_factor if self.normalize_compute else 1.0] * 2
        }
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Parameters saved to {filename}")
    
    def plot_fits_and_residuals(self, compute: np.ndarray, losses: np.ndarray,
                               single_params: np.ndarray, two_term_params: np.ndarray,
                               single_loss: float, two_term_loss: float,
                               save_plots: bool = True, plot_dir: str = "."):
        """
        Create comprehensive plots showing fits and residuals.
        
        Args:
            compute: Compute values
            losses: True loss values
            single_params: Single power law parameters
            two_term_params: Two-term power law parameters
            single_loss: Single model fitting loss
            two_term_loss: Two-term model fitting loss
            save_plots: Whether to save plots to disk
            plot_dir: Directory to save plots
        """
        fig = plt.figure(figsize=(16, 12))
        
        # Create a grid for subplots
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # Generate smooth curve for plotting fits
        compute_range = np.logspace(np.log10(compute.min()), np.log10(compute.max()), 1000)
        
        single_pred_smooth = self.single_power_law(compute_range, single_params)
        two_term_pred_smooth = self.two_term_power_law(compute_range, two_term_params)
        
        single_pred_data = self.single_power_law(compute, single_params)
        two_term_pred_data = self.two_term_power_law(compute, two_term_params)
        
        # Plot 1: Log-log scaling law fits
        ax1 = fig.add_subplot(gs[0, :])
        ax1.loglog(compute, losses, 'ko', alpha=0.6, markersize=6, label='Data')
        ax1.loglog(compute_range, single_pred_smooth, 'b-', linewidth=2, 
                  label=f'Single Power Law (Loss: {single_loss:.3f})')
        ax1.loglog(compute_range, two_term_pred_smooth, 'r-', linewidth=2, 
                  label=f'Two-Term Power Law (Loss: {two_term_loss:.3f})')
        
        ax1.set_xlabel('Compute' + (f' (÷{self.normalization_factor:.0e})' if self.normalize_compute else ''))
        ax1.set_ylabel('Loss')
        ax1.set_title('Neural Scaling Laws: Single vs Two-Term Power Law Fits')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Single model residuals
        ax2 = fig.add_subplot(gs[1, 0])
        single_residuals = self.compute_residuals(compute, losses, single_params, 'single')
        ax2.semilogx(compute, single_residuals, 'bo', alpha=0.6, markersize=5)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Compute' + (f' (÷{self.normalization_factor:.0e})' if self.normalize_compute else ''))
        ax2.set_ylabel('Residuals (True - Predicted)')
        ax2.set_title('Single Power Law Residuals')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Two-term model residuals
        ax3 = fig.add_subplot(gs[1, 1])
        two_term_residuals = self.compute_residuals(compute, losses, two_term_params, 'two_term')
        ax3.semilogx(compute, two_term_residuals, 'ro', alpha=0.6, markersize=5)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Compute' + (f' (÷{self.normalization_factor:.0e})' if self.normalize_compute else ''))
        ax3.set_ylabel('Residuals (True - Predicted)')
        ax3.set_title('Two-Term Power Law Residuals')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Residual statistics comparison
        ax4 = fig.add_subplot(gs[2, :])
        
        # Compute residual statistics
        single_rmse = np.sqrt(np.mean(single_residuals**2))
        two_term_rmse = np.sqrt(np.mean(two_term_residuals**2))
        single_mae = np.mean(np.abs(single_residuals))
        two_term_mae = np.mean(np.abs(two_term_residuals))
        
        metrics = ['RMSE', 'MAE', 'Fitting Loss']
        single_values = [single_rmse, single_mae, single_loss]
        two_term_values = [two_term_rmse, two_term_mae, two_term_loss]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax4.bar(x - width/2, single_values, width, label='Single Power Law', alpha=0.7)
        ax4.bar(x + width/2, two_term_values, width, label='Two-Term Power Law', alpha=0.7)
        
        ax4.set_xlabel('Metric')
        ax4.set_ylabel('Value')
        ax4.set_title('Model Comparison: Residual Statistics')
        ax4.set_xticks(x)
        ax4.set_xticklabels(metrics)
        ax4.legend()
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)
        
        # Add improvement percentage
        rmse_improvement = ((single_rmse - two_term_rmse) / single_rmse) * 100
        mae_improvement = ((single_mae - two_term_mae) / single_mae) * 100
        loss_improvement = ((single_loss - two_term_loss) / single_loss) * 100
        
        ax4.text(0.02, 0.98, f'RMSE Improvement: {rmse_improvement:.1f}%\n'
                             f'MAE Improvement: {mae_improvement:.1f}%\n'
                             f'Loss Improvement: {loss_improvement:.1f}%',
                transform=ax4.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f"{plot_dir}/dark_matter_scaling_analysis.png", dpi=300, bbox_inches='tight')
            print(f"Plot saved to {plot_dir}/dark_matter_scaling_analysis.png")
        
        plt.show()
    
    def analyze_dark_matter_hypothesis(self, compute: np.ndarray, losses: np.ndarray,
                                     save_results: bool = True, plot_dir: str = ".") -> Dict:
        """
        Complete analysis pipeline for testing the dark matter hypothesis.
        
        Args:
            compute: Compute values (FLOPs, parameters, etc.)
            losses: Loss values
            save_results: Whether to save results to disk
            plot_dir: Directory to save results
            
        Returns:
            Dictionary containing all fitted parameters and statistics
        """
        print("Starting Dark Matter Scaling Law Analysis...")
        print(f"Data: {len(compute)} points")
        print(f"Compute range: {compute.min():.2e} to {compute.max():.2e}")
        print(f"Loss range: {losses.min():.4f} to {losses.max():.4f}")
        
        if self.normalize_compute:
            print(f"Normalizing compute by {self.normalization_factor:.0e}")
        
        # Fit both models
        print("\nFitting single power law...")
        single_params, single_loss = self.fit_single_power_law(compute, losses)
        
        print("Fitting two-term power law...")
        two_term_params, two_term_loss = self.fit_two_term_power_law(compute, losses)
        
        # Print parameters
        self.print_fitted_parameters(single_params, two_term_params)
        
        # Create plots
        print("\nGenerating plots...")
        self.plot_fits_and_residuals(compute, losses, single_params, two_term_params,
                                    single_loss, two_term_loss, save_results, plot_dir)
        
        # Save parameters
        if save_results:
            self.save_fitted_parameters(single_params, two_term_params, 
                                      f"{plot_dir}/fitted_parameters.csv")
        
        # Compute model comparison statistics
        single_residuals = self.compute_residuals(compute, losses, single_params, 'single')
        two_term_residuals = self.compute_residuals(compute, losses, two_term_params, 'two_term')
        
        results = {
            'single_params': single_params,
            'two_term_params': two_term_params,
            'single_loss': single_loss,
            'two_term_loss': two_term_loss,
            'single_rmse': np.sqrt(np.mean(single_residuals**2)),
            'two_term_rmse': np.sqrt(np.mean(two_term_residuals**2)),
            'single_mae': np.mean(np.abs(single_residuals)),
            'two_term_mae': np.mean(np.abs(two_term_residuals)),
            'improvement_rmse_pct': ((np.sqrt(np.mean(single_residuals**2)) - 
                                    np.sqrt(np.mean(two_term_residuals**2))) / 
                                   np.sqrt(np.mean(single_residuals**2))) * 100,
            'improvement_loss_pct': ((single_loss - two_term_loss) / single_loss) * 100
        }
        
        print(f"\nAnalysis complete!")
        print(f"Two-term model improves RMSE by {results['improvement_rmse_pct']:.1f}%")
        print(f"Two-term model improves fitting loss by {results['improvement_loss_pct']:.1f}%")
        
        return results


def load_sample_data():
    """Load sample scaling law data for demonstration."""
    # This would load your actual data - modify as needed
    # For now, create synthetic data that follows a two-term power law
    np.random.seed(42)
    
    compute = np.logspace(18, 26, 50)  # FLOP range
    
    # True two-term power law with noise
    C1_true, alpha1_true = 100, 0.4
    C2_true, alpha2_true = 20, 0.1  
    E_true = 0.05
    
    true_losses = (C1_true / (compute / 1e18)**alpha1_true + 
                   C2_true / (compute / 1e18)**alpha2_true + E_true)
    
    # Add realistic noise
    noise = np.random.lognormal(0, 0.05, len(compute))
    losses = true_losses * noise
    
    return compute, losses


if __name__ == "__main__":
    # Example usage
    print("Dark Matter Scaling Law Analysis")
    print("=" * 50)
    
    # Load data (replace with your actual data loading)
    compute, losses = load_sample_data()
    
    # Create analyzer
    analyzer = DarkMatterScalingAnalysis(normalize_compute=True, normalization_factor=1e18)
    
    # Run analysis
    results = analyzer.analyze_dark_matter_hypothesis(compute, losses, save_results=True) 
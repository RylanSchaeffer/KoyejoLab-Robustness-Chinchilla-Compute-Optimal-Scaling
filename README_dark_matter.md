# Dark Matter Scaling Laws Analysis

This repository has been extended to test the "dark matter" hypothesis for neural scaling laws, which proposes that model loss follows a two-term additive power law:

```
Loss = C₁/Compute^α₁ + C₂/Compute^α₂ + noise
```

## New Files Added

### Core Implementation
- `src/dark_matter_scaling.py` - Main analysis class with fitting and plotting functionality
- `src/test_dark_matter_hypothesis.py` - Script to test the hypothesis on Chinchilla data
- `requirements.txt` - Required Python dependencies

## Features

### ✅ Two-Term Power Law Fitting
- Fits both single-term (`Loss = C/Compute^α + E`) and two-term models
- Uses robust Huber loss for optimization
- Multiple random initializations to avoid local minima
- Automatic differentiation with `autograd` for stable optimization

### ✅ Multiple Compute Metrics
Tests the hypothesis using different compute metrics:
- **Model Size (Parameters)** - Number of model parameters
- **Training FLOPs** - Total training compute in FLOPs 
- **Training Tokens** - Number of training tokens

### ✅ Comprehensive Visualization
Creates detailed plots showing:
- **Log-log scaling law fits** comparing single vs two-term models
- **Residual plots** for both models
- **Model comparison statistics** (RMSE, MAE, fitting loss)
- **Summary comparison** across all compute metrics

### ✅ Optional Compute Normalization
- Normalizes compute values (e.g., divide FLOPs by 1e18) for numerical stability
- Configurable normalization factors for different metrics

### ✅ Parameter Saving
- Prints fitted parameters (C₁, α₁, C₂, α₂, E) in readable format
- Saves parameters to CSV files
- Exports improvement statistics

## Usage

### Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Test dark matter hypothesis on Chinchilla data
PYTHONPATH=. python src/test_dark_matter_hypothesis.py
```

### Using the Analysis Class Directly
```python
from src.dark_matter_scaling import DarkMatterScalingAnalysis
import numpy as np

# Load your data
compute = your_compute_values  # FLOPs, parameters, etc.
losses = your_loss_values

# Create analyzer
analyzer = DarkMatterScalingAnalysis(
    normalize_compute=True, 
    normalization_factor=1e18
)

# Run complete analysis
results = analyzer.analyze_dark_matter_hypothesis(
    compute, losses, 
    save_results=True, 
    plot_dir="results"
)

# Access fitted parameters
single_params = results['single_params']
two_term_params = results['two_term_params']
improvement_pct = results['improvement_rmse_pct']
```

## Results on Chinchilla Data

The analysis on the Chinchilla dataset reveals:

| Compute Metric | RMSE Improvement | Best Evidence |
|----------------|------------------|---------------|
| Model Size | -0.6% | ❌ No evidence |
| Training FLOPs | -0.0% | ❌ No evidence |
| Training Tokens | **7.7%** | ✅ **Strong evidence** |

### Key Findings
- **Best evidence for dark matter**: Training Tokens (7.7% RMSE improvement)
- **Fitted parameters for Training Tokens**:
  - Two-term: `Loss = 0.090/Tokens^1.90 + 1.965/Tokens^0.15 + 1.28`
  - Single-term: `Loss = 1.659/Tokens^0.23 + 1.71`

### Interpretation
The results suggest that when using **training tokens** as the compute metric, there is evidence for a two-term power law structure in neural scaling. This could indicate:

1. **Multiple scaling regimes** - Different mechanisms dominating at different scales
2. **Hidden complexity** - Additional factors beyond simple power law scaling
3. **Data or token efficiency effects** - Separate contributions from data quantity vs. model capacity

However, the **low consistency across different compute metrics** suggests the evidence is not definitive and warrants further investigation.

## Output Files

When `save_results=True`, the analysis creates:
```
results/dark_matter_analysis/
├── Model_Size_Parameters/
│   ├── dark_matter_scaling_analysis.png
│   └── fitted_parameters.csv
├── Training_FLOPs/
│   ├── dark_matter_scaling_analysis.png
│   └── fitted_parameters.csv
├── Training_Tokens/
│   ├── dark_matter_scaling_analysis.png
│   └── fitted_parameters.csv
├── dark_matter_summary_comparison.png
├── dark_matter_summary_results.csv
└── dark_matter_improvements.csv
```

## Customization

### Testing on Your Own Data
```python
# Replace the load_chinchilla_data() function in test_dark_matter_hypothesis.py
def load_your_data():
    # Return pandas DataFrame with columns:
    # - Your compute metric (e.g., 'FLOPs', 'Parameters')  
    # - 'loss' column
    return your_dataframe

# Add your compute metrics to the compute_metrics dictionary
compute_metrics = {
    'Your Metric Name': {
        'column': 'your_column_name',
        'normalization': 1e18,  # Adjust as needed
        'description': 'Description of your metric'
    }
}
```

### Adjusting Fitting Parameters
```python
analyzer = DarkMatterScalingAnalysis(
    normalize_compute=True,           # Whether to normalize
    normalization_factor=1e18         # Normalization factor
)

# Custom initial parameters
results = analyzer.fit_two_term_power_law(
    compute, losses,
    initial_params=np.array([
        np.log(100),  # log(C₁)
        0.4,          # α₁
        np.log(50),   # log(C₂)  
        0.1,          # α₂
        np.log(0.1)   # log(E)
    ])
)
```

## Dependencies
- `numpy` >= 1.19.0
- `scipy` >= 1.7.0  
- `matplotlib` >= 3.3.0
- `pandas` >= 1.3.0
- `seaborn` >= 0.11.0
- `autograd` >= 1.3.0

## Technical Notes

### Model Formulation
- **Single Power Law**: `Loss = C/Compute^α + E`
- **Two-Term Power Law**: `Loss = C₁/Compute^α₁ + C₂/Compute^α₂ + E`
- Coefficients (C, C₁, C₂, E) are fitted in log-space for numerical stability
- Uses Huber loss for robust fitting against outliers

### Optimization Strategy
- Multiple random initializations (5 for single-term, 10 for two-term)
- BFGS optimization with automatic differentiation
- Ensures α₁ ≠ α₂ for identifiability in two-term model
- Graceful fallback if optimization fails

### Statistical Interpretation
- RMSE improvement > 5% = Strong evidence
- RMSE improvement 1-5% = Weak evidence  
- RMSE improvement < 1% = No evidence
- Consistency across metrics strengthens conclusions 
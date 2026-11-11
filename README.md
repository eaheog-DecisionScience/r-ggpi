# Robust Graph-Grouped Permutation Importance (R-GGPI)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%+-%233776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn Compatible](https://img.shields.io/badge/scikit--learn-compatible-%23F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

**R-GGPI** is an adaptive, collinearity-resilient feature-importance and selection package for tabular models. It groups features into cohesive branches using stability-regularized correlation graphs and estimates the importance of each group by permuting its rows jointly.

Jointly permuting a group's columns preserves their observed joint distribution within that group while breaking their association with the target. Groups are constructed adaptively using Minimum Spanning Trees (MST) with statistically sound correlation thresholding, maximum group size constraints, cross-resample stability analysis, and out-of-distribution (OOD) distributional shift minimization. Target-aware branch anchoring helps allocate group importance to predictive features, and a complexity gate avoids unnecessary multiprocessing overhead.

---

## Key Features

- **Adaptive Grouping & Sizing:** Dynamically determines minimum correlation significance thresholds and caps group sizes (`max_group_size='auto'`), eliminating the MST "chaining effect" that erroneously lumps disparate features or noise into giant monolithic clusters.
- **Distributional Shift Diagnostics:** Measures cross-group correlation energy (`group_distribution_shifts`) to monitor and minimize out-of-distribution (OOD) extrapolation risk when groups are permuted.
- **Stability Analysis:** Uses bootstrap subsampling (`stability_resamples=5`) to penalize fluctuating correlations and report Signal-to-Noise Ratio (SNR) scores (`group_stability_scores`) for every feature group.
- **Robust Multi-Criteria Selection:** `RGGPISelector` supports ranking and selecting groups via `selection_criterion='importance'`, `'stability'`, or `'robust'` (mean minus standard deviation).
- **Non-Linear & Outlier Resilience:** Supports rank-based Spearman correlation (`correlation_method='spearman'`, default) and Pearson correlation.
- **Group-level and per-feature scores:** `group_importances_mean`, `group_importances_std`, and compatibility-weighted `importances_mean` / `importances_std`.
- **Faster permutation scoring:** Permutation importance is scored on at most 2,000 randomly selected rows by default, while grouping and correlation analysis still use all rows. Set `eval_subsample_size=None` to score all rows or provide a different positive integer cap.
- **Collinearity Resilience:** Groups redundant features into MST branches and permutes them jointly.
- **Smart Complexity Gating:** Uses workload size to avoid multiprocessing overhead on smaller tasks and parallelize larger ones.
- **Scikit-Learn API Integration:** Fully compatible with Scikit-Learn pipelines via `BaseEstimator` and `TransformerMixin`.

Pass `random_state` to `RGGPIExplainer`, `RGGPISelector`, or `r_ggpi_importance` to make permutation repeats and stability subsampling reproducible.

`RGGPISelector` selects complete groups with positive group importance by default and raises a `ValueError` if none are positive. When `max_features` is set, it is a strict upper bound: groups are considered in descending importance order and groups that do not fit are skipped, never split. If no positively scoring group fits the limit, `fit` raises a `ValueError`; increase the limit or use `max_features=None`.

---

## Installation

After the package has been published, install it from PyPI:

```bash
pip install r-ggpi
```

Or install the repository in editable mode, including the test dependencies:

```bash
git clone https://github.com/deepnovus/r-ggpi.git
cd r-ggpi
pip install -e ".[test]"
```

Run the test suite with:

```bash
pytest
```

## Example

The [basic usage example](examples/basic_usage.py) generates synthetic data,
fits a model, reports R-GGPI feature importance, and demonstrates selecting
features before fitting a second model. Run it from the project root after
installing the package:

```bash
python examples/basic_usage.py
```

Build the source distribution and wheel with:

```bash
python -m pip install build
python -m build
```

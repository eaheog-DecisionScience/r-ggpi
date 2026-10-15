# Robust Graph-Grouped Permutation Importance (R-GGPI)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%+-%233776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn Compatible](https://img.shields.io/badge/scikit--learn-compatible-%23F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

**R-GGPI** is a fast, collinearity-resilient feature attribution and selection package designed for complex tabular machine learning models.

Traditional Permutation Feature Importance (PFI) breaks down under multi-collinearity because independent shuffling destroys joint feature distributions, while game-theoretic alternatives like TreeSHAP suffer from severe combinatorial computational explosion on large datasets. R-GGPI bridges this gap by combining notions/ideas like- **Minimum Spanning Tree (MST) graph partitioning**, **Target-Aware Branch Anchoring**, and an **Adaptive Complexity Cost Gate** for intelligent multi-core parallelization.

---

## Key Features

- **Collinearity Resilience:** Groups redundant features into disconnected MST coalition branches and permutes them jointly, preserving real-world data manifolds.
- **Target-Aware Branch Anchoring:** Prevents low-importance "hitchhiker" variables from draining attribution credit away from primary predictive features within a shared cluster.
- **Extreme Scalability:** Achieves up to an **84x speedup over TreeSHAP** and **5.6x over standard PFI** on large datasets ($N = 20,000$).
- **Smart Complexity Gating:** Automatically detects runtime workload dimensionality ($\mathcal{K} = N \cdot M \cdot R$) to bypass Python multiprocessing overhead on light tasks and scale seamlessly on heavy enterprise tables.
- **Scikit-Learn API Integration:** Fully compatible with Scikit-Learn pipelines via `BaseEstimator` and `TransformerMixin`.

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

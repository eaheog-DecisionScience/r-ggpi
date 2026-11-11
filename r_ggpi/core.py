import numpy as np
import pandas as pd
import networkx as nx
from joblib import Parallel, delayed
from sklearn.utils import check_random_state
from .utils import (
    validate_inputs,
    compute_target_correlations,
    compute_stable_correlation_matrix,
    compute_distribution_shifts,
    build_feature_groups,
)


class ResultContainer:
  """Container for R-GGPI feature importance results and diagnostics."""

  def __init__(
      self,
      importances_mean,
      importances_std,
      group_importances_mean,
      group_importances_std,
      feature_groups,
      n_branches,
      group_stability_scores=None,
      group_distribution_shifts=None,
      grouping_metadata=None,
  ):
    self.importances_mean = importances_mean
    self.importances_std = importances_std
    self.group_importances_mean = group_importances_mean
    self.group_importances_std = group_importances_std
    self.feature_groups = feature_groups
    self.n_branches = n_branches
    self.group_stability_scores = (
        group_stability_scores
        if group_stability_scores is not None
        else pd.Series(dtype=float)
    )
    self.group_distribution_shifts = (
        group_distribution_shifts
        if group_distribution_shifts is not None
        else pd.Series(dtype=float)
    )
    self.grouping_metadata = grouping_metadata or {}


class RGGPIExplainer:
  """Robust Graph-Grouped Permutation Importance (R-GGPI) Explainer.

  Parameters:
  -----------
  model : estimator object
      A fitted scikit-learn compatible estimator (e.g., RandomForestClassifier).
  pruning_percentile : float (default=75.0)
      Percentile threshold for MST edge pruning when grouping_method='percentile'.
  n_repeats : int (default=5)
      Number of permutation repeats to stabilize importance scores.
  grouping_method : str (default='adaptive')
      Grouping strategy: 'adaptive' (recommended) or 'percentile' (legacy).
      'adaptive' applies correlation significance thresholding, stability
      subsampling, distributional shift mitigation, and maximum group size constraints.
  min_correlation : float or 'auto' (default='auto')
      Minimum correlation required to connect features into a group. When 'auto',
      adaptively determined from sample size and independence null threshold.
  max_group_size : int, 'auto', or None (default='auto')
      Maximum allowable number of features in any single group. When 'auto',
      adaptively set to max(5, ceil(sqrt(n_active_features))) to prevent giant
      chained components that disrupt importance granularity and selection.
  correlation_method : str (default='spearman')
      Correlation metric: 'spearman' (rank correlation, robust to outliers and
      monotonic non-linearities) or 'pearson'.
  stability_resamples : int (default=5)
      Number of data subsamples used to evaluate correlation stability across
      data perturbations and prune fragile edges.
  stability_penalty : float (default=1.0)
      Penalty factor for correlation standard deviation across subsamples
      (stable_corr = max(0, mean - penalty * std)).
  n_jobs : int (default=1)
      Number of CPU cores to use if complexity threshold is met.
  parallel_threshold : int (default=1_000_000)
      Complexity proxy threshold (n_samples * n_features * n_repeats)
      to trigger multiprocessing and avoid small-scale spawning overhead.
  random_state : int, RandomState instance, or None (default=None)
      Random state for reproducible permutation repeats and stability subsampling.
  """

  def __init__(
      self,
      model,
      pruning_percentile=75.0,
      n_repeats=5,
      grouping_method="adaptive",
      min_correlation="auto",
      max_group_size="auto",
      correlation_method="spearman",
      stability_resamples=5,
      stability_penalty=1.0,
      n_jobs=1,
      parallel_threshold=1_000_000,
      random_state=None,
  ):
    self.model = model
    self.pruning_percentile = pruning_percentile
    self.n_repeats = n_repeats
    self.grouping_method = grouping_method
    self.min_correlation = min_correlation
    self.max_group_size = max_group_size
    self.correlation_method = correlation_method
    self.stability_resamples = stability_resamples
    self.stability_penalty = stability_penalty
    self.n_jobs = n_jobs
    self.parallel_threshold = parallel_threshold
    self.random_state = random_state

  def _compute_single_repeat(
      self,
      X,
      y,
      baseline_score,
      communities,
      active_features,
      node_variances,
      target_corrs,
      seed,
  ):
    importances = {feat: 0.0 for feat in X.columns}
    group_importances = {}
    rng = np.random.default_rng(seed)

    for group_index, community in enumerate(communities):
      nodes = sorted(
          list(community),
          key=lambda n: target_corrs.get(active_features[n], 0.0),
          reverse=True,
      )
      branch_bias = np.linspace(1.0, 0.2, len(nodes))
      node_vars = np.array([node_variances.get(n, 1.0) for n in nodes])
      weights = branch_bias * node_vars
      total_weight = np.sum(weights)
      if total_weight > 0:
        weights = weights / total_weight

      group_name = f"group_{group_index}"
      group_features = [active_features[node] for node in nodes]
      row_permutation = rng.permutation(len(X))
      X_permuted = X.copy()
      X_permuted.loc[:, group_features] = X.iloc[
          row_permutation
      ].loc[:, group_features].to_numpy()

      perm_score = self.model.score(X_permuted, y)
      group_drop = baseline_score - perm_score
      group_importances[group_name] = group_drop

      for idx, node in enumerate(nodes):
        feat_name = active_features[node]
        importances[feat_name] = group_drop * weights[idx]

    return importances, group_importances

  def explain(self, X, y):
    X, y = validate_inputs(X, y)
    feature_cols = X.columns.tolist()
    n_samples, n_features = X.shape

    baseline_score = self.model.score(X, y)
    target_corrs = compute_target_correlations(X, y, feature_cols)

    # Variance Screening
    variances = X.var()
    active_features = variances[variances > 1e-4].index.tolist()
    X_filtered = X[active_features]
    n_active = len(active_features)

    if n_active == 0:
      zero_importances = pd.Series(0.0, index=feature_cols)
      return ResultContainer(
          zero_importances,
          zero_importances,
          pd.Series(dtype=float),
          pd.Series(dtype=float),
          {},
          0,
          pd.Series(dtype=float),
          pd.Series(dtype=float),
          {},
      )

    node_variances = {
        i: float(X_filtered.iloc[:, i].var()) for i in range(n_active)
    }

    # Correlation & Stability Analysis
    stable_corr, mean_corr, std_corr = compute_stable_correlation_matrix(
        X_filtered,
        method=self.correlation_method,
        n_resamples=self.stability_resamples,
        penalty=self.stability_penalty,
        random_state=self.random_state,
    )

    # Adaptive Graph Construction & MST Grouping
    communities, feature_groups, metadata = build_feature_groups(
        active_features,
        stable_corr,
        grouping_method=self.grouping_method,
        min_correlation=self.min_correlation,
        max_group_size=self.max_group_size,
        pruning_percentile=self.pruning_percentile,
        n_samples=n_samples,
    )

    # Distributional Shift (OOD Risk) Analysis
    group_shifts = compute_distribution_shifts(
        feature_groups, stable_corr, active_features
    )
    metadata["mean_distribution_shift"] = (
        float(group_shifts.mean()) if not group_shifts.empty else 0.0
    )

    # Smart Gating Proxy
    complexity_score = n_samples * n_features * self.n_repeats
    use_parallel = (
        self.n_jobs != 1 and complexity_score > self.parallel_threshold
    )
    random_state = check_random_state(self.random_state)
    seeds = random_state.randint(
        np.iinfo(np.int32).max, size=self.n_repeats
    )

    if use_parallel:
      parallel_results = Parallel(n_jobs=self.n_jobs)(
          delayed(self._compute_single_repeat)(
              X,
              y,
              baseline_score,
              communities,
              active_features,
              node_variances,
              target_corrs,
              seed,
          )
          for seed in seeds
      )
      importance_accumulator = {feat: [] for feat in feature_cols}
      group_importance_accumulator = {
          group_name: [] for group_name in feature_groups
      }
      for feature_result, group_result in parallel_results:
        for feat, val in feature_result.items():
          importance_accumulator[feat].append(val)
        for group_name, val in group_result.items():
          group_importance_accumulator[group_name].append(val)
    else:
      importance_accumulator = {feat: [] for feat in feature_cols}
      group_importance_accumulator = {
          group_name: [] for group_name in feature_groups
      }
      for seed in seeds:
        feature_result, group_result = self._compute_single_repeat(
            X,
            y,
            baseline_score,
            communities,
            active_features,
            node_variances,
            target_corrs,
            seed,
        )
        for feat, val in feature_result.items():
          importance_accumulator[feat].append(val)
        for group_name, val in group_result.items():
          group_importance_accumulator[group_name].append(val)

    mean_importances = pd.Series(
        {feat: float(np.mean(vals)) for feat, vals in importance_accumulator.items()}
    )
    std_importances = pd.Series(
        {feat: float(np.std(vals)) for feat, vals in importance_accumulator.items()}
    )
    mean_group_importances = pd.Series(
        {
            group_name: float(np.mean(vals))
            for group_name, vals in group_importance_accumulator.items()
        },
        dtype=float,
    )
    std_group_importances = pd.Series(
        {
            group_name: float(np.std(vals))
            for group_name, vals in group_importance_accumulator.items()
        },
        dtype=float,
    )

    # Stability Analysis: Signal-to-Noise Ratio (SNR)
    group_stability_scores = (
        mean_group_importances.abs() / (std_group_importances + 1e-6)
    )

    return ResultContainer(
        mean_importances,
        std_importances,
        mean_group_importances,
        std_group_importances,
        feature_groups,
        len(communities),
        group_stability_scores=group_stability_scores,
        group_distribution_shifts=group_shifts,
        grouping_metadata=metadata,
    )


def r_ggpi_importance(
    model,
    X,
    y,
    pruning_percentile=75.0,
    n_repeats=5,
    grouping_method="adaptive",
    min_correlation="auto",
    max_group_size="auto",
    correlation_method="spearman",
    stability_resamples=5,
    stability_penalty=1.0,
    n_jobs=1,
    random_state=None,
):
  """Calculates R-GGPI feature and group importances."""
  explainer = RGGPIExplainer(
      model=model,
      pruning_percentile=pruning_percentile,
      n_repeats=n_repeats,
      grouping_method=grouping_method,
      min_correlation=min_correlation,
      max_group_size=max_group_size,
      correlation_method=correlation_method,
      stability_resamples=stability_resamples,
      stability_penalty=stability_penalty,
      n_jobs=n_jobs,
      random_state=random_state,
  )
  return explainer.explain(X, y)

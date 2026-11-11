from numbers import Integral

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from .core import RGGPIExplainer
from .utils import validate_inputs


class RGGPISelector(BaseEstimator, TransformerMixin):
  """Scikit-Learn Compatible Feature Selector using Adaptive R-GGPI.

  Parameters:
  -----------
  model : estimator object
      A fitted scikit-learn compatible estimator (e.g., RandomForestClassifier).
  pruning_percentile : float (default=75.0)
      Percentile threshold for MST edge pruning when grouping_method='percentile'.
  n_repeats : int (default=5)
      Number of permutation repeats to stabilize importance scores.
  max_features : int, 'auto', or None (default=None)
      Upper bound on the number of features to select:
      - 'auto': Dynamically determines the minimal feature subset using an
        elbow cutoff (min_relative_importance), cumulative Pareto threshold, and SNR gating.
      - int: Strict upper bound on the maximum number of features.
      - None: Selects all complete groups with positive importance.
  cumulative_importance : float (default=0.90)
      Fraction of total positive group importance required when max_features='auto'
      (e.g., 0.90 selects groups explaining >= 90% of predictive power).
  min_relative_importance : float (default=0.10)
      Minimum importance fraction relative to the top group required when
      max_features='auto'. Groups contributing < 10% of the top group's drop are
      treated as noise-tail diminishing returns.
  min_group_snr : float (default=1.0)
      Minimum Signal-to-Noise Ratio (mean / std) required for groups under 'auto'
      to filter out spurious or noisy fluctuations.
  grouping_method : str (default='adaptive')
      Grouping strategy: 'adaptive' (recommended) or 'percentile' (legacy).
  min_correlation : float or 'auto' (default='auto')
      Minimum correlation required to group features. When 'auto', adaptively
      estimated based on sample size and null distribution noise floor.
  max_group_size : int, 'auto', or None (default='auto')
      Upper bound on the size of any single feature group. When 'auto',
      co-adapted with max_features to prevent group skipping.
  correlation_method : str (default='spearman')
      Correlation metric: 'spearman' or 'pearson'.
  stability_resamples : int (default=5)
      Number of data subsamples used to evaluate correlation stability.
  stability_penalty : float (default=1.0)
      Penalty factor for correlation standard deviation across subsamples.
  selection_criterion : str (default='importance')
      Group ranking criterion: 'importance' (mean score drop), 'stability'
      (signal-to-noise ratio), or 'robust' (mean - 1.0 * std).
  n_jobs : int (default=1)
      Number of CPU cores to use if complexity threshold is met.
  random_state : int, RandomState instance, or None (default=None)
      Seed for reproducible permutation repeats and stability subsampling.
  """

  def __init__(
      self,
      model,
      pruning_percentile=75.0,
      n_repeats=5,
      max_features=None,
      cumulative_importance=0.90,
      min_relative_importance=0.10,
      min_group_snr=1.0,
      grouping_method="adaptive",
      min_correlation="auto",
      max_group_size="auto",
      correlation_method="spearman",
      stability_resamples=5,
      stability_penalty=1.0,
      selection_criterion="importance",
      n_jobs=1,
      random_state=None,
  ):
    self.model = model
    self.pruning_percentile = pruning_percentile
    self.n_repeats = n_repeats
    self.max_features = max_features
    self.cumulative_importance = cumulative_importance
    self.min_relative_importance = min_relative_importance
    self.min_group_snr = min_group_snr
    self.grouping_method = grouping_method
    self.min_correlation = min_correlation
    self.max_group_size = max_group_size
    self.correlation_method = correlation_method
    self.stability_resamples = stability_resamples
    self.stability_penalty = stability_penalty
    self.selection_criterion = selection_criterion
    self.n_jobs = n_jobs
    self.random_state = random_state

  def fit(self, X, y):
    X_df, y_arr = validate_inputs(X, y)

    if (
        self.max_features is not None
        and self.max_features != "auto"
        and (not isinstance(self.max_features, Integral) or self.max_features < 1)
    ):
      raise ValueError("max_features must be 'auto', a positive integer, or None.")

    # Determine effective max group size so that groups don't exceed max_features
    # if max_features is an integer > 1
    if self.max_group_size == "auto":
      if (
          self.max_features is not None
          and self.max_features != "auto"
          and self.max_features > 1
      ):
        eff_max_group_size = min(
            self.max_features,
            max(5, int(np.ceil(np.sqrt(X_df.shape[1])))),
        )
      else:
        eff_max_group_size = max(5, int(np.ceil(np.sqrt(X_df.shape[1]))))
    else:
      eff_max_group_size = self.max_group_size

    explainer = RGGPIExplainer(
        model=self.model,
        pruning_percentile=self.pruning_percentile,
        n_repeats=self.n_repeats,
        grouping_method=self.grouping_method,
        min_correlation=self.min_correlation,
        max_group_size=eff_max_group_size,
        correlation_method=self.correlation_method,
        stability_resamples=self.stability_resamples,
        stability_penalty=self.stability_penalty,
        n_jobs=self.n_jobs,
        random_state=self.random_state,
    )
    result = explainer.explain(X_df, y_arr)

    self.feature_importances_ = result.importances_mean.sort_values(
        ascending=False
    )
    self.feature_importances_std_ = result.importances_std
    self.group_importances_mean_ = result.group_importances_mean
    self.group_importances_std_ = result.group_importances_std
    self.group_stability_scores_ = result.group_stability_scores
    self.group_distribution_shifts_ = result.group_distribution_shifts
    self.grouping_metadata_ = result.grouping_metadata
    self.feature_groups_ = result.feature_groups

    if self.selection_criterion == "stability":
      ranked_groups = result.group_stability_scores.sort_values(ascending=False)
    elif self.selection_criterion == "robust":
      robust_scores = (
          result.group_importances_mean - 1.0 * result.group_importances_std
      )
      ranked_groups = robust_scores.sort_values(ascending=False)
    else:
      ranked_groups = result.group_importances_mean.sort_values(ascending=False)

    selected_group_names = []
    if self.max_features == "auto":
      # Data-driven Pareto + Elbow selection
      pos_groups = [
          g
          for g in ranked_groups.index
          if self.group_importances_mean_.get(g, 0.0) > 0
          and self.group_stability_scores_.get(g, 0.0) >= self.min_group_snr
      ]
      if pos_groups:
        max_imp = max(self.group_importances_mean_[g] for g in pos_groups)
        # Filter groups that meet relative importance threshold (default >= 10% of max)
        strong_groups = [
            g
            for g in pos_groups
            if self.group_importances_mean_[g]
            >= self.min_relative_importance * max_imp
        ]
        if not strong_groups:
          strong_groups = pos_groups[:1]

        total_pos_imp = sum(self.group_importances_mean_[g] for g in strong_groups)
        cum_share = 0.0
        for group_name in strong_groups:
          selected_group_names.append(group_name)
          cum_share += self.group_importances_mean_[group_name] / total_pos_imp
          if cum_share >= self.cumulative_importance:
            break
      else:
        # Fallback to top positive group if SNR filter was too aggressive
        fallback_pos = [
            g
            for g in ranked_groups.index
            if self.group_importances_mean_.get(g, 0.0) > 0
        ]
        if fallback_pos:
          selected_group_names = fallback_pos[:1]
    elif self.max_features is not None:
      selected_count = 0
      for group_name in ranked_groups.index:
        importance = self.group_importances_mean_.get(group_name, 0.0)
        group_features = self.feature_groups_[group_name]
        if (
            importance <= 0
            or selected_count + len(group_features) > self.max_features
        ):
          continue
        selected_group_names.append(group_name)
        selected_count += len(group_features)
    else:
      selected_group_names = [
          group_name
          for group_name in ranked_groups.index
          if self.group_importances_mean_.get(group_name, 0.0) > 0
      ]

    if not selected_group_names:
      if self.max_features is None or self.max_features == "auto":
        raise ValueError(
            "No feature groups have positive importance on the fitting data."
        )
      raise ValueError(
          "No positively scoring feature group fits max_features; increase "
          "the limit or use max_features=None."
      )

    self.selected_groups_ = selected_group_names
    selected_features = {
        feature
        for group_name in selected_group_names
        for feature in self.feature_groups_[group_name]
    }
    self.selected_features_ = [
        feature
        for feature in self.feature_importances_.index
        if feature in selected_features
    ]

    self.n_features_in_ = X_df.shape[1]
    return self

  def transform(self, X):
    check_is_fitted(self, attributes=["feature_importances_", "selected_features_"])
    X_df, _ = validate_inputs(X, np.zeros(len(X)))
    return X_df[self.selected_features_]

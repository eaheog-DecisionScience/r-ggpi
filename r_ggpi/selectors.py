import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from .core import RGGPIExplainer
from .utils import validate_inputs


class RGGPISelector(BaseEstimator, TransformerMixin):
  """Scikit-Learn Compatible Feature Selector using R-GGPI."""

  def __init__(
      self,
      model,
      pruning_percentile=75.0,
      n_repeats=5,
      max_features=None,
      n_jobs=1,
  ):
    self.model = model
    self.pruning_percentile = pruning_percentile
    self.n_repeats = n_repeats
    self.max_features = max_features
    self.n_jobs = n_jobs

  def fit(self, X, y):
    X_df, y_arr = validate_inputs(X, y)
    explainer = RGGPIExplainer(
        model=self.model,
        pruning_percentile=self.pruning_percentile,
        n_repeats=self.n_repeats,
        n_jobs=self.n_jobs,
    )
    result = explainer.explain(X_df, y_arr)

    self.feature_importances_ = result.importances_mean.sort_values(
        ascending=False
    )
    self.feature_importances_std_ = result.importances_std

    if self.max_features is not None:
      self.selected_features_ = self.feature_importances_.head(
          self.max_features
      ).index.tolist()
    else:
      self.selected_features_ = self.feature_importances_[
          self.feature_importances_ > 0
      ].index.tolist()

    self.n_features_in_ = X_df.shape[1]
    return self

  def transform(self, X):
    check_is_fitted(self, attributes=["feature_importances_", "selected_features_"])
    X_df, _ = validate_inputs(X, np.zeros(len(X)))
    return X_df[self.selected_features_]
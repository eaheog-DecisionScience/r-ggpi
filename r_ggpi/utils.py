import numpy as np
import pandas as pd


def validate_inputs(X, y):
  """Ensures X is a pandas DataFrame and y is a numpy array."""
  if not isinstance(X, pd.DataFrame):
    X = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
  if isinstance(y, pd.Series):
    y = y.values
  elif isinstance(y, list):
    y = np.array(y)
  return X, y


def compute_target_correlations(X, y, feature_cols):
  """Pre-computes absolute univariate target alignments for anchoring."""
  target_corrs = {}
  for col in feature_cols:
    valid_mask = ~np.isnan(X[col]) & ~np.isnan(y)
    if valid_mask.sum() > 1:
      corr = np.corrcoef(X.loc[valid_mask, col], y[valid_mask])[0, 1]
      target_corrs[col] = abs(corr) if not np.isnan(corr) else 0.0
    else:
      target_corrs[col] = 0.0
  return target_corrs
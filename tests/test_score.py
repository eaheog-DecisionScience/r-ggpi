import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from r_ggpi import RGGPIExplainer, r_ggpi_importance


def test_explainer_returns_importance_for_each_feature():
  X = pd.DataFrame(
      {
          "signal": [0, 1, 0, 1, 0, 1],
          "noise": [1, 0, 1, 1, 0, 0],
      }
  )
  y = np.array([0, 1, 0, 1, 0, 1])
  model = DummyClassifier(strategy="most_frequent").fit(X, y)

  result = RGGPIExplainer(model, n_repeats=2).explain(X, y)

  assert set(result.importances_mean.index) == set(X.columns)
  assert set(result.importances_std.index) == set(X.columns)
  assert np.isfinite(result.importances_mean.to_numpy()).all()
  assert np.isfinite(result.importances_std.to_numpy()).all()
  assert result.n_branches >= 1


def test_function_api_returns_result_container():
  X = pd.DataFrame({"signal": [0, 1, 0, 1], "noise": [1, 0, 1, 0]})
  y = np.array([0, 1, 0, 1])
  model = DummyClassifier(strategy="most_frequent").fit(X, y)

  result = r_ggpi_importance(model, X, y, n_repeats=1)

  assert list(result.importances_mean.index) == list(X.columns)
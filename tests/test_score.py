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
  assert np.isfinite(result.group_importances_mean.to_numpy()).all()
  assert np.isfinite(result.group_importances_std.to_numpy()).all()
  assert result.n_branches >= 1


def test_function_api_returns_result_container():
  X = pd.DataFrame({"signal": [0, 1, 0, 1], "noise": [1, 0, 1, 0]})
  y = np.array([0, 1, 0, 1])
  model = DummyClassifier(strategy="most_frequent").fit(X, y)

  result = r_ggpi_importance(model, X, y, n_repeats=1)

  assert list(result.importances_mean.index) == list(X.columns)


def test_group_permutation_preserves_within_group_dependence_and_is_repeatable():
  rng = np.random.default_rng(42)
  signal = rng.normal(size=200)
  X = pd.DataFrame({"signal": signal, "duplicate": signal.copy()})
  y = signal.copy()

  class CorrelationScorer:
    def __init__(self):
      self.feature_correlations = []

    def score(self, features, target):
      self.feature_correlations.append(features.iloc[:, 0].corr(features.iloc[:, 1]))
      return np.corrcoef(features.iloc[:, 0], target)[0, 1]

  model = CorrelationScorer()
  explainer = RGGPIExplainer(
      model, pruning_percentile=100, n_repeats=4, random_state=17
  )

  first = explainer.explain(X, y)
  first_correlations = model.feature_correlations.copy()
  model.feature_correlations.clear()
  second = explainer.explain(X, y)

  assert first.feature_groups == {"group_0": ("signal", "duplicate")}
  assert np.allclose(first_correlations, 1.0)
  assert np.allclose(
      first.group_importances_mean.to_numpy(),
      second.group_importances_mean.to_numpy(),
  )
  assert np.allclose(
      first.group_importances_std.to_numpy(),
      second.group_importances_std.to_numpy(),
  )
  assert np.isclose(
      first.group_importances_mean["group_0"],
      first.importances_mean.sum(),
  )


def test_variance_screening_keeps_full_model_input_schema():
  X = pd.DataFrame(
      {
          "signal": [0, 1, 0, 1, 0, 1],
          "constant": [1, 1, 1, 1, 1, 1],
      }
  )
  y = np.array([0, 1, 0, 1, 0, 1])
  model = DummyClassifier(strategy="most_frequent").fit(X, y)

  result = RGGPIExplainer(model, n_repeats=2, random_state=5).explain(X, y)

  assert set(result.importances_mean.index) == set(X.columns)
  assert result.importances_mean["constant"] == 0.0


def test_permutation_scoring_uses_configured_sample_size():
  rng = np.random.default_rng(42)
  signal = rng.normal(size=250)
  X = pd.DataFrame({"signal": signal, "duplicate": signal.copy()})

  class RecordingScorer:
    def __init__(self):
      self.scored_rows = []

    def score(self, features, target):
      self.scored_rows.append(len(features))
      return np.corrcoef(features.iloc[:, 0], target)[0, 1]

  model = RecordingScorer()
  RGGPIExplainer(
      model,
      n_repeats=2,
      eval_subsample_size=37,
      random_state=42,
  ).explain(X, signal)

  assert model.scored_rows.count(37) == 3
  assert all(row_count in (2, 37) for row_count in model.scored_rows)


def test_default_permutation_scoring_caps_at_2000_rows():
  rng = np.random.default_rng(42)
  signal = rng.normal(size=2100)
  X = pd.DataFrame({"signal": signal, "duplicate": signal.copy()})

  class RecordingScorer:
    def __init__(self):
      self.scored_rows = []

    def score(self, features, target):
      self.scored_rows.append(len(features))
      return np.corrcoef(features.iloc[:, 0], target)[0, 1]

  model = RecordingScorer()
  RGGPIExplainer(
      model,
      n_repeats=1,
      stability_resamples=1,
      random_state=42,
  ).explain(X, signal)

  assert model.scored_rows.count(2000) == 2
  assert model.scored_rows[0] == 2
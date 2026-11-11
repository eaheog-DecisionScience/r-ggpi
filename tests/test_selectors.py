import numpy as np
import pandas as pd

from r_ggpi import RGGPISelector


def test_selector_fit_and_transform_with_dataframe():
  X = pd.DataFrame(
      {
          "signal": [0, 1, 0, 1, 0, 1],
          "noise": [1, 0, 1, 1, 0, 0],
      }
  )
  y = np.array([0, 1, 0, 1, 0, 1])

  class SignalScorer:
    def score(self, features, target):
      return -np.mean((features["signal"].to_numpy() - target) ** 2)

  selector = RGGPISelector(SignalScorer(), n_repeats=2, max_features=2)

  transformed = selector.fit_transform(X, y)

  assert transformed.shape == (len(X), 2)
  assert list(transformed.columns) == selector.selected_features_
  assert selector.feature_groups_


def test_selector_selects_complete_positive_groups_by_default():
  rng = np.random.default_rng(42)
  signal = rng.normal(size=300)
  X = pd.DataFrame(
      {
          "signal": signal,
          "duplicate": signal.copy(),
          "noise": rng.normal(size=len(signal)),
      }
  )
  y = signal.copy()

  class SignalScorer:
    def score(self, features, target):
      return -np.mean((features["signal"].to_numpy() - target) ** 2)

  selector = RGGPISelector(
      SignalScorer(), pruning_percentile=50, n_repeats=3, random_state=10
  )
  transformed = selector.fit_transform(X, y)

  assert selector.selected_groups_ == ["group_0"]
  assert set(transformed.columns) == {"signal", "duplicate"}
  assert set(selector.selected_features_) == {
      feature
      for group_name in selector.selected_groups_
      for feature in selector.feature_groups_[group_name]
  }


def test_selector_max_features_is_a_strict_whole_group_cap():
  rng = np.random.default_rng(42)
  signal = rng.normal(size=300)
  X = pd.DataFrame(
      {
          "signal": signal,
          "duplicate": signal.copy(),
          "noise": rng.normal(size=len(signal)),
      }
  )

  class SignalScorer:
    def score(self, features, target):
      return -np.mean((features["signal"].to_numpy() - target) ** 2)

  selector = RGGPISelector(
      SignalScorer(),
      pruning_percentile=50,
      n_repeats=3,
      max_features=1,
      random_state=10,
  )

  try:
    selector.fit(X, signal)
  except ValueError as exc:
    assert "No positively scoring feature group fits max_features" in str(exc)
  else:
    raise AssertionError("Expected a clear error when no full group fits the cap.")
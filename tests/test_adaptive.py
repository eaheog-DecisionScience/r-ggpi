import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.dummy import DummyClassifier

from r_ggpi import RGGPIExplainer, RGGPISelector, r_ggpi_importance


def test_adaptive_grouping_isolates_noise_features():
  """Independent noise features should not be grouped with real features."""
  rng = np.random.default_rng(42)
  signal = rng.normal(size=200)
  duplicate = signal + rng.normal(scale=0.05, size=200)
  noise_1 = rng.normal(size=200)
  noise_2 = rng.normal(size=200)

  X = pd.DataFrame({
      "signal": signal,
      "duplicate": duplicate,
      "noise_1": noise_1,
      "noise_2": noise_2,
  })
  y = signal.copy()

  class SignalScorer:
    def score(self, features, target):
      return np.corrcoef(features["signal"].to_numpy(), target)[0, 1]

  explainer = RGGPIExplainer(
      SignalScorer(),
      grouping_method="adaptive",
      random_state=42,
  )
  res = explainer.explain(X, y)

  # Check groups
  signal_group = None
  for g, feats in res.feature_groups.items():
    if "signal" in feats:
      signal_group = feats

  assert "duplicate" in signal_group
  assert "noise_1" not in signal_group
  assert "noise_2" not in signal_group
  # Both noise features should be in singleton groups
  for g, feats in res.feature_groups.items():
    if "noise_1" in feats:
      assert len(feats) == 1
    if "noise_2" in feats:
      assert len(feats) == 1


def test_adaptive_group_size_capping():
  """Oversized components should be split into cohesive sub-clusters."""
  rng = np.random.default_rng(42)
  base = rng.normal(size=100)
  # Create 10 collinear features
  data = {f"feat_{i}": base + rng.normal(scale=0.1, size=100) for i in range(10)}
  X = pd.DataFrame(data)
  y = (base > 0).astype(int)

  model = DummyClassifier(strategy="most_frequent").fit(X, y)
  max_size = 4
  explainer = RGGPIExplainer(
      model,
      grouping_method="adaptive",
      max_group_size=max_size,
      random_state=42,
  )
  res = explainer.explain(X, y)

  # No group should exceed max_size
  for g, feats in res.feature_groups.items():
    assert len(feats) <= max_size
  # Should have formed at least ceil(10 / 4) = 3 groups
  assert len(res.feature_groups) >= 3


def test_stability_scores_and_distribution_shifts_present():
  """ResultContainer should report stability scores and distribution shifts."""
  rng = np.random.default_rng(42)
  X = pd.DataFrame({
      "f1": rng.normal(size=100),
      "f2": rng.normal(size=100),
  })
  y = np.array([0, 1] * 50)
  model = DummyClassifier(strategy="most_frequent").fit(X, y)

  explainer = RGGPIExplainer(model, random_state=42)
  res = explainer.explain(X, y)

  assert isinstance(res.group_stability_scores, pd.Series)
  assert isinstance(res.group_distribution_shifts, pd.Series)
  assert len(res.group_stability_scores) == len(res.feature_groups)
  assert len(res.group_distribution_shifts) == len(res.feature_groups)
  assert np.all(res.group_distribution_shifts >= 0.0)
  assert "min_correlation" in res.grouping_metadata
  assert "max_group_size" in res.grouping_metadata


def test_selector_adaptive_selection_criteria():
  """RGGPISelector should support importance, stability, and robust ranking."""
  rng = np.random.default_rng(42)
  signal = rng.normal(size=200)
  X = pd.DataFrame({
      "signal_a": signal,
      "signal_b": signal + rng.normal(scale=0.01, size=200),
      "noise": rng.normal(size=200),
  })
  y = signal.copy()

  class SignalScorer:
    def score(self, features, target):
      return -np.mean((features["signal_a"].to_numpy() - target) ** 2)

  for crit in ["importance", "stability", "robust"]:
    sel = RGGPISelector(
        SignalScorer(),
        selection_criterion=crit,
        n_repeats=3,
        random_state=42,
    )
    sel.fit(X, y)
    assert "signal_a" in sel.selected_features_
    assert "signal_b" in sel.selected_features_
    assert "noise" not in sel.selected_features_


def test_breast_cancer_real_world_benchmark():
  """Verify adaptive R-GGPI selects top predictive features on Breast Cancer."""
  data = load_breast_cancer()
  X = pd.DataFrame(data.data[:200], columns=data.feature_names)
  y = data.target[:200]

  model = RandomForestClassifier(n_estimators=30, random_state=42)
  model.fit(X, y)

  sel = RGGPISelector(
      model,
      max_features=6,
      grouping_method="adaptive",
      random_state=42,
  )
  sel.fit(X, y)

  assert len(sel.selected_features_) <= 6
  assert len(sel.selected_features_) >= 2
  # Should retain core geometry/texture features without raising error
  transformed = sel.transform(X)
  assert transformed.shape[1] == len(sel.selected_features_)


def test_selector_auto_max_features():
  """Verify max_features='auto' dynamically selects the optimal subset."""
  from sklearn.datasets import make_classification
  X_vals, y = make_classification(
      n_samples=300,
      n_features=20,
      n_informative=3,
      n_redundant=0,
      random_state=42,
  )
  X = pd.DataFrame(X_vals, columns=[f"feat_{i}" for i in range(20)])
  model = RandomForestClassifier(n_estimators=30, random_state=42).fit(X, y)

  sel = RGGPISelector(
      model,
      max_features="auto",
      cumulative_importance=0.90,
      random_state=42,
  )
  sel.fit(X, y)

  # Should select substantially fewer than 20 features (typically 3 to 6)
  assert 1 <= len(sel.selected_features_) <= 8
  transformed = sel.transform(X)
  assert transformed.shape[1] == len(sel.selected_features_)

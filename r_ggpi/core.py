import numpy as np
import pandas as pd
import networkx as nx
from joblib import Parallel, delayed
from .utils import validate_inputs, compute_target_correlations


class RGGPIExplainer:
  """Robust Graph-Grouped Permutation Importance (R-GGPI) Explainer.

  Parameters:
  -----------
  model : estimator object
      A fitted scikit-learn compatible estimator (e.g., RandomForestClassifier).
  pruning_percentile : float (default=75.0)
      Percentile threshold for MST edge pruning to form feature branches.
  n_repeats : int (default=5)
      Number of permutation repeats to stabilize importance scores.
  n_jobs : int (default=1)
      Number of CPU cores to use if complexity threshold is met.
  parallel_threshold : int (default=1_000_000)
      Complexity proxy threshold (n_samples * n_features * n_repeats)
      to trigger multiprocessing and avoid small-scale spawning overhead.
  """

  def __init__(
      self,
      model,
      pruning_percentile=75.0,
      n_repeats=5,
      n_jobs=1,
      parallel_threshold=1_000_000,
  ):
    self.model = model
    self.pruning_percentile = pruning_percentile
    self.n_repeats = n_repeats
    self.n_jobs = n_jobs
    self.parallel_threshold = parallel_threshold

  def _compute_single_repeat(
      self, X_filtered, y, baseline_score, communities, active_features, G, target_corrs
  ):
    importances = {feat: 0.0 for feat in X_filtered.columns}
    X_permuted = X_filtered.copy()

    for community in communities:
      nodes = sorted(
          list(community),
          key=lambda n: target_corrs[active_features[n]],
          reverse=True,
      )
      branch_bias = np.linspace(1.0, 0.2, len(nodes))

      node_vars = np.array([G.nodes[n]["variance"] for n in nodes])
      total_var = np.sum(node_vars) if np.sum(node_vars) > 0 else 1.0

      for idx, node in enumerate(nodes):
        feat_name = active_features[node]
        original_vals = X_permuted[feat_name].values.copy()
        shuffled_vals = np.random.permutation(original_vals)
        X_permuted[feat_name] = (
            original_vals * (1 - branch_bias[idx])
            + shuffled_vals * branch_bias[idx]
        )

      perm_score = self.model.score(X_permuted, y)
      group_drop = max(0.0, baseline_score - perm_score)

      for idx, node in enumerate(nodes):
        feat_name = active_features[node]
        node_weight = branch_bias[idx] * (G.nodes[node]["variance"] / total_var)
        importances[feat_name] = group_drop * node_weight

      for node in nodes:
        X_permuted[active_features[node]] = X_filtered.iloc[:, node].values

    return importances

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

    # Graph & MST Construction
    corr_matrix = X_filtered.corr().abs().fillna(0).values
    dist_matrix = 1.0 - corr_matrix

    G = nx.Graph()
    for i in range(n_active):
      G.add_node(i, name=active_features[i], variance=X_filtered.iloc[:, i].var())

    for i in range(n_active):
      for j in range(i + 1, n_active):
        G.add_edge(i, j, weight=dist_matrix[i, j])

    mst = nx.minimum_spanning_tree(G, weight="weight")

    # Dynamic Edge Pruning
    edge_weights = [data["weight"] for _, _, data in mst.edges(data=True)]
    if edge_weights:
      cutoff_weight = np.percentile(edge_weights, self.pruning_percentile)
      edges_to_remove = [
          (u, v) for u, v, data in mst.edges(data=True) if data["weight"] > cutoff_weight
      ]
      mst.remove_edges_from(edges_to_remove)

    communities = list(nx.connected_components(mst))

    # Smart Gating Proxy
    complexity_score = n_samples * n_features * self.n_repeats
    use_parallel = (
        self.n_jobs != 1 and complexity_score > self.parallel_threshold
    )

    if use_parallel:
      parallel_results = Parallel(n_jobs=self.n_jobs)(
          delayed(self._compute_single_repeat)(
              X_filtered,
              y,
              baseline_score,
              communities,
              active_features,
              G,
              target_corrs,
          )
          for _ in range(self.n_repeats)
      )
      importance_accumulator = {feat: [] for feat in feature_cols}
      for res in parallel_results:
        for feat, val in res.items():
          importance_accumulator[feat].append(val)
    else:
      importance_accumulator = {feat: [] for feat in feature_cols}
      for _ in range(self.n_repeats):
        res = self._compute_single_repeat(
            X_filtered,
            y,
            baseline_score,
            communities,
            active_features,
            G,
            target_corrs,
        )
        for feat, val in res.items():
          importance_accumulator[feat].append(val)

    mean_importances = pd.Series(
        {feat: np.mean(vals) for feat, vals in importance_accumulator.items()}
    )
    std_importances = pd.Series(
        {feat: np.std(vals) for feat, vals in importance_accumulator.items()}
    )

    class ResultContainer:

      def __init__(self, importances_mean, importances_std, n_branches):
        self.importances_mean = importances_mean
        self.importances_std = importances_std
        self.n_branches = n_branches

    return ResultContainer(mean_importances, std_importances, len(communities))


def r_ggpi_importance(model, X, y, pruning_percentile=75.0, n_repeats=5, n_jobs=1):
  explainer = RGGPIExplainer(
      model=model,
      pruning_percentile=pruning_percentile,
      n_repeats=n_repeats,
      n_jobs=n_jobs,
  )
  return explainer.explain(X, y)
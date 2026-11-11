import numpy as np
import pandas as pd
import networkx as nx
from sklearn.utils import check_random_state


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
      feature_values = X.loc[valid_mask, col].to_numpy()
      target_values = y[valid_mask]
      if np.ptp(feature_values) > 0 and np.ptp(target_values) > 0:
        corr = np.corrcoef(feature_values, target_values)[0, 1]
        target_corrs[col] = abs(corr) if not np.isnan(corr) else 0.0
      else:
        target_corrs[col] = 0.0
    else:
      target_corrs[col] = 0.0
  return target_corrs


def compute_stable_correlation_matrix(
    X,
    method="spearman",
    n_resamples=5,
    penalty=1.0,
    random_state=None,
):
  """Computes correlation matrix with stability penalty across subsamples.

  Penalizes fragile or fluctuating correlation pairs across resamples to guard
  against spurious grouping and noise connections.
  """
  n_samples, n_features = X.shape
  if n_samples < 20 or n_resamples <= 1:
    corr = X.corr(method=method).abs().fillna(0).to_numpy().copy()
    np.fill_diagonal(corr, 1.0)
    return corr, corr, np.zeros_like(corr)

  rng = check_random_state(random_state)
  corr_matrices = []
  sub_size = max(10, int(0.8 * n_samples))
  for _ in range(n_resamples):
    sub_idx = rng.choice(n_samples, size=sub_size, replace=True)
    X_sub = X.iloc[sub_idx]
    c = X_sub.corr(method=method).abs().fillna(0).to_numpy().copy()
    np.fill_diagonal(c, 1.0)
    corr_matrices.append(c)

  mean_corr = np.mean(corr_matrices, axis=0)
  std_corr = np.std(corr_matrices, axis=0)
  stable_corr = np.maximum(0.0, mean_corr - penalty * std_corr)
  np.fill_diagonal(stable_corr, 1.0)
  return stable_corr, mean_corr, std_corr


def compute_distribution_shifts(feature_groups, corr_matrix, active_features):
  """Computes cross-group correlation energy (distributional shift risk) for each group."""
  feat_to_idx = {feat: idx for idx, feat in enumerate(active_features)}
  n_active = len(active_features)
  shifts = {}
  for group_name, feats in feature_groups.items():
    if len(feats) == n_active or n_active <= 1:
      shifts[group_name] = 0.0
      continue
    in_idx = [feat_to_idx[f] for f in feats if f in feat_to_idx]
    out_idx = [i for i in range(n_active) if i not in in_idx]
    if not in_idx or not out_idx:
      shifts[group_name] = 0.0
      continue
    cross_sub = corr_matrix[np.ix_(in_idx, out_idx)]
    shift_val = float(np.mean(cross_sub ** 2))
    shifts[group_name] = shift_val
  return pd.Series(shifts, dtype=float)


def build_feature_groups(
    active_features,
    corr_matrix,
    grouping_method="adaptive",
    min_correlation="auto",
    max_group_size="auto",
    pruning_percentile=75.0,
    n_samples=None,
):
  """Builds feature groups using MST with adaptive pruning and max group size constraints."""
  n_active = len(active_features)
  if n_active <= 1:
    communities = [[0]] if n_active == 1 else []
    feature_groups = {
        "group_0": tuple(active_features) if n_active == 1 else ()
    }
    return communities, feature_groups, {}

  dist_matrix = np.clip(1.0 - corr_matrix, 0.0, 1.0)
  G = nx.Graph()
  for i in range(n_active):
    G.add_node(i, name=active_features[i])

  for i in range(n_active):
    for j in range(i + 1, n_active):
      G.add_edge(i, j, weight=dist_matrix[i, j])

  mst = nx.minimum_spanning_tree(G, weight="weight")

  metadata = {
      "grouping_method": grouping_method,
      "n_active_features": n_active,
  }

  if grouping_method == "percentile":
    edge_weights = [data["weight"] for _, _, data in mst.edges(data=True)]
    if edge_weights:
      cutoff_weight = np.percentile(edge_weights, pruning_percentile)
      edges_to_remove = [
          (u, v)
          for u, v, data in mst.edges(data=True)
          if data["weight"] > cutoff_weight
      ]
      mst.remove_edges_from(edges_to_remove)
    communities = list(nx.connected_components(mst))
    metadata["cutoff_distance"] = float(cutoff_weight) if edge_weights else 0.0
  else:
    # Adaptive grouping
    if min_correlation == "auto":
      if n_samples is not None and n_samples >= 20:
        eff_min_corr = min(0.5, max(0.15, 2.5 / np.sqrt(n_samples)))
      else:
        eff_min_corr = 0.30
    else:
      eff_min_corr = float(min_correlation)
    metadata["min_correlation"] = eff_min_corr

    # Prune edges with correlation below effective threshold
    cutoff_dist = 1.0 - eff_min_corr
    edges_to_remove = [
        (u, v)
        for u, v, data in mst.edges(data=True)
        if data["weight"] > cutoff_dist
    ]
    mst.remove_edges_from(edges_to_remove)

    # Determine max group size constraint
    if max_group_size == "auto":
      eff_max_group_size = max(5, int(np.ceil(np.sqrt(n_active))))
    elif max_group_size is None:
      eff_max_group_size = n_active
    else:
      eff_max_group_size = max(2, int(max_group_size))
    metadata["max_group_size"] = eff_max_group_size

    # Split oversized components along minimum-affinity (maximum-distance) edges
    raw_communities = list(nx.connected_components(mst))
    refined_communities = []
    for comm in raw_communities:
      nodes = list(comm)
      if len(nodes) <= eff_max_group_size:
        refined_communities.append(nodes)
      else:
        sub_g = mst.subgraph(nodes).copy()
        while True:
          comps = list(nx.connected_components(sub_g))
          oversized = [c for c in comps if len(c) > eff_max_group_size]
          if not oversized:
            refined_communities.extend(comps)
            break
          target_comp = max(oversized, key=len)
          sub_comp_g = sub_g.subgraph(target_comp)
          heaviest_edge = max(
              sub_comp_g.edges(data=True), key=lambda x: x[2]["weight"]
          )
          sub_g.remove_edge(heaviest_edge[0], heaviest_edge[1])

    communities = refined_communities

  communities.sort(key=lambda community: min(community))
  feature_groups = {
      f"group_{group_index}": tuple(
          active_features[node] for node in sorted(community)
      )
      for group_index, community in enumerate(communities)
  }
  metadata["n_groups"] = len(feature_groups)
  return communities, feature_groups, metadata

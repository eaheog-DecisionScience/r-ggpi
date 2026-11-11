import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
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
  """Builds feature groups using MST with adaptive pruning and max group size constraints.

  Uses scipy.sparse.csgraph for MST computation (faster than NetworkX for dense
  distance matrices).
  """
  n_active = len(active_features)
  if n_active <= 1:
    communities = [[0]] if n_active == 1 else []
    feature_groups = {
        "group_0": tuple(active_features) if n_active == 1 else ()
    }
    return communities, feature_groups, {}

  # Build dense distance matrix and compute MST via scipy
  dist_matrix = np.clip(1.0 - corr_matrix, 0.0, 1.0)
  np.fill_diagonal(dist_matrix, 0.0)
  # scipy minimum_spanning_tree treats explicit zeros in CSR as absent edges.
  # Keep zero-distance edges as a tiny positive value in the sparse graph.
  _EPSILON = 1e-12
  dist_for_mst = dist_matrix.copy()
  zero_rows, zero_cols = np.where(
      (dist_matrix == 0) & ~np.eye(n_active, dtype=bool)
  )
  dist_for_mst[zero_rows, zero_cols] = _EPSILON

  dist_sparse = csr_matrix(dist_for_mst)
  mst_sparse = minimum_spanning_tree(dist_sparse)

  # Convert MST to a symmetric representation for pruning / component analysis
  mst_sym = mst_sparse + mst_sparse.T  # make undirected

  metadata = {
      "grouping_method": grouping_method,
      "n_active_features": n_active,
  }

  if grouping_method == "percentile":
    edge_weights = np.where(mst_sparse.data == _EPSILON, 0.0, mst_sparse.data)
    if len(edge_weights) > 0:
      cutoff_weight = np.percentile(edge_weights, pruning_percentile)
      # Zero out edges above the cutoff (remove them)
      mst_sym = mst_sym.tocsr()
      effective_weights = np.where(mst_sym.data == _EPSILON, 0.0, mst_sym.data)
      mst_sym.data[effective_weights > cutoff_weight] = 0
      mst_sym.eliminate_zeros()
    n_components, labels = connected_components(mst_sym, directed=False)
    communities = _labels_to_communities(labels, n_components)
    metadata["cutoff_distance"] = float(cutoff_weight) if len(edge_weights) > 0 else 0.0
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
    mst_sym = mst_sym.tocsr()
    effective_weights = np.where(mst_sym.data == _EPSILON, 0.0, mst_sym.data)
    mst_sym.data[effective_weights > cutoff_dist] = 0
    mst_sym.eliminate_zeros()

    # Determine max group size constraint
    if max_group_size == "auto":
      eff_max_group_size = max(5, int(np.ceil(np.sqrt(n_active))))
    elif max_group_size is None:
      eff_max_group_size = n_active
    else:
      eff_max_group_size = max(2, int(max_group_size))
    metadata["max_group_size"] = eff_max_group_size

    # Split oversized components along minimum-affinity (maximum-distance) edges
    n_components, labels = connected_components(mst_sym, directed=False)
    raw_communities = _labels_to_communities(labels, n_components)

    refined_communities = []
    for comm in raw_communities:
      if len(comm) <= eff_max_group_size:
        refined_communities.append(comm)
      else:
        # Iteratively remove heaviest edge in the sub-MST until all components
        # are within the size limit. Work with the subgraph of the MST.
        sub_nodes = sorted(comm)
        node_to_local = {n: i for i, n in enumerate(sub_nodes)}
        local_to_node = {i: n for n, i in node_to_local.items()}
        n_sub = len(sub_nodes)

        # Extract sub-MST as dense (small enough after initial pruning)
        sub_adj = np.zeros((n_sub, n_sub))
        cx = mst_sym.tocoo()
        for r, c, v in zip(cx.row, cx.col, cx.data):
          if r in node_to_local and c in node_to_local:
            lr, lc = node_to_local[r], node_to_local[c]
            sub_adj[lr, lc] = v
            sub_adj[lc, lr] = v

        while True:
          sub_sparse = csr_matrix(sub_adj)
          n_c, sub_labels = connected_components(sub_sparse, directed=False)
          sub_comms = _labels_to_communities(sub_labels, n_c)
          oversized = [sc for sc in sub_comms if len(sc) > eff_max_group_size]
          if not oversized:
            for sc in sub_comms:
              refined_communities.append([local_to_node[i] for i in sc])
            break
          target_comm = max(oversized, key=len)
          # Find the heaviest edge within this component
          best_edge = None
          best_weight = -1.0
          for i in target_comm:
            for j in target_comm:
              if j > i and sub_adj[i, j] > best_weight:
                best_weight = sub_adj[i, j]
                best_edge = (i, j)
          if best_edge is None:
            # No edges to remove; break out
            for sc in sub_comms:
              refined_communities.append([local_to_node[i] for i in sc])
            break
          sub_adj[best_edge[0], best_edge[1]] = 0
          sub_adj[best_edge[1], best_edge[0]] = 0

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


def _labels_to_communities(labels, n_components):
  """Convert scipy connected_components labels array to list of node-index lists."""
  communities = [[] for _ in range(n_components)]
  for node_idx, label in enumerate(labels):
    communities[label].append(node_idx)
  # Remove empty communities (shouldn't happen, but defensive)
  return [c for c in communities if c]

import numpy as np
import torch


TASK_ORACLE_BOUNDS = {
    "TFBind8-Exact-v0": {"min": 0.0, "max": 1.0, "best": 0.43929616},
    "TFBind10-Exact-v0": {"min": -1.8585268, "max": 2.1287067, "best": 0.005328223},
    "AntMorphology-Exact-v0": {"min": -386.90036, "max": 590.24445, "best": 165.32648},
    "DKittyMorphology-Exact-v0": {"min": -880.4585, "max": 340.90985, "best": 199.36252},
    "Superconductor-RandomForest-v0": {"min": 0.00021, "max": 185.0, "best": 0.0},
}


def oracle_bounds(task_name, scores=None):
    if task_name in TASK_ORACLE_BOUNDS:
        bounds = TASK_ORACLE_BOUNDS[task_name]
        return float(bounds["min"]), float(bounds["max"])
    if scores is None:
        raise KeyError(f"No oracle bounds for {task_name}")
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    return float(np.min(scores)), float(np.max(scores))


def normalize_oracle_scores(scores, task_name):
    scores = np.asarray(scores, dtype=np.float32)
    y_min, y_max = oracle_bounds(task_name, scores)
    return ((scores - y_min) / (y_max - y_min + 1e-8)).astype(np.float32)


def normalized_score_summary(scores, task_name):
    normalized = normalize_oracle_scores(scores, task_name).reshape(-1)
    percentiles = torch.quantile(
        torch.from_numpy(normalized),
        torch.tensor([1.0, 0.8, 0.5]),
        interpolation="higher",
    )
    return {
        "normalized_min": float(np.min(normalized)),
        "normalized_max": float(np.max(normalized)),
        "normalized_mean": float(np.mean(normalized)),
        "normalized_std": float(np.std(normalized)),
        "normalized_p100": float(percentiles[0].item()),
        "normalized_p80": float(percentiles[1].item()),
        "normalized_p50": float(percentiles[2].item()),
    }


def offline_scores_for_indices(y_values, indices):
    y_values = np.asarray(y_values, dtype=np.float32).reshape(-1)
    return y_values[np.asarray(indices, dtype=np.int64)].reshape(-1)


def sample_proposal_diagnostic_indices(num_proposals, selected_indices, max_raw=0, seed=0):
    num_proposals = int(num_proposals)
    if num_proposals <= 0:
        return np.zeros((0,), dtype=np.int64)
    selected_indices = np.asarray(selected_indices, dtype=np.int64).reshape(-1)
    selected_indices = np.unique(selected_indices[(selected_indices >= 0) & (selected_indices < num_proposals)])
    if max_raw is None or int(max_raw) <= 0:
        return np.arange(num_proposals, dtype=np.int64)

    all_indices = np.arange(num_proposals, dtype=np.int64)
    raw_candidates = np.setdiff1d(all_indices, selected_indices, assume_unique=True)
    max_raw = min(int(max_raw), raw_candidates.size)
    if max_raw > 0:
        rng = np.random.default_rng(int(seed))
        sampled_raw = np.sort(rng.choice(raw_candidates, size=max_raw, replace=False)).astype(np.int64)
    else:
        sampled_raw = np.zeros((0,), dtype=np.int64)
    return np.concatenate([selected_indices.astype(np.int64), sampled_raw]).astype(np.int64)

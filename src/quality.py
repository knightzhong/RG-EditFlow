import numpy as np


def _as_float_array(values):
    return np.asarray(values, dtype=np.float32)


def _normalize_positive(values, eps=1e-8):
    values = _as_float_array(values)
    if values.size == 0:
        return values
    min_value = float(np.min(values))
    shifted = values - min_value
    max_value = float(np.max(shifted))
    if max_value < eps:
        return np.ones_like(values, dtype=np.float32)
    return shifted / (max_value + eps)


def compute_quality_weights(
    improvement,
    monotonicity,
    uncertainty,
    manifold_distance,
    mode="full",
    improvement_temperature=2.0,
    uncertainty_weight=0.25,
    distance_weight=0.25,
    min_weight=0.05,
):
    improvement = _as_float_array(improvement)
    monotonicity = np.clip(_as_float_array(monotonicity), 0.0, 1.0)
    uncertainty = _as_float_array(uncertainty)
    manifold_distance = _as_float_array(manifold_distance)
    if mode not in {"none", "score", "geometry", "full"}:
        raise ValueError("mode must be one of: none, score, geometry, full")
    if mode == "none":
        return np.ones_like(improvement, dtype=np.float32)

    positive_improvement = np.maximum(improvement, 0.0)
    improvement_score = _normalize_positive(positive_improvement)
    improvement_score = np.power(improvement_score, 1.0 / max(improvement_temperature, 1e-8))
    uncertainty_penalty = np.exp(-uncertainty_weight * _normalize_positive(uncertainty))
    distance_penalty = np.exp(-distance_weight * _normalize_positive(manifold_distance))

    score_weight = improvement_score * monotonicity
    if mode == "score":
        weights = score_weight
    elif mode == "geometry":
        weights = distance_penalty
    else:
        weights = score_weight * uncertainty_penalty * distance_penalty
    return np.clip(weights, min_weight, 1.0).astype(np.float32)


def conservative_rerank_scores(
    predicted_score,
    uncertainty,
    manifold_distance,
    diversity_bonus=None,
    uncertainty_weight=0.25,
    distance_weight=0.1,
    diversity_weight=0.0,
):
    predicted_score = _as_float_array(predicted_score)
    uncertainty = _as_float_array(uncertainty)
    manifold_distance = _as_float_array(manifold_distance)

    scores = predicted_score - uncertainty_weight * uncertainty - distance_weight * manifold_distance
    if diversity_bonus is not None:
        scores = scores + diversity_weight * _as_float_array(diversity_bonus)
    return scores.astype(np.float32)


def trust_region_rerank_scores(
    predicted_score,
    uncertainty,
    manifold_distance,
    seed_displacement,
    diversity_bonus=None,
    uncertainty_weight=0.25,
    distance_weight=0.1,
    displacement_weight=0.0,
    diversity_weight=0.0,
):
    scores = conservative_rerank_scores(
        predicted_score,
        uncertainty,
        manifold_distance,
        diversity_bonus=diversity_bonus,
        uncertainty_weight=uncertainty_weight,
        distance_weight=distance_weight,
        diversity_weight=diversity_weight,
    )
    if displacement_weight > 0.0:
        scores = scores - float(displacement_weight) * _as_float_array(seed_displacement)
    return scores.astype(np.float32)


def clip_proposals_to_trust_region(proposals, seed_ids, seeds, max_displacement=None):
    proposals = _as_float_array(proposals)
    seed_ids = np.asarray(seed_ids)
    seeds = _as_float_array(seeds)
    if max_displacement is None or float(max_displacement) <= 0.0:
        return proposals

    max_displacement = float(max_displacement)
    seed_points = seeds[seed_ids]
    offsets = proposals - seed_points
    distances = np.linalg.norm(offsets, axis=1, keepdims=True)
    scale = np.minimum(1.0, max_displacement / np.maximum(distances, 1e-8))
    return (seed_points + offsets * scale).astype(np.float32)


def select_best_per_seed_with_fallback(proposals, scores, seed_ids, seeds, min_score=None):
    proposals = _as_float_array(proposals)
    scores = _as_float_array(scores).reshape(-1)
    seed_ids = np.asarray(seed_ids)
    seeds = _as_float_array(seeds)

    selected = []
    fallback_mask = []
    unique_seed_ids = np.unique(seed_ids)
    for seed_id in unique_seed_ids:
        indices = np.where(seed_ids == seed_id)[0]
        best_index = indices[np.argmax(scores[indices])]

        threshold = None
        if min_score is not None:
            min_scores = np.asarray(min_score, dtype=np.float32)
            threshold = float(min_scores.reshape(-1)[int(seed_id)] if min_scores.size > 1 else min_scores.reshape(-1)[0])

        if threshold is not None and float(scores[best_index]) < threshold:
            selected.append(seeds[int(seed_id)])
            fallback_mask.append(True)
        else:
            selected.append(proposals[best_index])
            fallback_mask.append(False)

    return np.stack(selected, axis=0).astype(np.float32), np.asarray(fallback_mask, dtype=bool)


def select_top_indices(scores, k):
    scores = _as_float_array(scores)
    if k <= 0:
        return np.array([], dtype=np.int64)
    k = min(k, scores.shape[0])
    return np.argsort(-scores)[:k].astype(np.int64)


def _to_numpy_scalar(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return float(np.asarray(value).reshape(-1)[0])


def _to_numpy_vector(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32).reshape(-1)


def quality_from_gp_samples(GP_samples, mode="full"):
    improvements = []
    monotonicity = []
    uncertainty = []
    manifold_distance = []

    for _, samples in GP_samples.items():
        for sample in samples:
            trajectory_scores = None
            if isinstance(sample, dict):
                x_high = sample["x_high"]
                y_high = sample["y_high"]
                x_low = sample["x_low"]
                y_low = sample["y_low"]
                trajectory_scores = sample.get("trajectory_scores")
            else:
                (x_high, y_high), (x_low, y_low) = sample
            x_high_np = _to_numpy_vector(x_high)
            x_low_np = _to_numpy_vector(x_low)
            improvement = _to_numpy_scalar(y_high) - _to_numpy_scalar(y_low)
            step_distance = float(np.linalg.norm(x_high_np - x_low_np))

            if trajectory_scores is not None:
                scores = _as_float_array(_to_numpy_vector(trajectory_scores))
                score_steps = np.diff(scores)
                mono = float(np.mean(score_steps >= -1e-6)) if score_steps.size > 0 else float(improvement > 0.0)
                unc = float(np.std(score_steps)) if score_steps.size > 0 else 1.0 / max(abs(improvement), 1e-6)
            else:
                mono = 1.0 if improvement > 0.0 else 0.0
                unc = 1.0 / max(abs(improvement), 1e-6)

            improvements.append(improvement)
            monotonicity.append(mono)
            uncertainty.append(unc)
            manifold_distance.append(step_distance)

    improvements = np.asarray(improvements, dtype=np.float32)
    monotonicity = np.asarray(monotonicity, dtype=np.float32)
    uncertainty = np.asarray(uncertainty, dtype=np.float32)
    manifold_distance = np.asarray(manifold_distance, dtype=np.float32)
    weights = compute_quality_weights(
        improvement=improvements,
        monotonicity=monotonicity,
        uncertainty=uncertainty,
        manifold_distance=manifold_distance,
        mode=mode,
    )
    return {
        "improvement": improvements,
        "monotonicity": monotonicity,
        "uncertainty": uncertainty,
        "manifold_distance": manifold_distance,
        "weights": weights,
    }


def knn_proxy_scores(candidates, train_x, train_y, k=5, chunk_size=512):
    candidates = _as_float_array(candidates)
    train_x = _as_float_array(train_x)
    train_y = _as_float_array(train_y).reshape(-1)
    k = max(1, min(int(k), train_x.shape[0]))

    predicted_scores = []
    uncertainties = []
    manifold_distances = []

    for start in range(0, candidates.shape[0], chunk_size):
        chunk = candidates[start:start + chunk_size]
        diff = chunk[:, None, :] - train_x[None, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        nn_idx = np.argpartition(dists, kth=k - 1, axis=1)[:, :k]
        nn_dists = np.take_along_axis(dists, nn_idx, axis=1)
        nn_scores = train_y[nn_idx]

        predicted_scores.append(np.mean(nn_scores, axis=1))
        uncertainties.append(np.std(nn_scores, axis=1))
        manifold_distances.append(np.mean(nn_dists, axis=1))

    return {
        "predicted_score": np.concatenate(predicted_scores).astype(np.float32),
        "uncertainty": np.concatenate(uncertainties).astype(np.float32),
        "manifold_distance": np.concatenate(manifold_distances).astype(np.float32),
    }


def select_best_per_seed_indices(scores, seed_ids, k):
    scores = _as_float_array(scores)
    seed_ids = np.asarray(seed_ids)
    if k <= 0:
        return np.array([], dtype=np.int64)

    best_by_seed = []
    for seed_id in np.unique(seed_ids):
        indices = np.where(seed_ids == seed_id)[0]
        best_local = indices[np.argmax(scores[indices])]
        best_by_seed.append(best_local)

    best_by_seed = np.asarray(best_by_seed, dtype=np.int64)
    ordered = best_by_seed[np.argsort(-scores[best_by_seed])]
    return ordered[:min(k, ordered.shape[0])].astype(np.int64)


def bootstrap_knn_proxy_scores(candidates, train_x, train_y, k=8, num_bootstrap=16, seed=0, chunk_size=512):
    candidates = _as_float_array(candidates)
    train_x = _as_float_array(train_x)
    train_y = _as_float_array(train_y).reshape(-1)
    k = max(1, min(int(k), train_x.shape[0]))
    rng = np.random.default_rng(seed)

    predicted_scores = []
    uncertainties = []
    manifold_distances = []

    for start in range(0, candidates.shape[0], chunk_size):
        chunk = candidates[start:start + chunk_size]
        diff = chunk[:, None, :] - train_x[None, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        nn_idx = np.argpartition(dists, kth=k - 1, axis=1)[:, :k]
        nn_dists = np.take_along_axis(dists, nn_idx, axis=1)
        nn_scores = train_y[nn_idx]

        boot_means = []
        for _ in range(max(1, int(num_bootstrap))):
            boot_cols = rng.integers(0, k, size=(nn_scores.shape[0], k))
            boot_scores = np.take_along_axis(nn_scores, boot_cols, axis=1)
            boot_means.append(np.mean(boot_scores, axis=1))
        boot_means = np.stack(boot_means, axis=1)

        predicted_scores.append(np.mean(boot_means, axis=1))
        uncertainties.append(np.std(boot_means, axis=1))
        manifold_distances.append(np.mean(nn_dists, axis=1))

    return {
        "predicted_score": np.concatenate(predicted_scores).astype(np.float32),
        "uncertainty": np.concatenate(uncertainties).astype(np.float32),
        "manifold_distance": np.concatenate(manifold_distances).astype(np.float32),
    }


def filter_trajectories_by_quality(trajectories, quality, keep_ratio=1.0):
    trajectories = _as_float_array(trajectories)
    if quality is None or "weights" not in quality:
        return trajectories, quality

    weights = _as_float_array(quality["weights"]).reshape(-1)
    if keep_ratio >= 1.0 or weights.shape[0] == 0:
        return trajectories, quality

    keep_ratio = max(0.0, float(keep_ratio))
    keep_count = int(np.ceil(weights.shape[0] * keep_ratio))
    keep_count = max(1, min(keep_count, weights.shape[0]))
    selected = np.argsort(-weights)[:keep_count]

    filtered_quality = {}
    for key, value in quality.items():
        array_value = np.asarray(value)
        if array_value.shape[:1] == weights.shape[:1]:
            filtered_quality[key] = array_value[selected]
        else:
            filtered_quality[key] = value

    return trajectories[selected], filtered_quality

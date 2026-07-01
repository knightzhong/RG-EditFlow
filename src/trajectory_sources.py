import numpy as np

from src.quality import compute_quality_weights


def generate_knn_mixup_trajectories(
    x_train,
    y_train,
    num_pairs,
    num_steps,
    seed=0,
    high_quantile=0.8,
    low_quantile=0.6,
    neighbor_pool=16,
):
    x_train = np.asarray(x_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32).reshape(-1)
    rng = np.random.default_rng(seed)

    high_threshold = np.quantile(y_train, high_quantile)
    low_threshold = np.quantile(y_train, low_quantile)
    high_indices = np.where(y_train >= high_threshold)[0]
    low_indices = np.where(y_train <= low_threshold)[0]
    if high_indices.size == 0:
        high_indices = np.argsort(y_train)[-max(1, min(num_pairs, y_train.shape[0])):]
    if low_indices.size == 0:
        low_indices = np.argsort(y_train)[:max(1, min(num_pairs, y_train.shape[0]))]

    selected_low = rng.choice(low_indices, size=num_pairs, replace=low_indices.size < num_pairs)
    x_low = x_train[selected_low]
    y_low = y_train[selected_low]
    high_x = x_train[high_indices]
    high_y = y_train[high_indices]

    chosen_high = []
    uncertainty = []
    for low_point, low_score in zip(x_low, y_low):
        distances = np.linalg.norm(high_x - low_point[None, :], axis=1)
        valid = np.where(high_y > low_score)[0]
        if valid.size == 0:
            valid = np.arange(high_indices.size)
        valid_distances = distances[valid]
        pool_size = min(max(1, neighbor_pool), valid.size)
        nearest_local = np.argpartition(valid_distances, kth=pool_size - 1)[:pool_size]
        pool = valid[nearest_local]
        chosen = pool[rng.integers(pool.shape[0])]
        chosen_high.append(high_indices[chosen])
        uncertainty.append(float(np.std(high_y[pool])))

    chosen_high = np.asarray(chosen_high, dtype=np.int64)
    x_high = x_train[chosen_high]
    y_high = y_train[chosen_high]
    improvement = y_high - y_low
    monotonicity = (improvement > 0.0).astype(np.float32)
    manifold_distance = np.linalg.norm(x_high - x_low, axis=1).astype(np.float32)
    uncertainty = np.asarray(uncertainty, dtype=np.float32)

    alphas = np.linspace(0.0, 1.0, num_steps + 1, dtype=np.float32).reshape(1, -1, 1)
    trajectories = (1.0 - alphas) * x_low[:, None, :] + alphas * x_high[:, None, :]
    weights = compute_quality_weights(
        improvement=improvement,
        monotonicity=monotonicity,
        uncertainty=uncertainty,
        manifold_distance=manifold_distance,
    )

    return trajectories.astype(np.float32), {
        "x_low": x_low.astype(np.float32),
        "x_high": x_high.astype(np.float32),
        "y_low": y_low.astype(np.float32),
        "y_high": y_high.astype(np.float32),
        "improvement": improvement.astype(np.float32),
        "monotonicity": monotonicity.astype(np.float32),
        "uncertainty": uncertainty.astype(np.float32),
        "manifold_distance": manifold_distance.astype(np.float32),
        "weights": weights.astype(np.float32),
    }

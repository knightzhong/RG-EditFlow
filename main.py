# main.py
import argparse
import json
import design_bench
import torch
import torch.optim as optim
import numpy as np

from src.config import Config
from src.utils import set_seed, Normalizer
from src.oracle import NTKOracle
from src.generator import GP, sampling_data_from_GP, generate_trajectories_from_GP_samples, select_gp_fit_tensors
from src.models import VectorFieldNet
from src.flow import train_cfm_step,train_cfm, inference_ode, inference_ode_multi
import time
import os
from src.trajectory_sources import generate_knn_mixup_trajectories
from src.evaluation import normalized_score_summary, normalize_oracle_scores, offline_scores_for_indices, sample_proposal_diagnostic_indices
from src.checkpointing import should_save_epoch_checkpoint
from src.compat import patch_numpy_legacy_pickle_aliases
from src.quality import (
    quality_from_gp_samples,
    knn_proxy_scores,
    bootstrap_knn_proxy_scores,
    conservative_rerank_scores,
    trust_region_rerank_scores,
    clip_proposals_to_trust_region,
    select_top_indices,
    select_best_per_seed_indices,
    select_best_per_seed_with_fallback,
    filter_trajectories_by_quality,
)


def parse_args():
    parser = argparse.ArgumentParser(description="GGFM a407d81 GP-supervised Flow Matching")
    parser.add_argument("--task-name", default=None, help="Override Config.TASK_NAME")
    parser.add_argument("--device", default=None, help="Override Config.DEVICE")
    parser.add_argument("--seed", type=int, default=None, help="Override Config.SEED")
    parser.add_argument("--fm-epochs", type=int, default=None, help="Override Config.FM_EPOCHS")
    parser.add_argument("--gp-num-functions", type=int, default=None, help="Override Config.GP_NUM_FUNCTIONS")
    parser.add_argument("--gp-num-points", type=int, default=None, help="Override Config.GP_NUM_POINTS")
    parser.add_argument("--gp-gradient-steps", type=int, default=None, help="Override Config.GP_NUM_GRADIENT_STEPS")
    parser.add_argument("--gp-traj-steps", type=int, default=None, help="Override Config.GP_TRAJ_STEPS")
    parser.add_argument("--gp-num-fit-samples", type=int, default=None, help="Override Config.GP_NUM_FIT_SAMPLES")
    parser.add_argument("--num-test-samples", type=int, default=None, help="Override Config.NUM_TEST_SAMPLES")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Directory for model checkpoints")
    parser.add_argument("--save-every", type=int, default=10, help="Save epoch checkpoints every N epochs; 0 saves only final checkpoint")
    parser.add_argument("--metrics-path", default=None, help="Optional JSON path for final metrics")
    parser.add_argument("--proposal-diagnostics-path", default=None, help="Optional JSON path for proposal-level rerank diagnostics")
    parser.add_argument("--proposal-diagnostics-max-raw", type=int, default=0, help="When >0, save all selected proposals plus at most this many unselected raw proposals")
    parser.add_argument("--load-checkpoint", default=None, help="Load a trained CFM checkpoint before evaluation")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and only run evaluation")
    parser.add_argument("--use-quality-gating", action="store_true", help="Weight CFM training by trajectory quality")
    parser.add_argument("--quality-gate-mode", choices=["none", "score", "geometry", "full"], default="full", help="Reliability signals used for trajectory weights")
    parser.add_argument("--quality-keep-ratio", type=float, default=1.0, help="Keep only the top-ratio trajectories by quality weight during training")
    parser.add_argument("--trajectory-source", choices=["gp", "knn-mixup"], default="gp", help="Trajectory source for training")
    parser.add_argument("--endpoint-only", action="store_true", help="Use one segment per trajectory as endpoint-only ablation")
    parser.add_argument("--gp-record-paths", action="store_true", help="Use recorded GP descent/ascent paths instead of endpoint interpolation")
    parser.add_argument("--path-supervision-mode", choices=["recorded", "endpoint"], default="recorded", help="How recorded GP paths supervise CFM training")
    parser.add_argument("--endpoint-mix", type=float, default=0.0, help="Mix endpoint bridge velocity into path velocity during CFM training")
    parser.add_argument("--mixup-high-quantile", type=float, default=0.8, help="High-score quantile for kNN/mixup trajectories")
    parser.add_argument("--mixup-low-quantile", type=float, default=0.6, help="Low-score quantile for kNN/mixup trajectories")
    parser.add_argument("--mixup-neighbor-pool", type=int, default=16, help="Nearest high-score pool for kNN/mixup trajectories")
    parser.add_argument("--num-proposals", type=int, default=1, help="Number of ODE proposals per evaluation seed")
    parser.add_argument("--proposal-noise-scale", type=float, default=0.0, help="Gaussian noise added to proposal starts")
    parser.add_argument("--proposal-max-displacement", type=float, default=0.0, help="Clip each proposal to this max normalized distance from its seed")
    parser.add_argument("--rerank-k", type=int, default=5, help="kNN count for offline proxy reranking")
    parser.add_argument("--rerank-uncertainty-weight", type=float, default=0.25, help="Penalty for kNN score uncertainty")
    parser.add_argument("--rerank-distance-weight", type=float, default=0.1, help="Penalty for distance to offline data")
    parser.add_argument("--rerank-displacement-weight", type=float, default=0.0, help="Penalty for proposal displacement from its seed")
    parser.add_argument("--rerank-max-distance-quantile", type=float, default=1.0, help="Drop proposals above this manifold-distance quantile before reranking")
    parser.add_argument("--fallback-to-seed", action="store_true", help="Use original seed if all proposals score worse than the seed")
    parser.add_argument("--rerank-mode", choices=["per-seed", "global"], default="per-seed", help="Select best proposal per seed before final top-k, or global top-k")
    parser.add_argument("--uncertainty-mode", choices=["label-variance", "bootstrap-knn"], default="label-variance", help="Uncertainty estimator for conservative rerank")
    parser.add_argument("--uncertainty-bootstrap", type=int, default=16, help="Number of bootstraps for bootstrap-knn uncertainty")
    return parser.parse_args()


def apply_cli_overrides(args):
    overrides = {
        "TASK_NAME": args.task_name,
        "DEVICE": args.device,
        "SEED": args.seed,
        "FM_EPOCHS": args.fm_epochs,
        "GP_NUM_FUNCTIONS": args.gp_num_functions,
        "GP_NUM_POINTS": args.gp_num_points,
        "GP_NUM_GRADIENT_STEPS": args.gp_gradient_steps,
        "GP_TRAJ_STEPS": args.gp_traj_steps,
        "GP_NUM_FIT_SAMPLES": args.gp_num_fit_samples,
        "NUM_TEST_SAMPLES": args.num_test_samples,
    }
    for name, value in overrides.items():
        if value is not None:
            setattr(Config, name, value)

def get_design_bench_data(task_name):
    """
    加载并标准化 Design-Bench 数据，支持离散任务转换
    完全对齐 ROOT 的处理方式
    """
    print(f"Loading task: {task_name}...")
    patch_numpy_legacy_pickle_aliases()
    if task_name != 'TFBind10-Exact-v0':
        task = design_bench.make(task_name)
    else:
        # 显存优化（与 ROOT 一致）
        task = design_bench.make(task_name, dataset_kwargs={"max_samples": 10000})
    
    offline_x = task.x
    logits_shape = None  # 保存 logits 形状信息
    
    if task.is_discrete:
        # ROOT 风格：使用 map_to_logits 修改 task 内部状态
        # 这样 task.predict() 才能正确处理 logits 格式的数据
        task.map_to_logits()
        offline_x = task.x  # 现在 task.x 已经是 logits 格式 (N, L, V-1)
        logits_shape = offline_x.shape  # 保存形状 (N, L, V-1)
        offline_x = offline_x.reshape(offline_x.shape[0], -1)  # 展平为 (N, L*(V-1))
        print(f"[数据编码] 离散任务：已调用 map_to_logits，Logits {logits_shape} -> 展平 {offline_x.shape}")
    else:
        print("[数据编码] 连续任务：直接使用原始数据")
    
    # 计算统计量（与 ROOT 完全一致）
    mean_x = np.mean(offline_x, axis=0)
    std_x = np.std(offline_x, axis=0)
    std_x = np.where(std_x == 0, 1.0, std_x)  # ROOT 使用 == 0，不是 < 1e-6
    offline_x_norm = (offline_x - mean_x) / std_x
    
    # 处理 Y（与 ROOT 一致）
    offline_y = task.y.reshape(-1)  # ROOT 使用 reshape(-1)，不是 reshape(-1, 1)
    mean_y = np.mean(offline_y, axis=0)
    std_y = np.std(offline_y, axis=0)
    
    # 洗牌数据（与 ROOT 一致）
    shuffle_idx = np.random.permutation(offline_x.shape[0])
    offline_x_norm = offline_x_norm[shuffle_idx]
    offline_y = offline_y[shuffle_idx]
    
    # 标准化 Y
    offline_y_norm = (offline_y - mean_y) / std_y
    
    return task, offline_x_norm, offline_y_norm, mean_x, std_x, mean_y, std_y, logits_shape

# main.py 预处理逻辑
# def preprocess_trajectories(oracle, X_train_norm):
#     device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")
#     if os.path.exists(Config.TRAJECTORY_PATH):
#         print(f"Loading cached trajectories from {Config.TRAJECTORY_PATH}")
#         return np.load(Config.TRAJECTORY_PATH)['trajs']

#     print("=== Generating ALL long trajectories (GD reverse + GA) ===")
#     all_indices = np.arange(len(X_train_norm))
#     # 按照你的要求：用全量数据，或者至少 10000 条
#     sample_size = len(all_indices)
#     selected_idx = all_indices
#     # selected_idx = np.random.choice(all_indices, sample_size, replace=False)
    
#     all_valid = []
#     batch_size = 256
#     for i in range(0, sample_size, batch_size):
#         batch_x = X_train_norm[selected_idx[i : i + batch_size]]
#         trajs = generate_long_trajectories(oracle, batch_x, device)
#         all_valid.append(trajs)
#         print(f"Progress: {i + batch_size}/{sample_size}")
        
#     pool = np.concatenate(all_valid, axis=0)
#     np.savez_compressed(Config.TRAJECTORY_PATH, trajs=pool)
#     return pool

def main():
    args = parse_args()
    apply_cli_overrides(args)
    run_start = time.time()

    # 0. 初始化环境
    print(f"=== GGFM Trajectory Flow: {Config.TASK_NAME} ===")
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")
    
    # 1. 加载并编码数据（完全对齐 ROOT 的处理方式）
    task, X_train_norm, y_train_norm, mean_x, std_x, mean_y, std_y, logits_shape = get_design_bench_data(Config.TASK_NAME)
    y_train_raw = y_train_norm * std_y + mean_y
    
    # 同步 Normalizer 状态
    x_normalizer = Normalizer(np.zeros((1, X_train_norm.shape[1])))
    x_normalizer.mean, x_normalizer.std, x_normalizer.device = mean_x, std_x, device
    
    # 2. 转换为 Tensor 供 GP 使用（与 ROOT 一致）
    X_train_tensor = torch.FloatTensor(X_train_norm).to(device)
    # y_train_norm 现在是 (N,) 形状，需要转换为 (N, 1) 供 Oracle 使用
    y_train_tensor = torch.FloatTensor(y_train_norm).reshape(-1, 1).to(device)
    
    # 保存原始统计量（用于反标准化）
    mean_x_torch = torch.FloatTensor(mean_x).to(device)
    std_x_torch = torch.FloatTensor(std_x).to(device)
    mean_y_torch = torch.FloatTensor([mean_y]).to(device)
    std_y_torch = torch.FloatTensor([std_y]).to(device)
    
    # 3. 初始化 GP 超参数
    lengthscale = torch.tensor(Config.GP_INITIAL_LENGTHSCALE, device=device)
    variance = torch.tensor(Config.GP_INITIAL_OUTPUTSCALE, device=device)
    noise = torch.tensor(Config.GP_NOISE, device=device)
    mean_prior = torch.tensor(0.0, device=device)
    
    # 4. 选择用于 GP 拟合的初始点（完全对齐 ROOT）
    if Config.GP_TYPE_INITIAL_POINTS == 'highest':
        # ROOT: 固定 1024 个样本，每次全选但顺序不同
        best_indices = torch.argsort(y_train_tensor.view(-1))[-1024:]
        best_x = X_train_tensor[best_indices]
        print(f"[GP Init] Using top 1024 samples for GP sampling (ROOT style: same samples, different order each epoch)")
    elif Config.GP_TYPE_INITIAL_POINTS == 'lowest':
        best_indices = torch.argsort(y_train_tensor.view(-1))[:1024]
        best_x = X_train_tensor[best_indices]
        print(f"[GP Init] Using bottom 1024 samples for GP sampling")
    else:
        best_x = X_train_tensor
        print(f"[GP Init] Using all samples for GP sampling")
    
    # 5. 初始化 Flow Matching 网络
    input_dim = X_train_norm.shape[1]
    cfm_model = VectorFieldNet(input_dim, hidden_dim=Config.HIDDEN_DIM).to(device)
    optimizer = optim.Adam(cfm_model.parameters(), lr=Config.FM_LR)

    if args.load_checkpoint:
        checkpoint = torch.load(args.load_checkpoint, map_location=device)
        cfm_model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint and not args.eval_only:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"[Checkpoint] Loaded {args.load_checkpoint}")

    # --- 核心修改：每个 Epoch 动态采样 GP 函数生成轨迹 ---
    print(f"=== Training: {args.trajectory_source} trajectories ({Config.FM_EPOCHS} Epochs) ===")
    print(f"每个 Epoch 采样 {Config.GP_NUM_FUNCTIONS} × {Config.GP_NUM_POINTS} 条候选轨迹")

    # y_train_norm 已经是 (N,) 形状了，不需要 flatten
    y_scores_flat = y_train_norm

    quality_stats = []

    if args.eval_only:
        print("[Eval Only] Skipping training loop")
    for epoch in range(0 if args.eval_only else Config.FM_EPOCHS):
        # 每个 Epoch 重新采样具有不同超参数的 GP
        epoch_start = time.time()  # 记录 epoch 开始时间
        print(f"\n=== Epoch {epoch+1}/{Config.FM_EPOCHS} ===")
        
        gp_init_time = 0.0
        sampling_start = time.time()
        num_traj_steps = 1 if args.endpoint_only else Config.GP_TRAJ_STEPS
        trajectory_weights = None

        if args.trajectory_source == "gp":
            gp_init_start = time.time()
            gp_fit_x, gp_fit_y = select_gp_fit_tensors(
                X_train_tensor,
                y_train_tensor.view(-1),
                max_samples=Config.GP_NUM_FIT_SAMPLES,
            )
            if epoch == 0:
                print(f"[GP Fit] Using {gp_fit_x.shape[0]} / {X_train_tensor.shape[0]} samples for GP posterior")
            GP_Model = GP(
                device=device,
                x_train=gp_fit_x,
                y_train=gp_fit_y,
                lengthscale=lengthscale,
                variance=variance,
                noise=noise,
                mean_prior=mean_prior
            )
            gp_init_time = time.time() - gp_init_start
            data_from_GP = sampling_data_from_GP(
                x_train=best_x,
                device=device,
                GP_Model=GP_Model,
                num_functions=Config.GP_NUM_FUNCTIONS,
                num_gradient_steps=Config.GP_NUM_GRADIENT_STEPS,
                num_points=Config.GP_NUM_POINTS,
                learning_rate=Config.GP_LEARNING_RATE,
                delta_lengthscale=Config.GP_DELTA_LENGTHSCALE,
                delta_variance=Config.GP_DELTA_VARIANCE,
                seed=epoch,
                threshold_diff=Config.GP_THRESHOLD_DIFF,
                verbose=(epoch == 0),
                record_paths=args.gp_record_paths,
            )
            quality = quality_from_gp_samples(data_from_GP, mode=args.quality_gate_mode) if args.use_quality_gating else None
            traj_gen_start = time.time()
            trajs_array = generate_trajectories_from_GP_samples(
                data_from_GP,
                device=device,
                num_steps=num_traj_steps,
                path_mode=args.path_supervision_mode if args.gp_record_paths else "endpoint",
            )
            traj_gen_time = time.time() - traj_gen_start
        else:
            traj_gen_start = time.time()
            trajs_array, quality = generate_knn_mixup_trajectories(
                X_train_norm,
                y_train_norm,
                num_pairs=Config.GP_NUM_FUNCTIONS * Config.GP_NUM_POINTS,
                num_steps=num_traj_steps,
                seed=epoch,
                high_quantile=args.mixup_high_quantile,
                low_quantile=args.mixup_low_quantile,
                neighbor_pool=args.mixup_neighbor_pool,
            )
            traj_gen_time = time.time() - traj_gen_start
        sampling_time = time.time() - sampling_start

        if args.use_quality_gating and quality is not None:
            before_filter = len(trajs_array)
            trajs_array, quality = filter_trajectories_by_quality(
                trajs_array,
                quality,
                keep_ratio=args.quality_keep_ratio,
            )
            trajectory_weights = quality["weights"]
            if before_filter != len(trajs_array):
                print(f"  [Quality] kept {len(trajs_array)} / {before_filter} trajectories")
            if trajectory_weights.shape[0] > 0:
                quality_stats.append({
                    "weight_mean": float(np.mean(trajectory_weights)),
                    "improvement_mean": float(np.mean(quality["improvement"])),
                    "monotonicity_mean": float(np.mean(quality["monotonicity"])),
                    "uncertainty_mean": float(np.mean(quality["uncertainty"])),
                    "manifold_distance_mean": float(np.mean(quality["manifold_distance"])),
                })
                print(
                    f"  [Quality] weight={np.mean(trajectory_weights):.4f} | "
                    f"Δy={np.mean(quality['improvement']):.4f} | "
                    f"dist={np.mean(quality['manifold_distance']):.4f}"
                )
        
        if len(trajs_array) == 0:
            print(f"Warning: No valid trajectories generated in epoch {epoch+1}")
            continue
        
        print(f"Generated {len(trajs_array)} candidate trajectories")
        print(f"  [⏱️ Time] GP初始化: {gp_init_time:.2f}s | GP采样: {sampling_time:.2f}s | 轨迹生成: {traj_gen_time:.2f}s")
        
        # 对这批轨迹进行流匹配训练更新
        train_start = time.time()
        avg_loss = train_cfm_step(cfm_model, trajs_array, optimizer, device, trajectory_weights=trajectory_weights, endpoint_mix=args.endpoint_mix)
        train_time = time.time() - train_start
        
        epoch_total_time = time.time() - epoch_start
        
        print(f"  [⏱️ Time] 训练: {train_time:.2f}s | Epoch总计: {epoch_total_time:.2f}s")
        
        if should_save_epoch_checkpoint(epoch + 1, args.save_every):
            print(f"Epoch {epoch+1}/{Config.FM_EPOCHS} | Loss: {avg_loss:.4f} | Trajs: {len(trajs_array)}")
            # 保存检查点
            checkpoint_path = os.path.join(args.checkpoint_dir, f"cfm_model_epoch_{epoch+1}.pt")
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': cfm_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"  [💾 Checkpoint] Saved to {checkpoint_path}")
    
    # 保存最终模型
    if not args.eval_only:
        final_model_path = os.path.join(args.checkpoint_dir, "cfm_model_final.pt")
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        torch.save({
            'epoch': Config.FM_EPOCHS,
            'model_state_dict': cfm_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'input_dim': input_dim,
            'hidden_dim': Config.HIDDEN_DIM,
        }, final_model_path)
        print(f"\n[💾 Final Model] Saved to {final_model_path}")

    # 4. 推理与 SOTA 评估 (Q=128)（完全对齐 ROOT 的测试逻辑）
    print(f"\n=== SOTA Evaluation (Highest-point, Q={Config.NUM_TEST_SAMPLES}) ===")
    
    # 对齐 ROOT：从得分最高的 128 个标准化样本出发
    test_q = Config.NUM_TEST_SAMPLES
    
    # 使用标准化后的 y 来选择高分样本（与 ROOT 一致）
    # y_train_norm 是 numpy array，所以这里的索引都是 numpy
    sorted_indices = np.argsort(y_train_norm)
    high_indices = sorted_indices[-test_q:]
    
    # 获取标准化的高分样本作为起点
    X_test_norm = X_train_norm[high_indices]
    y_test_start = y_train_norm[high_indices]
    
    print(f"Selected {test_q} highest samples as starting points")
    print(f"Starting scores (normalized): mean={np.mean(y_test_start):.4f}, max={np.max(y_test_start):.4f}")
    
    # ODE 推理（添加 y 条件和 CFG）
    # 与 ROOT 完全对齐：使用 Oracle 理论最大值而非数据集分位数！
    
    rerank_info = None
    proposal_diagnostics = None
    if args.num_proposals > 1 or args.proposal_noise_scale > 0:
        proposals_norm, seed_ids = inference_ode_multi(
            cfm_model,
            X_test_norm,
            device,
            num_samples=args.num_proposals,
            noise_scale=args.proposal_noise_scale,
        )
        raw_seed_displacement = np.linalg.norm(proposals_norm - X_test_norm[seed_ids], axis=1)
        proposals_norm = clip_proposals_to_trust_region(
            proposals_norm,
            seed_ids,
            X_test_norm,
            max_displacement=args.proposal_max_displacement,
        )
        clipped_seed_displacement = np.linalg.norm(proposals_norm - X_test_norm[seed_ids], axis=1)
        if args.uncertainty_mode == "bootstrap-knn":
            proxy = bootstrap_knn_proxy_scores(
                proposals_norm,
                X_train_norm,
                y_train_norm,
                k=args.rerank_k,
                num_bootstrap=args.uncertainty_bootstrap,
                seed=Config.SEED,
            )
        else:
            proxy = knn_proxy_scores(
                proposals_norm,
                X_train_norm,
                y_train_norm,
                k=args.rerank_k,
            )
        seed_displacement = clipped_seed_displacement
        rerank_scores = trust_region_rerank_scores(
            proxy["predicted_score"],
            proxy["uncertainty"],
            proxy["manifold_distance"],
            seed_displacement,
            uncertainty_weight=args.rerank_uncertainty_weight,
            distance_weight=args.rerank_distance_weight,
            displacement_weight=args.rerank_displacement_weight,
        )
        max_distance_threshold = None
        if args.rerank_max_distance_quantile < 1.0:
            quantile = float(np.clip(args.rerank_max_distance_quantile, 0.0, 1.0))
            max_distance_threshold = float(np.quantile(proxy["manifold_distance"], quantile))
            rerank_scores = rerank_scores.copy()
            rerank_scores[proxy["manifold_distance"] > max_distance_threshold] = -np.inf

        fallback_mask = None
        seed_scores = None
        if args.fallback_to_seed:
            seed_proxy = knn_proxy_scores(
                X_test_norm,
                X_train_norm,
                y_train_norm,
                k=args.rerank_k,
            )
            seed_scores = conservative_rerank_scores(
                seed_proxy["predicted_score"],
                seed_proxy["uncertainty"],
                seed_proxy["manifold_distance"],
                uncertainty_weight=args.rerank_uncertainty_weight,
                distance_weight=args.rerank_distance_weight,
            )

        if args.rerank_mode == "per-seed":
            if args.fallback_to_seed:
                opt_X_norm, fallback_mask = select_best_per_seed_with_fallback(
                    proposals_norm,
                    rerank_scores,
                    seed_ids,
                    X_test_norm,
                    min_score=seed_scores,
                )
                selected_indices = select_best_per_seed_indices(rerank_scores, seed_ids, test_q)
            else:
                selected_indices = select_best_per_seed_indices(rerank_scores, seed_ids, test_q)
                opt_X_norm = proposals_norm[selected_indices]
        else:
            selected_indices = select_top_indices(rerank_scores, test_q)
            opt_X_norm = proposals_norm[selected_indices]
        rerank_info = {
            "num_raw_proposals": int(proposals_norm.shape[0]),
            "num_selected": int(selected_indices.shape[0]),
            "proxy_mean": float(np.mean(proxy["predicted_score"])),
            "selected_proxy_mean": float(np.mean(proxy["predicted_score"][selected_indices])),
            "selected_rerank_mean": float(np.mean(rerank_scores[selected_indices])),
            "selected_distance_mean": float(np.mean(proxy["manifold_distance"][selected_indices])),
            "selected_displacement_mean": float(np.mean(seed_displacement[selected_indices])),
            "raw_displacement_mean": float(np.mean(raw_seed_displacement)),
            "proposal_max_displacement": float(args.proposal_max_displacement),
            "max_distance_threshold": max_distance_threshold,
            "fallback_count": int(np.sum(fallback_mask)) if fallback_mask is not None else 0,
            "unique_seed_count": int(len(np.unique(seed_ids[selected_indices]))),
            "rerank_mode": args.rerank_mode,
            "uncertainty_mode": args.uncertainty_mode,
            "rerank_displacement_weight": float(args.rerank_displacement_weight),
            "rerank_max_distance_quantile": float(args.rerank_max_distance_quantile),
            "fallback_to_seed": bool(args.fallback_to_seed),
        }
        proposal_diagnostics = {
            "seed_ids": seed_ids.astype(np.int64),
            "selected_indices": selected_indices.astype(np.int64),
            "predicted_score": proxy["predicted_score"],
            "uncertainty": proxy["uncertainty"],
            "manifold_distance": proxy["manifold_distance"],
            "rerank_score": rerank_scores,
            "raw_seed_displacement": raw_seed_displacement,
            "clipped_seed_displacement": clipped_seed_displacement,
            "max_distance_threshold": max_distance_threshold,
        }
        print(
            f"[Rerank] selected {selected_indices.shape[0]} / {proposals_norm.shape[0]} proposals | "
            f"proxy={rerank_info['selected_proxy_mean']:.4f} | "
            f"dist={rerank_info['selected_distance_mean']:.4f} | "
            f"disp={rerank_info['selected_displacement_mean']:.4f} | "
            f"fallback={rerank_info['fallback_count']} | "
            f"unique_seeds={rerank_info['unique_seed_count']}"
        )
    else:
        opt_X_norm = inference_ode(cfm_model, X_test_norm, device)
    
    # 反标准化（与 ROOT 一致）
    opt_X_denorm = opt_X_norm * std_x + mean_x
    
    # 还原形状供 task.predict 打分（与 ROOT 一致）
    if task.is_discrete and logits_shape is not None:
        # 离散任务：需要 reshape 回 (N, L, V-1) 的形状
        # 使用数据加载时保存的 logits_shape 信息
        opt_X_for_predict = opt_X_denorm.reshape(test_q, logits_shape[1], logits_shape[2])
        # 原始样本也需要相同处理
        original_X_denorm = X_test_norm * std_x + mean_x
        original_X_for_predict = original_X_denorm.reshape(test_q, logits_shape[1], logits_shape[2])
        
        print(f"[Discrete Task] Reshaped to Logits format: {opt_X_for_predict.shape}")
    else:
        # 连续任务：直接使用
        opt_X_for_predict = opt_X_denorm
        original_X_for_predict = X_test_norm * std_x + mean_x
    
    # 使用 Oracle 评估（与 ROOT 一致，直接传入 numpy array）
    final_scores = task.predict(opt_X_for_predict).flatten()
    original_scores = offline_scores_for_indices(y_train_raw, high_indices)
    
    # 计算标准化分数（与 ROOT 一致）
    # oracle_y_min, oracle_y_max = np.min(task.y), np.max(task.y)
    # final_score_norm = (final_scores - oracle_y_min) / (oracle_y_max - oracle_y_min)
    
    # 计算百分位数（与 ROOT 一致）
    final_scores_sorted = np.sort(final_scores)
    print(f"\n[Result] Final scores distribution:")
    print(f"  Min: {final_scores_sorted[0]:.4f}")
    print(f"  Max: {final_scores_sorted[-1]:.4f}")
    print(f"  Mean: {np.mean(final_scores):.4f}")
    print(f"  Std: {np.std(final_scores):.4f}")
    
    # 使用 torch.quantile 计算百分位数（与 ROOT 一致）
    final_scores_tensor = torch.from_numpy(final_scores)
    percentiles = torch.quantile(final_scores_tensor, torch.tensor([1.0, 0.8, 0.5]), interpolation='higher')
    p100_score = percentiles[0].item()
    p80_score = percentiles[1].item()
    p50_score = percentiles[2].item()
    normalized_metrics = normalized_score_summary(final_scores, Config.TASK_NAME)
    original_normalized_metrics = normalized_score_summary(original_scores, Config.TASK_NAME)
    
    print("-" * 60)
    print(f"Original Mean (Top {test_q}): {np.mean(original_scores):.4f}")
    print(f"Optimized Mean (Top {test_q}): {np.mean(final_scores):.4f}")
    print(f"Improvement:                   {np.mean(final_scores) - np.mean(original_scores):.4f}")
    print("-" * 60)
    print(f"100th Percentile (Max):      {p100_score:.4f}")
    print(f"80th Percentile:             {p80_score:.4f}")
    print(f"50th Percentile (Median):    {p50_score:.4f}")
    print("-" * 60)
    print("[Normalized by Design-Bench oracle min/max]")
    print(f"Normalized Mean:             {normalized_metrics['normalized_mean']:.4f}")
    print(f"Normalized 100th:            {normalized_metrics['normalized_p100']:.4f}")
    print(f"Normalized 80th:             {normalized_metrics['normalized_p80']:.4f}")
    print(f"Normalized 50th:             {normalized_metrics['normalized_p50']:.4f}")
    print("-" * 60)

    if args.proposal_diagnostics_path and proposal_diagnostics is not None:
        diagnostic_indices = sample_proposal_diagnostic_indices(
            proposals_norm.shape[0],
            proposal_diagnostics["selected_indices"],
            max_raw=args.proposal_diagnostics_max_raw,
            seed=Config.SEED,
        )
        diagnostic_proposals_norm = proposals_norm[diagnostic_indices]
        proposal_denorm = diagnostic_proposals_norm * std_x + mean_x
        if task.is_discrete and logits_shape is not None:
            proposal_for_predict = proposal_denorm.reshape(diagnostic_proposals_norm.shape[0], logits_shape[1], logits_shape[2])
        else:
            proposal_for_predict = proposal_denorm
        proposal_scores = task.predict(proposal_for_predict).flatten()
        proposal_normalized_scores = normalize_oracle_scores(proposal_scores, Config.TASK_NAME)
        diagnostic_seed_ids = proposal_diagnostics["seed_ids"][diagnostic_indices]
        seed_original_scores = original_scores[diagnostic_seed_ids]
        seed_normalized_scores = normalize_oracle_scores(seed_original_scores, Config.TASK_NAME)
        selected_mask = np.isin(diagnostic_indices, proposal_diagnostics["selected_indices"])
        diagnostics_payload = {
            "commit": os.popen("git rev-parse HEAD").read().strip(),
            "task_name": Config.TASK_NAME,
            "seed": Config.SEED,
            "num_test_samples": int(test_q),
            "num_proposals": int(args.num_proposals),
            "num_raw_proposals": int(proposals_norm.shape[0]),
            "num_diagnostic_proposals": int(diagnostic_indices.shape[0]),
            "proposal_diagnostics_max_raw": int(args.proposal_diagnostics_max_raw),
            "proposal_noise_scale": float(args.proposal_noise_scale),
            "proposal_max_displacement": float(args.proposal_max_displacement),
            "rerank_uncertainty_weight": float(args.rerank_uncertainty_weight),
            "rerank_distance_weight": float(args.rerank_distance_weight),
            "rerank_displacement_weight": float(args.rerank_displacement_weight),
            "rerank_max_distance_quantile": float(args.rerank_max_distance_quantile),
            "rerank_mode": args.rerank_mode,
            "uncertainty_mode": args.uncertainty_mode,
            "max_distance_threshold": proposal_diagnostics["max_distance_threshold"],
            "proposal_indices": diagnostic_indices.astype(int).tolist(),
            "seed_ids": diagnostic_seed_ids.astype(int).tolist(),
            "selected_indices": proposal_diagnostics["selected_indices"].astype(int).tolist(),
            "selected_mask": selected_mask.astype(bool).tolist(),
            "predicted_score": proposal_diagnostics["predicted_score"][diagnostic_indices].astype(float).tolist(),
            "uncertainty": proposal_diagnostics["uncertainty"][diagnostic_indices].astype(float).tolist(),
            "manifold_distance": proposal_diagnostics["manifold_distance"][diagnostic_indices].astype(float).tolist(),
            "rerank_score": proposal_diagnostics["rerank_score"][diagnostic_indices].astype(float).tolist(),
            "raw_seed_displacement": proposal_diagnostics["raw_seed_displacement"][diagnostic_indices].astype(float).tolist(),
            "clipped_seed_displacement": proposal_diagnostics["clipped_seed_displacement"][diagnostic_indices].astype(float).tolist(),
            "proposal_oracle_score": proposal_scores.astype(float).tolist(),
            "normalized_proposal_oracle_score": proposal_normalized_scores.astype(float).tolist(),
            "seed_oracle_score": seed_original_scores.astype(float).tolist(),
            "normalized_seed_oracle_score": seed_normalized_scores.astype(float).tolist(),
            "oracle_improvement": (proposal_scores - seed_original_scores).astype(float).tolist(),
            "normalized_oracle_improvement": (proposal_normalized_scores - seed_normalized_scores).astype(float).tolist(),
        }
        os.makedirs(os.path.dirname(args.proposal_diagnostics_path) or ".", exist_ok=True)
        with open(args.proposal_diagnostics_path, "w", encoding="utf-8") as f:
            json.dump(diagnostics_payload, f, indent=2, ensure_ascii=False)
        print(f"[Proposal Diagnostics] Saved to {args.proposal_diagnostics_path}")

    if args.metrics_path:
        metrics = {
            "commit": os.popen("git rev-parse HEAD").read().strip(),
            "task_name": Config.TASK_NAME,
            "seed": Config.SEED,
            "device": str(device),
            "fm_epochs": Config.FM_EPOCHS,
            "gp_num_functions": Config.GP_NUM_FUNCTIONS,
            "gp_num_points": Config.GP_NUM_POINTS,
            "gp_gradient_steps": Config.GP_NUM_GRADIENT_STEPS,
            "gp_traj_steps": Config.GP_TRAJ_STEPS,
            "num_test_samples": test_q,
            "final_min": float(final_scores_sorted[0]),
            "final_max": float(final_scores_sorted[-1]),
            "final_mean": float(np.mean(final_scores)),
            "final_std": float(np.std(final_scores)),
            "original_mean": float(np.mean(original_scores)),
            "improvement": float(np.mean(final_scores) - np.mean(original_scores)),
            "p100": float(p100_score),
            "p80": float(p80_score),
            "p50": float(p50_score),
            "normalized_final_mean": normalized_metrics["normalized_mean"],
            "normalized_final_std": normalized_metrics["normalized_std"],
            "normalized_p100": normalized_metrics["normalized_p100"],
            "normalized_p80": normalized_metrics["normalized_p80"],
            "normalized_p50": normalized_metrics["normalized_p50"],
            "normalized_original_mean": original_normalized_metrics["normalized_mean"],
            "normalized_improvement": normalized_metrics["normalized_mean"] - original_normalized_metrics["normalized_mean"],
            "run_seconds": float(time.time() - run_start),
            "checkpoint_dir": args.checkpoint_dir,
            "trajectory_source": args.trajectory_source,
            "endpoint_only": bool(args.endpoint_only),
            "endpoint_mix": float(args.endpoint_mix),
            "gp_record_paths": bool(args.gp_record_paths),
            "path_supervision_mode": args.path_supervision_mode,
            "use_quality_gating": bool(args.use_quality_gating),
            "quality_gate_mode": args.quality_gate_mode,
            "quality_keep_ratio": float(args.quality_keep_ratio),
            "num_proposals": int(args.num_proposals),
            "proposal_noise_scale": float(args.proposal_noise_scale),
            "rerank_mode": args.rerank_mode,
            "uncertainty_mode": args.uncertainty_mode,
            "quality_stats": quality_stats,
            "rerank_info": rerank_info,
            "proposal_diagnostics_path": args.proposal_diagnostics_path,
            "proposal_diagnostics_max_raw": int(args.proposal_diagnostics_max_raw),
        }
        os.makedirs(os.path.dirname(args.metrics_path) or ".", exist_ok=True)
        with open(args.metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"[Metrics] Saved to {args.metrics_path}")

if __name__ == "__main__":
    main()

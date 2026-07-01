# src/flow.py
import torch
import numpy as np
from src.config import Config


def build_velocity_targets(trajs, k, alpha, idx_range, endpoint_mix=0.0):
    """Build local path velocities with optional endpoint-bridge anchoring."""
    num_segments = trajs.shape[1] - 1
    x_k = trajs[idx_range, k, :]
    x_k_next = trajs[idx_range, k + 1, :]
    path_velocity = num_segments * (x_k_next - x_k)

    endpoint_mix = float(endpoint_mix)
    if endpoint_mix <= 0.0:
        return path_velocity

    endpoint_mix = min(endpoint_mix, 1.0)
    endpoint_velocity = trajs[idx_range, -1, :] - trajs[idx_range, 0, :]
    return (1.0 - endpoint_mix) * path_velocity + endpoint_mix * endpoint_velocity

def train_cfm_step(model, trajectories, optimizer, device, trajectory_weights=None, endpoint_mix=0.0):
    """
    对一批在线生成的轨迹执行一次训练更新
    trajectories: numpy array [N, Steps+1, Dim]
    """
    model.train()
    trajs = torch.FloatTensor(trajectories).to(device)
    weights = None
    if trajectory_weights is not None:
        weights = torch.FloatTensor(trajectory_weights).to(device).view(-1)
    N, T, Dim = trajs.shape
    M = T - 1 
    
    perm = torch.randperm(N)
    total_loss = 0
    num_batches = 0
    
    for i in range(0, N, Config.FM_BATCH_SIZE):
        indices = perm[i:i+Config.FM_BATCH_SIZE]
        batch_traj = trajs[indices]
        batch_x0 = batch_traj[:, 0, :]
        
        # 采样时间段 k
        k = torch.randint(0, M, (len(indices),)).to(device)
        idx_range = torch.arange(len(indices))
        x_k = batch_traj[idx_range, k, :]
        x_k_next = batch_traj[idx_range, k+1, :]
        
        # 线性插值与目标速度计算
        alpha = torch.rand(len(indices), 1).to(device)
        t_global = (k.unsqueeze(1) + alpha) / M
        x_t = (1 - alpha) * x_k + alpha * x_k_next
        v_target = build_velocity_targets(
            batch_traj,
            k,
            alpha,
            idx_range,
            endpoint_mix=endpoint_mix,
        )
        
        # 预测与优化
        v_pred = model(x_t, t_global, batch_x0)
        loss_per_sample = torch.mean((v_pred - v_target) ** 2, dim=-1)
        if weights is not None:
            batch_weights = weights[indices]
            loss = torch.sum(loss_per_sample * batch_weights) / torch.clamp(batch_weights.sum(), min=1e-8)
        else:
            loss = torch.mean(loss_per_sample)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches
def train_cfm(model, trajectories, optimizer, device):
    """
    Phase 3: 训练条件流匹配
    trajectories: numpy array [N, Steps+1, Dim]
    """
    model.train()
    trajs = torch.FloatTensor(trajectories).to(device)
    N, T, Dim = trajs.shape
    M = T - 1 # 段数
    
    print(f"[Flow] Start training on {N} trajectories...")
    
    for epoch in range(Config.FM_EPOCHS):
        perm = torch.randperm(N)
        epoch_loss = 0
        
        for i in range(0, N, Config.FM_BATCH_SIZE):
            indices = perm[i:i+Config.FM_BATCH_SIZE]
            batch_traj = trajs[indices]
            batch_x0 = batch_traj[:, 0, :]
            
            # 随机采样时间段 k
            k = torch.randint(0, M, (len(indices),)).to(device)
            
            # 获取 x_k, x_{k+1}
            idx_range = torch.arange(len(indices))
            x_k = batch_traj[idx_range, k, :]
            x_k_next = batch_traj[idx_range, k+1, :]
            
            # 线性插值
            alpha = torch.rand(len(indices), 1).to(device)
            t_global = (k.unsqueeze(1) + alpha) / M
            x_t = (1 - alpha) * x_k + alpha * x_k_next
            
            # 目标速度
            v_target = M * (x_k_next - x_k)
            
            # 预测与 Loss
            v_pred = model(x_t, t_global, batch_x0)
            loss = torch.mean((v_pred - v_target) ** 2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1} | Loss: {epoch_loss / (N/Config.FM_BATCH_SIZE):.4f}")

def inference_ode(model, x_query, device):
    """
    Phase 4: 使用 Euler 法解 ODE
    
    Args:
        model: Flow Matching 模型
        x_query: 起始点 [N, D]
        device: torch.device
    
    Returns:
        numpy array [N, D]
    """
    model.eval()
    x_curr = torch.FloatTensor(x_query).to(device)
    x_0 = x_curr.clone() # Condition
    
    steps = Config.INFERENCE_STEPS
    dt = 1.0 / steps
    
    with torch.no_grad():
        for i in range(steps):
            t_val = i / steps
            t_tensor = torch.full((x_curr.shape[0], 1), t_val, device=device)
            
            velocity = model(x_curr, t_tensor, x_0)
            x_curr = x_curr + velocity * dt
            
    return x_curr.cpu().numpy()

def inference_ode_multi(model, x_query, device, num_samples=1, noise_scale=0.0):
    """
    Generate multiple ODE proposals per seed and return flattened endpoints.

    Returns:
        proposals: numpy array [N * num_samples, D]
        seed_ids: numpy array [N * num_samples]
    """
    model.eval()
    x_query_t = torch.FloatTensor(x_query).to(device)
    n_seed, dim = x_query_t.shape
    num_samples = max(1, int(num_samples))

    x_seed = x_query_t.unsqueeze(1).repeat(1, num_samples, 1).reshape(n_seed * num_samples, dim)
    x_curr = x_seed.clone()
    if noise_scale > 0:
        x_curr = x_curr + noise_scale * torch.randn_like(x_curr)
    x_0 = x_seed.clone()

    steps = Config.INFERENCE_STEPS
    dt = 1.0 / steps

    with torch.no_grad():
        for i in range(steps):
            t_val = i / steps
            t_tensor = torch.full((x_curr.shape[0], 1), t_val, device=device)
            velocity = model(x_curr, t_tensor, x_0)
            x_curr = x_curr + velocity * dt

    seed_ids = np.repeat(np.arange(n_seed), num_samples)
    return x_curr.cpu().numpy(), seed_ids

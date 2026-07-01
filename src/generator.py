# src/generator.py
import torch
import torch.optim as optim
import numpy as np
from src.config import Config
from gpytorch.kernels import(
    RBFKernel, LinearKernel, MaternKernel, RQKernel, PeriodicKernel,
    CosineKernel, PolynomialKernel 
)

kernel_dict = {'rbf': RBFKernel,'matern': MaternKernel, 
                'rq' : RQKernel, 'period': PeriodicKernel, 'cosine': CosineKernel,
                'poly': PolynomialKernel}

def select_gp_fit_tensors(x_train, y_train, max_samples, seed=None):
    max_samples = int(max_samples)
    if max_samples <= 0 or x_train.shape[0] <= max_samples:
        return x_train, y_train

    if seed is None:
        indices = torch.randperm(x_train.shape[0], device=x_train.device)[:max_samples]
    else:
        generator = torch.Generator(device=x_train.device)
        generator.manual_seed(int(seed))
        indices = torch.randperm(x_train.shape[0], generator=generator, device=x_train.device)[:max_samples]
    return x_train[indices], y_train[indices]


class GP: 
    """ROOT 风格的 GP 类，用于生成动态轨迹"""
    def __init__(self, device, x_train, y_train, lengthscale, variance, noise, mean_prior, kernel='rbf'):
        self.device = device 
        self.x_train = x_train
        self.y_train = y_train 
        self.kernel = kernel_dict[kernel]().to(device)
        self.noise = noise
        self.variance = variance
        self.mean_prior = mean_prior
        self.kernel.lengthscale = lengthscale
        
    def set_hyper(self, lengthscale, variance): 
        self.variance = variance 
        self.kernel.lengthscale = lengthscale
        if hasattr(self, 'coef'):
            del self.coef
        with torch.no_grad():
            # 计算核矩阵
            K_train_train = self.variance * self.kernel.forward(self.x_train, self.x_train)
            K_train_train.diagonal().add_(self.noise)  # In-place modification
            
            # Cholesky 分解（这是最耗时的操作）
            L = torch.linalg.cholesky(K_train_train)
            b = (self.y_train - self.mean_prior).unsqueeze(-1)
            self.coef = torch.cholesky_solve(b, L).squeeze(-1).detach()
    
    def mean_posterior(self, x_test): 
        # Posterior mean
        K_train_test = self.variance * self.kernel.forward(self.x_train, x_test)
        mu_star = self.mean_prior + torch.matmul(K_train_test.T, self.coef)
        return mu_star


def sampling_data_from_GP(x_train, device, GP_Model, num_gradient_steps=50, num_functions=5, num_points=10, 
                          learning_rate=0.001, delta_lengthscale=0.1, delta_variance=0.1, seed=0, threshold_diff=0.1, verbose=False,
                          record_paths=False):
    """
    ROOT 风格的 GP 采样函数
    每个函数采样 num_points 个 (x_low, y_low) -> (x_high, y_high) 配对
    返回格式: datasets = {f'f{i}': [(x_high, y_high), (x_low, y_low)], ...}
    """
    import time
    lengthscale = GP_Model.kernel.lengthscale
    variance = GP_Model.variance 
    torch.manual_seed(seed=seed)
    datasets = {}
    learning_rate_vec = torch.cat((
        -learning_rate*torch.ones(num_points, x_train.shape[1], device=device), 
        learning_rate*torch.ones(num_points, x_train.shape[1], device=device)
    ))

    total_set_hyper_time = 0
    total_gradient_time = 0
    total_posterior_time = 0
    
    for iter in range(num_functions):
        datasets[f'f{iter}'] = []
        
        # 为每个函数采样不同的超参数
        set_hyper_start = time.time()
        new_lengthscale = lengthscale + delta_lengthscale*(torch.rand(1, device=device)*2 -1)
        new_variance = variance + delta_variance*(torch.rand(1, device=device)*2 -1)
        GP_Model.set_hyper(lengthscale=new_lengthscale, variance=new_variance)
        total_set_hyper_time += time.time() - set_hyper_start
    
        # 随机选择起始点
        selected_indices = torch.randperm(x_train.shape[0])[:num_points]
        low_x = x_train[selected_indices].clone().detach().requires_grad_(True)
        high_x = x_train[selected_indices].clone().detach().requires_grad_(True)
        joint_x = torch.cat((low_x, high_x)) 
        
        # 使用梯度上升和下降来寻找高分和低分设计
        gradient_start = time.time()
        low_history = [joint_x[:num_points, :].detach().clone()]
        high_history = [joint_x[num_points:, :].detach().clone()]
        for t in range(num_gradient_steps): 
            mu_star = GP_Model.mean_posterior(joint_x)
            grad = torch.autograd.grad(mu_star.sum(), joint_x)[0]
            joint_x += learning_rate_vec*grad
            if record_paths:
                low_history.append(joint_x[:num_points, :].detach().clone())
                high_history.append(joint_x[num_points:, :].detach().clone())
        total_gradient_time += time.time() - gradient_start 

        posterior_start = time.time()
        joint_y = GP_Model.mean_posterior(joint_x)
        total_posterior_time += time.time() - posterior_start
        
        low_x = joint_x[:num_points, :]
        high_x = joint_x[num_points:, :]
        low_y = joint_y[:num_points]
        high_y = joint_y[num_points:]
        
        full_paths = None
        full_scores = None
        if record_paths:
            low_paths = torch.stack(low_history, dim=0).flip(0)
            high_paths = torch.stack(high_history, dim=0)
            full_paths = torch.cat((low_paths, high_paths[1:]), dim=0)
            with torch.no_grad():
                flat_paths = full_paths.reshape(-1, full_paths.shape[-1])
                full_scores = GP_Model.mean_posterior(flat_paths).reshape(full_paths.shape[0], full_paths.shape[1])

        # 过滤掉分数差太小的配对
        for i in range(num_points):
            if high_y[i] - low_y[i] <= threshold_diff:
                continue
            if record_paths:
                sample = {
                    "trajectory": full_paths[:, i, :].detach(),
                    "trajectory_scores": full_scores[:, i].detach(),
                    "x_high": high_x[i].detach(),
                    "y_high": high_y[i].detach(),
                    "x_low": low_x[i].detach(),
                    "y_low": low_y[i].detach(),
                }
            else:
                sample = [(high_x[i].detach(), high_y[i].detach()), (low_x[i].detach(), low_y[i].detach())]
            datasets[f'f{iter}'].append(sample)

    # 恢复原始超参数
    GP_Model.kernel.lengthscale = lengthscale
    GP_Model.variance = variance
    
    if verbose:
        print(f"    [GP内部] set_hyper: {total_set_hyper_time:.2f}s | 梯度采样: {total_gradient_time:.2f}s | 后验: {total_posterior_time:.2f}s")
    
    return datasets


def generate_trajectories_from_GP_samples(GP_samples, device, num_steps=50, path_mode="recorded"):
    """
    从 GP 采样结果生成轨迹。
    兼容两种格式：
    1) 旧格式: [(x_high, y_high), (x_low, y_low)]，使用线性插值。
    2) 记录路径格式: {"trajectory": Tensor[T, D], ...}，保留 GP 梯度搜索路径并按需重采样。
    """
    import torch.nn.functional as F

    if path_mode not in {"recorded", "endpoint"}:
        raise ValueError("path_mode must be either 'recorded' or 'endpoint'")

    recorded_paths = []
    all_x_low = []
    all_x_high = []

    for _, samples in GP_samples.items():
        for sample in samples:
            if isinstance(sample, dict) and "trajectory" in sample:
                if path_mode == "recorded":
                    recorded_paths.append(sample["trajectory"].detach().to(device))
                else:
                    all_x_low.append(sample["x_low"].detach())
                    all_x_high.append(sample["x_high"].detach())
            else:
                (x_high, y_high), (x_low, y_low) = sample
                all_x_low.append(x_low)
                all_x_high.append(x_high)

    if len(recorded_paths) > 0:
        paths = torch.stack(recorded_paths, dim=0).to(device)
        target_len = num_steps + 1
        if paths.shape[1] != target_len:
            paths = F.interpolate(
                paths.permute(0, 2, 1),
                size=target_len,
                mode="linear",
                align_corners=True,
            ).permute(0, 2, 1)
        return paths.cpu().numpy()

    if len(all_x_low) == 0:
        return np.array([]).reshape(0, num_steps + 1, 0)

    x_low_batch = torch.stack(all_x_low, dim=0).to(device)
    x_high_batch = torch.stack(all_x_high, dim=0).to(device)
    alphas = torch.linspace(0, 1, num_steps + 1, device=device).view(-1, 1, 1)
    x_low_expand = x_low_batch.unsqueeze(0)
    x_high_expand = x_high_batch.unsqueeze(0)
    trajs_tensor = (1 - alphas) * x_low_expand + alphas * x_high_expand
    trajs_tensor = trajs_tensor.permute(1, 0, 2)
    return trajs_tensor.cpu().numpy()


def generate_long_trajectories(oracle, X_init_numpy, device):
    """
    保留原有函数以保持兼容性（如果其他地方还在使用）
    ROOT 风格长轨迹：GD 反转(500步) + GA 冲顶(800步)
    """
    X_curr = torch.FloatTensor(X_init_numpy).to(device).requires_grad_(True)
    X_start = torch.FloatTensor(X_init_numpy).to(device)
    
    # --- 阶段 B: 冲顶 (GA) 800步 ---
    X_curr.data = X_start.data.clone()
    y_target_high = torch.full((X_start.shape[0], 1), 4.0).to(device)
    traj_part2 = [X_curr.detach().cpu().clone()]
    
    opt_asc = optim.Adam([X_curr], lr=Config.TRAJ_LR)
    for _ in range(Config.TRAJ_STEPS_ASC):
        opt_asc.zero_grad()
        mu = oracle.predict_mean(X_curr)
        sigma_sq = oracle.predict_uncertainty(X_curr)
        
        loss_fwd = torch.mean((mu - y_target_high) ** 2)
        loss_bwd = torch.mean((X_curr - X_start) ** 2)
        loss_unc = torch.mean(sigma_sq)
        
        loss = (loss_fwd 
                + Config.LAMBDA_BACKWARD * loss_bwd 
                + Config.LAMBDA_UNCERTAINTY * loss_unc)
        
        loss.backward()
        opt_asc.step()
        traj_part2.append(X_curr.detach().cpu().clone())

    full_traj = traj_part2
    traj_tensor = torch.stack(full_traj, dim=1).to(device)
    
    valid_trajs = _filter_trajectories(oracle, traj_tensor)
    
    return valid_trajs.cpu().numpy()


def _filter_trajectories(oracle, traj_tensor):
    """验证: mu(XT) - k*sigma(XT) > mu(X0) + k*sigma(X0)"""
    with torch.no_grad():
        mu_0 = oracle.predict_mean(traj_tensor[:, 0, :])
        sig_0 = torch.sqrt(oracle.predict_uncertainty(traj_tensor[:, 0, :]))
        
        mu_T = oracle.predict_mean(traj_tensor[:, -1, :])
        sig_T = torch.sqrt(oracle.predict_uncertainty(traj_tensor[:, -1, :]))
    
    lower_bound_T = mu_T - Config.KAPPA * sig_T
    upper_bound_0 = mu_0 + Config.KAPPA * sig_0
    
    valid_mask = (lower_bound_T > upper_bound_0).squeeze()
    num_valid = valid_mask.sum().item()
    print(f"[Generator] Filtered {traj_tensor.shape[0]} -> {num_valid} valid trajectories.")
    
    return traj_tensor[valid_mask]
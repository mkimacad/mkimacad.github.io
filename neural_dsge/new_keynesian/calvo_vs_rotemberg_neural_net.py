import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import time

# ==========================================
# 0. Hardware & Precision
# ==========================================
torch.set_float32_matmul_precision('high')
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Training Unified Master Script on: {device}")

# ==========================================
# 1. Global Economic Parameters
# ==========================================
beta     = 0.99
sigma    = 1.0
varphi   = 1.0
epsilon  = 6.0
phi_pi   = 1.5
phi_y    = 0.125
rho_a    = 0.9
rho_nu   = 0.5
sigma_a  = 0.01
sigma_nu = 0.005

T_zlb_base = 15       
rho_bar    = 0.02     
tau_max    = 20.0     

R_ss, Pi_ss, Y_ss = 1.0 / beta, 1.0, 1.0

a_max  = 3 * (sigma_a  / (1 - rho_a **2) ** 0.5)
nu_max = 3 * (sigma_nu / (1 - rho_nu**2) ** 0.5)
v_center, v_radius = 1.020, 0.030
theta_center, theta_radius = 0.43, 0.42

# Quadrature Nodes (7x7)
n_nodes = 7
x_base, w_base = np.polynomial.hermite.hermgauss(n_nodes)
X_a, X_nu = np.meshgrid(np.sqrt(2)*sigma_a*x_base, np.sqrt(2)*sigma_nu*x_base)
W_a, W_nu = np.meshgrid(w_base/np.sqrt(np.pi), w_base/np.sqrt(np.pi))
quad_nodes = torch.tensor(np.stack([X_a.flatten(), X_nu.flatten()], axis=1), dtype=torch.float32, device=device)
quad_weights = torch.tensor((W_a * W_nu).flatten(), dtype=torch.float32, device=device)
Q = quad_nodes.shape[0]

def smooth_zlb(R_target: torch.Tensor, s: float = 0.002, hard_zlb: bool = False) -> torch.Tensor:
    if hard_zlb: return torch.clamp(R_target, min=1.0)
    return 1.0 + s * F.softplus((R_target - 1.0) / s)

class ResBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.lin1 = nn.Linear(width, width)
        self.lin2 = nn.Linear(width, width)
        self.act  = nn.Mish()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.lin1(self.norm(x)))
        return self.act(x + self.lin2(h))

# ==============================================================================
# PART A: THE CALVO MODEL
# ==============================================================================
print("\n" + "="*50 + "\nINITIALIZING CALVO MODEL\n" + "="*50)

class DSGENet_Calvo(nn.Module):
    def __init__(self, width: int = 512, n_blocks: int = 5):
        super().__init__()
        self.register_buffer('a_b', torch.tensor(a_max, dtype=torch.float32))
        self.register_buffer('nu_b', torch.tensor(nu_max, dtype=torch.float32))
        self.register_buffer('v_c', torch.tensor(v_center, dtype=torch.float32))
        self.register_buffer('v_r', torch.tensor(v_radius, dtype=torch.float32))
        self.register_buffer('th_c', torch.tensor(theta_center, dtype=torch.float32))
        self.register_buffer('th_r', torch.tensor(theta_radius, dtype=torch.float32))
        self.register_buffer('tau_m', torch.tensor(tau_max, dtype=torch.float32))

        self.stem   = nn.Linear(5, width) 
        self.blocks = nn.ModuleList([ResBlock(width) for _ in range(n_blocks)])
        self.head   = nn.Linear(width, 4) 

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = torch.cat([
            state[:, 0:1] / self.a_b, state[:, 1:2] / self.nu_b,
            (state[:, 2:3] - self.v_c) / self.v_r, (state[:, 3:4] - self.th_c) / self.th_r,
            state[:, 4:5] / self.tau_m,
        ], dim=1)

        h = F.mish(self.stem(x))
        for block in self.blocks: h = block(h)
        raw = self.head(h)

        theta_in = state[:, 3:4]
        KF_ref   = 1.0 / (1.0 - theta_in * beta).clamp(min=0.05)
        raw_scaled = raw * 0.1 
        C  = Y_ss  * F.softplus(raw_scaled[:, 0:1] + 0.5413) 
        Pi = Pi_ss * F.softplus(raw_scaled[:, 1:2] + 0.5413)
        K  = KF_ref * F.softplus(raw_scaled[:, 2:3] + 0.5413)
        F_val = KF_ref * F.softplus(raw_scaled[:, 3:4] + 0.5413)
        return torch.cat([C, Pi, K, F_val], dim=1)

def compute_residuals_calvo(net, state_t, nodes, weights, hard_zlb=False):
    B = state_t.shape[0]
    A_t, nu_t, v_lag, theta, tau_t = state_t[:,0:1], state_t[:,1:2], state_t[:,2:3], state_t[:,3:4], state_t[:,4:5]
    rho_t = torch.where(tau_t > 0.1, rho_bar, 0.0)

    preds = net(state_t)
    C_t, Pi_t, K_t, F_t = preds[:,0:1], preds[:,1:2], preds[:,2:3], preds[:,3:4]
    Y_t = C_t

    R_target = R_ss * (Pi_t / Pi_ss)**phi_pi * (Y_t / Y_ss)**phi_y * torch.exp(nu_t)
    R_t = smooth_zlb(R_target, hard_zlb=hard_zlb)

    p_star_t = (K_t / F_t.clamp(min=1e-5)).clamp(min=0.1, max=10.0)
    v_t = ((1 - theta) * p_star_t**(-epsilon) + theta * Pi_t**epsilon * v_lag).clamp(max=50.0)

    A_rep, nu_rep, v_rep, th_rep = A_t.expand(B, Q).reshape(-1, 1), nu_t.expand(B, Q).reshape(-1, 1), v_t.expand(B, Q).reshape(-1, 1), theta.expand(B, Q).reshape(-1, 1)
    tau_t1 = torch.clamp(tau_t - 1.0, min=0.0).expand(B, Q).reshape(-1, 1)
    eps_a, eps_nu = nodes[:, 0].expand(B, Q).reshape(-1, 1), nodes[:, 1].expand(B, Q).reshape(-1, 1)

    preds1 = net(torch.cat([rho_a*A_rep + eps_a, rho_nu*nu_rep + eps_nu, v_rep, th_rep, tau_t1], dim=1))
    C_t1, Pi_t1, K_t1, F_t1 = preds1[:,0:1].view(B,Q,1), preds1[:,1:2].view(B,Q,1), preds1[:,2:3].view(B,Q,1), preds1[:,3:4].view(B,Q,1)
    
    w = weights.view(1, Q, 1)
    C_t_exp = C_t.unsqueeze(1)

    Exp_Euler = (w * (C_t1 / C_t_exp.clamp(min=1e-5))**(-sigma) / Pi_t1.clamp(min=1e-5)).sum(dim=1)
    Exp_K     = (w * Pi_t1**epsilon * K_t1).sum(dim=1)
    Exp_F     = (w * Pi_t1**(epsilon - 1) * F_t1).sum(dim=1)
    real_mc_t = ((Y_t * v_t / torch.exp(A_t))**varphi * (Y_t / torch.exp(A_t)) * C_t.clamp(min=1e-5)**(-sigma)).clamp(max=1e4)

    res_euler = beta * torch.exp(rho_t) * R_t * Exp_Euler - 1.0
    res_Pi    = 1.0 - ((1-theta) * p_star_t**(1-epsilon) + theta * Pi_t**(epsilon-1.0))
    res_K     = 1.0 - (real_mc_t + theta * beta * Exp_K) / K_t.clamp(min=1e-5)
    res_F     = 1.0 - (Y_t * C_t.clamp(min=1e-5)**(-sigma) + theta*beta*Exp_F) / F_t.clamp(min=1e-5)
    return res_euler, res_K, res_F, res_Pi

def generate_batch_calvo(n_samples: int) -> torch.Tensor:
    n_bnd, n_int = n_samples // 5, n_samples - n_samples // 5
    u = torch.rand(n_int, 4, device=device) * torch.pi
    theta_int = theta_center + torch.cos(u[:, 3:4]) * theta_radius

    edge = 0.1 * theta_radius
    th_lo = theta_center - theta_radius + torch.rand(n_bnd//2, 1, device=device) * edge
    th_hi = theta_center + theta_radius - edge + torch.rand(n_bnd - n_bnd//2, 1, device=device) * edge
    theta_bnd = torch.cat([th_lo, th_hi], dim=0)
    u_bnd = torch.rand(n_bnd, 4, device=device) * torch.pi

    n_trans, n_rand = int(n_samples * 0.3), n_samples - int(n_samples * 0.3)
    tau_rand = torch.randint(0, int(T_zlb_base) + 3, (n_rand, 1), device=device).float()
    trans_choices = torch.tensor([0.0, 1.0, float(T_zlb_base-1), float(T_zlb_base)], device=device)
    tau_trans = trans_choices[torch.randint(0, 4, (n_trans, 1), device=device)]
    tau_all = torch.cat([tau_rand, tau_trans], dim=0)[torch.randperm(n_samples, device=device)]

    return torch.cat([
        torch.cat([torch.cos(u[:, 0:1]) * a_max, torch.cos(u_bnd[:, 0:1]) * a_max], dim=0),
        torch.cat([torch.cos(u[:, 1:2]) * nu_max, torch.cos(u_bnd[:, 1:2]) * nu_max], dim=0),
        torch.cat([v_center + torch.cos(u[:, 2:3]) * v_radius, v_center + torch.cos(u_bnd[:, 2:3]) * v_radius], dim=0),
        torch.cat([theta_int, theta_bnd], dim=0), tau_all
    ], dim=1)

net_calvo = DSGENet_Calvo().to(device)
for m in net_calvo.modules():
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight, gain=0.1)
        nn.init.zeros_(m.bias)

opt_calvo = optim.AdamW(net_calvo.parameters(), lr=5e-5, weight_decay=1e-5)
sch_calvo = optim.lr_scheduler.CosineAnnealingLR(opt_calvo, T_max=40_000, eta_min=1e-7)

batch_size, epochs_calvo = 2048, 40_000 
print("Training Calvo Network (OHEM & Strict Terminal)...")
for epoch in range(epochs_calvo):
    state_explore = generate_batch_calvo(batch_size * 2)
    with torch.no_grad():
        re, rK, rF, rP = compute_residuals_calvo(net_calvo, state_explore, quad_nodes, quad_weights)
        loss_exp = (re.pow(2) + rK.pow(2) + rF.pow(2) + rP.pow(2)).mean(dim=1)
    
    _, top_idx = torch.topk(loss_exp, batch_size // 2)
    state_t = torch.cat([state_explore[top_idx], generate_batch_calvo(batch_size - batch_size // 2)], dim=0)

    opt_calvo.zero_grad(set_to_none=True)
    re, rK, rF, rP = compute_residuals_calvo(net_calvo, state_t, quad_nodes, quad_weights)
    
    loss_euler = (re.clamp(-10,10).pow(2).mean() + rK.clamp(-10,10).pow(2).mean() + rF.clamp(-10,10).pow(2).mean() + rP.clamp(-10,10).pow(2).mean()) / 4.0

    state_term = state_t.clone()
    state_term[:, 4:5], state_term[:, 0:2] = 0.0, 0.0
    pred_term = net_calvo(state_term)
    loss_term = 10.0 * ((pred_term[:, 0] - Y_ss).pow(2).mean() + (pred_term[:, 1] - Pi_ss).pow(2).mean())
    
    loss = loss_euler + loss_term
    if not torch.isfinite(loss): continue
    loss.backward()
    if not all(p.grad is None or torch.isfinite(p.grad).all() for p in net_calvo.parameters()): continue
    torch.nn.utils.clip_grad_norm_(net_calvo.parameters(), max_norm=0.5)
    opt_calvo.step(); sch_calvo.step()
    if (epoch + 1) % 5000 == 0: print(f"Calvo Epoch {epoch+1}/{epochs_calvo} | Loss: {loss.item():.4e}")

# ==============================================================================
# PART B: THE ROTEMBERG MODEL (Stabilized & Clamped)
# ==============================================================================
print("\n" + "="*50 + "\nINITIALIZING ROTEMBERG MODEL\n" + "="*50)

class DSGENet_Rotemberg(nn.Module):
    def __init__(self, width: int = 512, n_blocks: int = 5):
        super().__init__()
        self.register_buffer('a_b', torch.tensor(a_max, dtype=torch.float32))
        self.register_buffer('nu_b', torch.tensor(nu_max, dtype=torch.float32))
        self.register_buffer('th_c', torch.tensor(theta_center, dtype=torch.float32))
        self.register_buffer('th_r', torch.tensor(theta_radius, dtype=torch.float32))
        self.register_buffer('tau_m', torch.tensor(tau_max, dtype=torch.float32))

        self.stem   = nn.Linear(4, width) 
        self.blocks = nn.ModuleList([ResBlock(width) for _ in range(n_blocks)])
        self.head   = nn.Linear(width, 2) 

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = torch.cat([
            state[:, 0:1] / self.a_b, state[:, 1:2] / self.nu_b,
            (state[:, 2:3] - self.th_c) / self.th_r, state[:, 3:4] / self.tau_m,
        ], dim=1)
        h = F.mish(self.stem(x))
        for block in self.blocks: h = block(h)
        raw_scaled = self.head(h) * 0.1 
        Y  = Y_ss  * F.softplus(raw_scaled[:, 0:1] + 0.5413) 
        Pi = Pi_ss * F.softplus(raw_scaled[:, 1:2] + 0.5413)
        return torch.cat([Y, Pi], dim=1)

def compute_residuals_rotemberg(net, state_t, nodes, weights, hard_zlb=False):
    B = state_t.shape[0]
    A_t, nu_t, theta, tau_t = state_t[:, 0:1], state_t[:, 1:2], state_t[:, 2:3], state_t[:, 3:4]
    rho_t = torch.where(tau_t > 0.1, rho_bar, 0.0)

    preds = net(state_t)
    Y_t, Pi_t = preds[:, 0:1], preds[:, 1:2]
    
    phi_rot = (theta * (epsilon - 1.0)) / ((1.0 - theta) * (1.0 - beta * theta))
    phi_rot_safe = phi_rot.view(-1, 1).clamp(min=1e-4)
    
    adj_frac = torch.clamp((phi_rot_safe / 2.0) * (Pi_t - 1.0)**2, max=0.95)
    C_t = (Y_t * (1.0 - adj_frac)).clamp(min=1e-5)

    R_t = smooth_zlb(R_ss * (Pi_t / Pi_ss)**phi_pi * (Y_t / Y_ss)**phi_y * torch.exp(nu_t), hard_zlb=hard_zlb)

    A_rep, nu_rep, th_rep = A_t.expand(B, Q).reshape(-1, 1), nu_t.expand(B, Q).reshape(-1, 1), theta.expand(B, Q).reshape(-1, 1)
    tau_t1 = torch.clamp(tau_t - 1.0, min=0.0).expand(B, Q).reshape(-1, 1)
    eps_a, eps_nu = nodes[:, 0].expand(B, Q).reshape(-1, 1), nodes[:, 1].expand(B, Q).reshape(-1, 1)

    preds1 = net(torch.cat([rho_a*A_rep + eps_a, rho_nu*nu_rep + eps_nu, th_rep, tau_t1], dim=1))
    Y_t1, Pi_t1 = preds1[:, 0:1].view(B, Q, 1), preds1[:, 1:2].view(B, Q, 1)
    
    adj_frac_t1 = torch.clamp((phi_rot.view(-1,1,1) / 2.0) * (Pi_t1 - 1.0)**2, max=0.95)
    C_t1 = (Y_t1 * (1.0 - adj_frac_t1)).clamp(min=1e-5)

    w = weights.view(1, Q, 1)
    C_t_exp = C_t.unsqueeze(1).clamp(min=1e-5)
    Y_t_exp = Y_t.unsqueeze(1).clamp(min=1e-5)
    
    Exp_Euler = (w * (C_t1 / C_t_exp)**(-sigma) / Pi_t1.clamp(min=1e-5)).sum(dim=1)
    res_euler = beta * torch.exp(rho_t) * R_t * Exp_Euler - 1.0

    mc_t = (C_t**sigma) * (Y_t**varphi) * (torch.exp(A_t)**(-(1.0 + varphi)))
    Exp_NKPC_scaled = (w * (C_t1 / C_t_exp)**(-sigma) * (Y_t1 / Y_t_exp) * (Pi_t1 - 1.0) * Pi_t1).sum(dim=1)
    res_Pi = (Pi_t - 1.0) * Pi_t - (epsilon - 1.0)*(mc_t - 1.0) / phi_rot_safe - beta * Exp_NKPC_scaled
    
    return res_euler, res_Pi

def generate_batch_rotemberg(n_samples: int) -> torch.Tensor:
    n_bnd, n_int = n_samples // 5, n_samples - n_samples // 5
    u = torch.rand(n_int, 3, device=device) * torch.pi
    th_lo = theta_center - theta_radius + torch.rand(n_bnd//2, 1, device=device) * (0.1*theta_radius)
    th_hi = theta_center + theta_radius - (0.1*theta_radius) + torch.rand(n_bnd - n_bnd//2, 1, device=device) * (0.1*theta_radius)
    u_bnd = torch.rand(n_bnd, 2, device=device) * torch.pi

    tau_rand = torch.randint(0, int(T_zlb_base) + 3, (n_samples - int(n_samples*0.3), 1), device=device).float()
    tau_trans = torch.tensor([0.0, 1.0, float(T_zlb_base-1), float(T_zlb_base)], device=device)[torch.randint(0, 4, (int(n_samples*0.3), 1), device=device)]
    tau_all = torch.cat([tau_rand, tau_trans], dim=0)[torch.randperm(n_samples, device=device)]

    return torch.cat([
        torch.cat([torch.cos(u[:,0:1])*a_max, torch.cos(u_bnd[:,0:1])*a_max], dim=0),
        torch.cat([torch.cos(u[:,1:2])*nu_max, torch.cos(u_bnd[:,1:2])*nu_max], dim=0),
        torch.cat([theta_center + torch.cos(u[:, 2:3])*theta_radius, torch.cat([th_lo, th_hi], dim=0)], dim=0),
        tau_all
    ], dim=1)

net_rot = DSGENet_Rotemberg().to(device)
for m in net_rot.modules():
    if isinstance(m, nn.Linear): nn.init.xavier_normal_(m.weight, gain=0.1); nn.init.zeros_(m.bias)

opt_rot = optim.AdamW(net_rot.parameters(), lr=5e-5, weight_decay=1e-5)
sch_rot = optim.lr_scheduler.CosineAnnealingLR(opt_rot, T_max=40_000, eta_min=1e-7)

epochs_rot = 40_000
print("Training Rotemberg Network (OHEM & Strict Terminal)...")
for epoch in range(epochs_rot):
    state_explore = generate_batch_rotemberg(batch_size * 2)
    with torch.no_grad():
        re, rP = compute_residuals_rotemberg(net_rot, state_explore, quad_nodes, quad_weights)
        loss_exp = (re.pow(2) + rP.pow(2)).mean(dim=1)
    
    _, top_idx = torch.topk(loss_exp, batch_size // 2)
    state_t = torch.cat([state_explore[top_idx], generate_batch_rotemberg(batch_size - batch_size // 2)], dim=0)

    opt_rot.zero_grad(set_to_none=True)
    re, rP = compute_residuals_rotemberg(net_rot, state_t, quad_nodes, quad_weights)
    
    loss_euler = (re.clamp(-10,10).pow(2).mean() + rP.clamp(-10,10).pow(2).mean()) / 2.0

    state_term = state_t.clone(); state_term[:, 3:4], state_term[:, 0:2] = 0.0, 0.0
    pred_term = net_rot(state_term)
    loss_term = 10.0 * ((pred_term[:, 0] - Y_ss).pow(2).mean() + (pred_term[:, 1] - Pi_ss).pow(2).mean())
    
    loss = loss_euler + loss_term
    if not torch.isfinite(loss): continue
    loss.backward()
    if not all(p.grad is None or torch.isfinite(p.grad).all() for p in net_rot.parameters()): continue
    torch.nn.utils.clip_grad_norm_(net_rot.parameters(), max_norm=0.5)
    opt_rot.step(); sch_rot.step()
    if (epoch + 1) % 5000 == 0: print(f"Rotemberg Epoch {epoch+1}/{epochs_rot} | Loss: {loss.item():.4e}")

# ==============================================================================
# PART C: CONVENTIONAL NON-LINEAR PERFECT FORESIGHT SOLVERS
# ==============================================================================
def solve_conventional_pf_calvo(th, T_zlb):
    C_path = nn.Parameter(torch.ones(T_zlb, device=device))
    Pi_path = nn.Parameter(torch.ones(T_zlb, device=device))
    KF_ss = 1.0 / (1.0 - th * beta)
    K_path = nn.Parameter(torch.full((T_zlb,), KF_ss, device=device))
    F_path = nn.Parameter(torch.full((T_zlb,), KF_ss, device=device))
    
    opt_pf = optim.LBFGS([C_path, Pi_path, K_path, F_path], max_iter=250, tolerance_grad=1e-7, line_search_fn="strong_wolfe")
    
    def closure():
        opt_pf.zero_grad()
        C = torch.cat([C_path, torch.tensor([Y_ss], device=device)])
        Pi = torch.cat([Pi_path, torch.tensor([Pi_ss], device=device)])
        K = torch.cat([K_path, torch.tensor([KF_ss], device=device)])
        F_var = torch.cat([F_path, torch.tensor([KF_ss], device=device)])
        
        Y = C
        R_target = R_ss * (Pi / Pi_ss)**phi_pi * (Y / Y_ss)**phi_y
        R = torch.clamp(R_target, min=1.0)
        
        re_list, rK_list, rF_list, rP_list = [], [], [], []
        v_prev = 1.0
        for t in range(T_zlb):
            p_star = K[t] / torch.clamp(F_var[t], min=1e-5)
            v_t = (1 - th) * p_star**(-epsilon) + th * Pi[t]**epsilon * v_prev
            
            C_safe = torch.clamp(C, min=1e-5)
            exp_euler = (C_safe[t+1]/C_safe[t])**(-sigma) / torch.clamp(Pi[t+1], min=1e-5)
            re = beta * np.exp(rho_bar) * R[t] * exp_euler - 1.0
            mc_t = (Y[t] * v_t)**varphi * Y[t] * C_safe[t]**(-sigma)
            
            rk = 1.0 - (mc_t + th * beta * Pi[t+1]**epsilon * K[t+1]) / torch.clamp(K[t], min=1e-5)
            rf = 1.0 - (Y[t] * C_safe[t]**(-sigma) + th * beta * Pi[t+1]**(epsilon-1) * F_var[t+1]) / torch.clamp(F_var[t], min=1e-5)
            rp = 1.0 - ((1 - th) * p_star**(1 - epsilon) + th * Pi[t]**(epsilon - 1))
            
            re_list.append(re); rK_list.append(rk); rF_list.append(rf); rP_list.append(rp)
            v_prev = v_t
            
        loss = torch.stack(re_list).pow(2).mean() + torch.stack(rK_list).pow(2).mean() + torch.stack(rF_list).pow(2).mean() + torch.stack(rP_list).pow(2).mean()
        loss.backward()
        return loss
        
    opt_pf.step(closure)
    
    C = torch.cat([C_path, torch.tensor([Y_ss], device=device)])
    Pi = torch.cat([Pi_path, torch.tensor([Pi_ss], device=device)])
    K = torch.cat([K_path, torch.tensor([KF_ss], device=device)])
    F_var = torch.cat([F_path, torch.tensor([KF_ss], device=device)])
    v_path = []
    v_prev = 1.0
    for t in range(T_zlb):
        p_star = K[t] / torch.clamp(F_var[t], min=1e-5)
        v_t = (1 - th) * p_star**(-epsilon) + th * Pi[t]**epsilon * v_prev
        v_path.append(v_t.item())
        v_prev = v_t
        
    return C_path.detach().cpu().numpy(), Pi_path.detach().cpu().numpy(), np.array(v_path)

def solve_conventional_pf_rotemberg(th, T_zlb):
    Y_path = nn.Parameter(torch.ones(T_zlb, device=device))
    Pi_path = nn.Parameter(torch.ones(T_zlb, device=device))
    opt_pf = optim.LBFGS([Y_path, Pi_path], max_iter=250, tolerance_grad=1e-7, line_search_fn="strong_wolfe")
    
    phi_rot = (th * (epsilon - 1.0)) / ((1.0 - th) * (1.0 - beta * th))
    phi_rot_safe = max(phi_rot, 1e-4)
    
    def closure():
        opt_pf.zero_grad()
        Y = torch.cat([Y_path, torch.tensor([Y_ss], device=device)])
        Pi = torch.cat([Pi_path, torch.tensor([Pi_ss], device=device)])
        
        adj_frac = torch.clamp((phi_rot/2.0)*(Pi-1.0)**2, max=0.95)
        C = Y * (1.0 - adj_frac)
        
        R_target = R_ss * (Pi / Pi_ss)**phi_pi * (Y / Y_ss)**phi_y
        R = torch.clamp(R_target, min=1.0)
        
        re_list, rp_list = [], []
        for t in range(T_zlb):
            C_safe = torch.clamp(C, min=1e-5)
            exp_euler = (C_safe[t+1]/C_safe[t])**(-sigma) / torch.clamp(Pi[t+1], min=1e-5)
            re = beta * np.exp(rho_bar) * R[t] * exp_euler - 1.0
            mc_t = (C_safe[t]**sigma) * (Y[t]**varphi)
            
            exp_nkpc_scaled = (C_safe[t+1]/C_safe[t])**(-sigma) * (Y[t+1]/Y[t]) * (Pi[t+1]-1.0)*Pi[t+1]
            rp = (Pi[t]-1.0)*Pi[t] - (epsilon - 1.0)*(mc_t - 1.0)/phi_rot_safe - beta*exp_nkpc_scaled
            
            re_list.append(re); rp_list.append(rp)
            
        loss = torch.stack(re_list).pow(2).mean() + torch.stack(rp_list).pow(2).mean()
        loss.backward()
        return loss
        
    opt_pf.step(closure)
    return Y_path.detach().cpu().numpy(), Pi_path.detach().cpu().numpy()

# ==============================================================================
# PART D: EVALUATION & SEPARATED PLOTTING
# ==============================================================================
print("\n" + "="*50 + "\nGENERATING 7 BENCHMARK PLOTS\n" + "="*50)
thetas_irf = [0.75, 0.50, 0.25, 0.01]
colors     = ['#d73027', '#fc8d59', '#91bfdb', '#4575b4']
T_plot = 40
rn_base = np.zeros(T_plot)

for t_step in range(T_plot): 
    rn_base[t_step] = (-np.log(beta) - (rho_bar if t_step < T_zlb_base else 0.0))*400

def style_6panel(axes_arr, title, has_dispersion=False):
    axes_arr[0,0].set_title('Output (% Dev)'); axes_arr[0,1].set_title('Inflation (Ann. bps)')
    axes_arr[0,2].set_title('Actual Nom. Rate (Ann. %)'); axes_arr[1,0].set_title('Shadow Nom. Rate (Ann. %)')
    axes_arr[1,1].set_title('Price Dispersion ($v_t$)' if has_dispersion else '')
    axes_arr[1,2].set_title('Natural Real Rate (Ann. %)')
    axes_arr[1,2].plot(rn_base, color='black', linewidth=2)
    axes_arr[1,2].fill_between(range(T_plot), rn_base, -np.log(beta)*400, where=(rn_base < -np.log(beta)*400), color='red', alpha=0.2)
    for i, ax in enumerate(axes_arr.flatten()): 
        if i == 4 and not has_dispersion: 
            ax.set_visible(False); continue
        ax.grid(alpha=0.3); ax.axhline(0 if i<5 else -np.log(beta)*400, color='black', alpha=0.5)

# ------------------------------------------------------------------------------
# 1. Figure 1: Linear Benchmark - Werning (Explosive)
# ------------------------------------------------------------------------------
fig_wern, axes_wern = plt.subplots(2, 3, figsize=(16, 9))
fig_wern.suptitle(f'Log-Linear Benchmark: Standard Werning Solution ($T={T_zlb_base}$)', fontsize=16)

for idx, th in enumerate(thetas_irf):
    kappa_lin = ((1 - th) * (1 - beta * th) / th) * (sigma + varphi)
    delta_lin = rho_bar + np.log(beta)
    
    Y_std, Pi_std = np.zeros(T_plot), np.zeros(T_plot)
    for t_step in range(int(T_zlb_base)-1, -1, -1):
        Y_std[t_step] = Y_std[t_step+1] + (1/sigma)*Pi_std[t_step+1] - (1/sigma)*delta_lin
        Pi_std[t_step] = kappa_lin * Y_std[t_step] + beta * Pi_std[t_step+1]
        
    Y_std_plot = (np.exp(Y_std) - 1.0) * 100
    Pi_std_plot = (np.exp(Pi_std) - 1.0) * 400

    lbl = f'$\\theta={th}$'
    axes_wern[0,0].plot(Y_std_plot, color=colors[idx], linestyle='--', alpha=0.8, linewidth=2, label=lbl)
    axes_wern[0,1].plot(Pi_std_plot, color=colors[idx], linestyle='--', alpha=0.8, linewidth=2)
    axes_wern[0,2].plot(np.zeros(T_plot), color=colors[idx], linestyle='--', alpha=0.8) 

style_6panel(axes_wern, "", has_dispersion=False)
axes_wern[0,0].legend(loc='lower left', fontsize=9)
plt.tight_layout()

# ------------------------------------------------------------------------------
# 2. Figure 2: Linear Benchmark - Cochrane (Bounded)
# ------------------------------------------------------------------------------
fig_coch, axes_coch = plt.subplots(2, 3, figsize=(16, 9))
fig_coch.suptitle(f'Log-Linear Benchmark: Bounded Cochrane Alternative ($T={T_zlb_base}$)', fontsize=16)

for idx, th in enumerate(thetas_irf):
    kappa_lin = ((1 - th) * (1 - beta * th) / th) * (sigma + varphi)
    delta_lin = rho_bar + np.log(beta)
    
    Y_coch, Pi_coch = np.zeros(T_plot), np.zeros(T_plot)
    for t_step in range(int(T_zlb_base)):
        Y_coch[t_step] = (1 - beta) / kappa_lin * delta_lin
        Pi_coch[t_step] = delta_lin

    Y_coch_plot = (np.exp(Y_coch) - 1.0) * 100
    Pi_coch_plot = (np.exp(Pi_coch) - 1.0) * 400

    lbl = f'$\\theta={th}$'
    axes_coch[0,0].plot(Y_coch_plot, color=colors[idx], linestyle=':', alpha=0.9, linewidth=2.5, label=lbl)
    axes_coch[0,1].plot(Pi_coch_plot, color=colors[idx], linestyle=':', alpha=0.9, linewidth=2.5)
    axes_coch[0,2].plot(np.zeros(T_plot), color=colors[idx], linestyle=':', alpha=0.9) 

style_6panel(axes_coch, "", has_dispersion=False)
axes_coch[0,0].legend(loc='lower left', fontsize=9)
plt.tight_layout()

# ------------------------------------------------------------------------------
# 3. Figure 3: Global NN - Calvo
# ------------------------------------------------------------------------------
fig_calvo_nn, axes_calvo_nn = plt.subplots(2, 3, figsize=(16, 9))
fig_calvo_nn.suptitle(f'Global Neural Network Solution: Calvo Pricing ($T={T_zlb_base}$)', fontsize=16)

for idx, th in enumerate(thetas_irf):
    Y_cal, Pi_cal, R_cal, R_shadow_cal, v_cal = [], [], [], [], []
    tau_cal, vc = float(T_zlb_base), 1.0

    with torch.no_grad():
        for t_step in range(T_plot):
            p_c = net_calvo(torch.tensor([[0.0, 0.0, vc, th, tau_cal]], dtype=torch.float32, device=device))[0]
            Y_cal.append((p_c[0].item() - Y_ss)*100); Pi_cal.append((p_c[1].item() - Pi_ss)*400)
            
            R_target = R_ss*(p_c[1].item()/Pi_ss)**phi_pi*(p_c[0].item()/Y_ss)**phi_y
            R_cal.append((max(R_target, 1.0) - 1.0)*400)
            R_shadow_cal.append((R_target - 1.0)*400)
            
            ps = max(max(p_c[2].item(),1e-4)/max(p_c[3].item(),1e-4), 0.5)
            vc = (1-th)*ps**(-epsilon) + th*p_c[1].item()**epsilon*vc
            v_cal.append(vc); tau_cal = max(tau_cal - 1.0, 0.0)

    lbl = f'$\\theta={th}$'
    axes_calvo_nn[0,0].plot(Y_cal, color=colors[idx], linestyle='-', linewidth=2, label=lbl)
    axes_calvo_nn[0,1].plot(Pi_cal, color=colors[idx], linestyle='-', linewidth=2)
    axes_calvo_nn[0,2].plot(R_cal, color=colors[idx], linestyle='-', linewidth=2)
    axes_calvo_nn[1,0].plot(R_shadow_cal, color=colors[idx], linestyle='-', linewidth=2)
    axes_calvo_nn[1,1].plot(v_cal, color=colors[idx], linestyle='-', linewidth=2)

style_6panel(axes_calvo_nn, "", has_dispersion=True)
axes_calvo_nn[0,0].legend(loc='lower left', fontsize=9)
plt.tight_layout()

# ------------------------------------------------------------------------------
# 4. Figure 4: Global NN - Rotemberg
# ------------------------------------------------------------------------------
fig_rot_nn, axes_rot_nn = plt.subplots(2, 3, figsize=(16, 9))
fig_rot_nn.suptitle(f'Global Neural Network Solution: Rotemberg Pricing ($T={T_zlb_base}$)', fontsize=16)

for idx, th in enumerate(thetas_irf):
    Y_rot, Pi_rot, R_rot, R_shadow_rot = [], [], [], []
    tau_rot = float(T_zlb_base)

    with torch.no_grad():
        for t_step in range(T_plot):
            p_r = net_rot(torch.tensor([[0.0, 0.0, th, tau_rot]], dtype=torch.float32, device=device))[0]
            Y_rot.append((p_r[0].item() - Y_ss)*100); Pi_rot.append((p_r[1].item() - Pi_ss)*400)
            
            R_target = R_ss*(p_r[1].item()/Pi_ss)**phi_pi*(p_r[0].item()/Y_ss)**phi_y
            R_rot.append((max(R_target, 1.0) - 1.0)*400)
            R_shadow_rot.append((R_target - 1.0)*400)
            
            tau_rot = max(tau_rot - 1.0, 0.0)

    lbl = f'$\\theta={th}$'
    axes_rot_nn[0,0].plot(Y_rot, color=colors[idx], linestyle='-', linewidth=2, label=lbl)
    axes_rot_nn[0,1].plot(Pi_rot, color=colors[idx], linestyle='-', linewidth=2)
    axes_rot_nn[0,2].plot(R_rot, color=colors[idx], linestyle='-', linewidth=2)
    axes_rot_nn[1,0].plot(R_shadow_rot, color=colors[idx], linestyle='-', linewidth=2)

style_6panel(axes_rot_nn, "", has_dispersion=False)
axes_rot_nn[0,0].legend(loc='lower left', fontsize=9)
plt.tight_layout()

# ------------------------------------------------------------------------------
# 5. Figure 5: Calvo Conventional Non-Linear PF
# ------------------------------------------------------------------------------
fig_calvo_pf, axes_calvo_pf = plt.subplots(2, 3, figsize=(16, 9))
fig_calvo_pf.suptitle(f'Conventional Perfect Foresight Solver: Calvo Pricing ($T={T_zlb_base}$)', fontsize=16)

for idx, th in enumerate(thetas_irf):
    Y_pf, Pi_pf, v_pf = solve_conventional_pf_calvo(th, int(T_zlb_base))
    
    Y_pf_plot = np.concatenate([(Y_pf - Y_ss)*100, np.zeros(T_plot - int(T_zlb_base))])
    Pi_pf_plot = np.concatenate([(Pi_pf - Pi_ss)*400, np.zeros(T_plot - int(T_zlb_base))])
    v_pf_plot = np.concatenate([v_pf, np.ones(T_plot - int(T_zlb_base))])
    
    R_target_pf = R_ss * (Pi_pf / Pi_ss)**phi_pi * (Y_pf / Y_ss)**phi_y
    R_pf = np.maximum(R_target_pf, 1.0)
    
    R_pf_plot = np.concatenate([(R_pf - 1.0)*400, np.zeros(T_plot - int(T_zlb_base))])
    R_shadow_pf = np.concatenate([(R_target_pf - 1.0)*400, np.zeros(T_plot - int(T_zlb_base))])

    lbl = f'$\\theta={th}$'
    axes_calvo_pf[0,0].plot(Y_pf_plot, color=colors[idx], linestyle='-', linewidth=2, label=lbl)
    axes_calvo_pf[0,1].plot(Pi_pf_plot, color=colors[idx], linestyle='-', linewidth=2)
    axes_calvo_pf[0,2].plot(R_pf_plot, color=colors[idx], linestyle='-', linewidth=2)
    axes_calvo_pf[1,0].plot(R_shadow_pf, color=colors[idx], linestyle='-', linewidth=2)
    axes_calvo_pf[1,1].plot(v_pf_plot, color=colors[idx], linestyle='-', linewidth=2)

style_6panel(axes_calvo_pf, "", has_dispersion=True)
axes_calvo_pf[0,0].legend(loc='lower left', fontsize=9)
plt.tight_layout()

# ------------------------------------------------------------------------------
# 6. Figure 6: Rotemberg Conventional Non-Linear PF
# ------------------------------------------------------------------------------
fig_rot_pf, axes_rot_pf = plt.subplots(2, 3, figsize=(16, 9))
fig_rot_pf.suptitle(f'Conventional Perfect Foresight Solver: Rotemberg Pricing ($T={T_zlb_base}$)', fontsize=16)

for idx, th in enumerate(thetas_irf):
    Y_pf, Pi_pf = solve_conventional_pf_rotemberg(th, int(T_zlb_base))
    
    Y_pf_plot = np.concatenate([(Y_pf - Y_ss)*100, np.zeros(T_plot - int(T_zlb_base))])
    Pi_pf_plot = np.concatenate([(Pi_pf - Pi_ss)*400, np.zeros(T_plot - int(T_zlb_base))])
    
    R_target_pf = R_ss * (Pi_pf / Pi_ss)**phi_pi * (Y_pf / Y_ss)**phi_y
    R_pf = np.maximum(R_target_pf, 1.0)
    
    R_pf_plot = np.concatenate([(R_pf - 1.0)*400, np.zeros(T_plot - int(T_zlb_base))])
    R_shadow_pf = np.concatenate([(R_target_pf - 1.0)*400, np.zeros(T_plot - int(T_zlb_base))])

    lbl = f'$\\theta={th}$'
    axes_rot_pf[0,0].plot(Y_pf_plot, color=colors[idx], linestyle='-', linewidth=2, label=lbl)
    axes_rot_pf[0,1].plot(Pi_pf_plot, color=colors[idx], linestyle='-', linewidth=2)
    axes_rot_pf[0,2].plot(R_pf_plot, color=colors[idx], linestyle='-', linewidth=2)
    axes_rot_pf[1,0].plot(R_shadow_pf, color=colors[idx], linestyle='-', linewidth=2)

style_6panel(axes_rot_pf, "", has_dispersion=False)
axes_rot_pf[0,0].legend(loc='lower left', fontsize=9)
plt.tight_layout()

# ------------------------------------------------------------------------------
# 7. Figure 7: Duration Bifurcation Analysis (Calvo & Rotemberg)
# ------------------------------------------------------------------------------
th_eval = 0.50 # BUG FIX: Assigned th_eval before plot generation
fig_bif, axes_bif = plt.subplots(2, 2, figsize=(12, 8))
fig_bif.suptitle(f'Forward Guidance Puzzle Resolution: Calvo vs Rotemberg ($\\theta={th_eval}$)', fontsize=16)
durations = [4, 8, 12, 16]
colors_dur = ['#abdda4', '#ffffbf', '#fdae61', '#d7191c']

for i, dur in enumerate(durations):
    Y_c, Pi_c, Y_r, Pi_r = [], [], [], []
    tc, tr, vc = float(dur), float(dur), 1.0
    with torch.no_grad():
        for _ in range(30):
            pc = net_calvo(torch.tensor([[0.0, 0.0, vc, th_eval, tc]], dtype=torch.float32, device=device))[0]
            pr = net_rot(torch.tensor([[0.0, 0.0, th_eval, tr]], dtype=torch.float32, device=device))[0]
            
            Y_c.append((pc[0].item()-Y_ss)*100); Pi_c.append((pc[1].item()-Pi_ss)*400)
            Y_r.append((pr[0].item()-Y_ss)*100); Pi_r.append((pr[1].item()-Pi_ss)*400)
            
            ps = max(max(pc[2].item(),1e-4)/max(pc[3].item(),1e-4), 0.5)
            vc = (1-th_eval)*ps**(-epsilon) + th_eval*pc[1].item()**epsilon*vc
            tc = max(tc - 1.0, 0.0); tr = max(tr - 1.0, 0.0)
            
    axes_bif[0,0].plot(Y_c, color=colors_dur[i], label=f'T={dur}', lw=2)
    axes_bif[0,1].plot(Pi_c, color=colors_dur[i], label=f'T={dur}', lw=2)
    axes_bif[1,0].plot(Y_r, color=colors_dur[i], label=f'T={dur}', lw=2)
    axes_bif[1,1].plot(Pi_r, color=colors_dur[i], label=f'T={dur}', lw=2)

axes_bif[0,0].set_title('Calvo Output (NN)'); axes_bif[0,1].set_title('Calvo Inflation (NN)')
axes_bif[1,0].set_title('Rotemberg Output (NN)'); axes_bif[1,1].set_title('Rotemberg Inflation (NN)')
for ax in axes_bif.flatten(): ax.grid(alpha=0.3); ax.axhline(0, color='black', ls='--'); ax.legend()
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# 8. Out-of-Sample L_inf Evaluation (Both Models)
# ------------------------------------------------------------------------------
print("\n" + "="*50 + "\nL_inf OOS EVALUATION (10,000 Nodes)\n" + "="*50)
net_calvo.eval(); net_rot.eval()
with torch.no_grad():
    g_A = torch.linspace(-a_max, a_max, 10, device=device); g_nu = torch.linspace(-nu_max, nu_max, 10, device=device)
    g_th = torch.linspace(theta_center-theta_radius, theta_center+theta_radius, 10, device=device)
    
    g_v = torch.linspace(v_center-v_radius, v_center+v_radius, 10, device=device)
    Ag_c, ng_c, vg_c, tg_c = torch.meshgrid(g_A, g_nu, g_v, g_th, indexing='ij')
    
    g_A_r = torch.linspace(-a_max, a_max, 20, device=device); g_nu_r = torch.linspace(-nu_max, nu_max, 20, device=device)
    g_th_r = torch.linspace(theta_center-theta_radius, theta_center+theta_radius, 25, device=device)
    Ag_r, ng_r, tg_r = torch.meshgrid(g_A_r, g_nu_r, g_th_r, indexing='ij')

    for t_val in [0, 1, T_zlb_base // 2, T_zlb_base]:
        st_c = torch.stack([Ag_c.flatten(), ng_c.flatten(), vg_c.flatten(), tg_c.flatten(), torch.full_like(Ag_c.flatten(), float(t_val))], dim=1)
        st_r = torch.stack([Ag_r.flatten(), ng_r.flatten(), tg_r.flatten(), torch.full_like(Ag_r.flatten(), float(t_val))], dim=1)
        
        errs_c, errs_r = [], []
        for i in range(0, st_c.shape[0], 4096):
            e_c, _, _, _ = compute_residuals_calvo(net_calvo, st_c[i:i+4096], quad_nodes, quad_weights, hard_zlb=True)
            errs_c.append(e_c.abs())
        for i in range(0, st_r.shape[0], 4096):
            e_r, _ = compute_residuals_rotemberg(net_rot, st_r[i:i+4096], quad_nodes, quad_weights, hard_zlb=True)
            errs_r.append(e_r.abs())
            
        print(f"Tau={t_val:>2} | Calvo Max L_inf: {torch.cat(errs_c).max():.4e} | Rotemberg Max L_inf: {torch.cat(errs_r).max():.4e}")

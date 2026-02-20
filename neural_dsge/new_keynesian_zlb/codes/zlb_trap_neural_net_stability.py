import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time

# ==========================================
# 0. Hardware & Precision
# ==========================================
torch.set_float32_matmul_precision('high')
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Training on device: {device}")

# ==========================================
# 1. Economic Parameters 
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

R_ss   = 1.0 / beta
Pi_ss  = 1.0
Y_ss   = 1.0

R_min  = 1.0   # Zero Lower Bound
R_max  = 1.10  # Policy Ceiling

# The two valid limits we care about
Pi_target = 1.0
Pi_trap   = beta * R_min

# ==========================================
# 2. Quadrature & Architecture
# ==========================================
n_nodes = 7
x_base, w_base = np.polynomial.hermite.hermgauss(n_nodes)
x_a, x_nu = np.sqrt(2) * sigma_a * x_base, np.sqrt(2) * sigma_nu * x_base
w_gh = w_base / np.sqrt(np.pi)

X_a, X_nu = np.meshgrid(x_a, x_nu)
W_a, W_nu = np.meshgrid(w_gh, w_gh)
quad_nodes   = torch.tensor(np.stack([X_a.flatten(), X_nu.flatten()], axis=1), dtype=torch.float32, device=device)
quad_weights = torch.tensor((W_a * W_nu).flatten(), dtype=torch.float32, device=device)
Q = quad_nodes.shape[0]

class DSGENet_Limits(nn.Module):
    def __init__(self, width=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, width), nn.Mish(),
            nn.Linear(width, width), nn.Mish(),
            nn.Linear(width, width), nn.Mish(),
            nn.Linear(width, 4)
        )
    def forward(self, state):
        raw = self.net(state) * 0.1
        theta_in = state[:, 3:4]
        KF_ref = 1.0 / (1.0 - theta_in * beta).clamp(min=0.05)
        
        C  = Y_ss  * F.softplus(raw[:, 0:1] + 0.5413) 
        Pi = Pi_ss * F.softplus(raw[:, 1:2] + 0.5413)
        K  = KF_ref * F.softplus(raw[:, 2:3] + 0.5413)
        F_v = KF_ref * F.softplus(raw[:, 3:4] + 0.5413)
        return torch.cat([C, Pi, K, F_v], dim=1)

# ==========================================
# 3. Bounded Policy Residuals
# ==========================================
def smooth_bounds(R_target, minimum, maximum, s=0.005):
    R_lower = minimum + s * F.softplus((R_target - minimum) / s)
    return maximum - s * F.softplus((maximum - R_lower) / s)

def compute_residuals(net, state_t):
    B = state_t.shape[0]
    A_t, nu_t, v_lag, theta = state_t[:, 0:1], state_t[:, 1:2], state_t[:, 2:3], state_t[:, 3:4]

    preds = net(state_t)
    C_t, Pi_t, K_t, F_t = preds[:, 0:1], preds[:, 1:2], preds[:, 2:3], preds[:, 3:4]
    Y_t = C_t

    R_target = R_ss * (Pi_t / Pi_ss) ** phi_pi * (Y_t / Y_ss) ** phi_y * torch.exp(nu_t)
    R_t = smooth_bounds(R_target, R_min, R_max)

    p_star_t = (K_t / F_t.clamp(min=1e-5)).clamp(min=0.1, max=10.0)
    v_t      = ((1 - theta) * p_star_t ** (-epsilon) + theta * Pi_t ** epsilon * v_lag).clamp(max=50.0)

    A_rep  = A_t.repeat_interleave(Q, dim=0)
    nu_rep = nu_t.repeat_interleave(Q, dim=0)
    v_rep  = v_t.repeat_interleave(Q, dim=0)
    th_rep = theta.repeat_interleave(Q, dim=0)

    eps_a  = quad_nodes[:, 0].repeat(B)
    eps_nu = quad_nodes[:, 1].repeat(B)

    state_t1 = torch.stack([rho_a*A_rep.squeeze() + eps_a, rho_nu*nu_rep.squeeze() + eps_nu, v_rep.squeeze(), th_rep.squeeze()], dim=1)
    
    preds1 = net(state_t1)
    C_t1, Pi_t1 = preds1[:, 0].view(B, Q), preds1[:, 1].view(B, Q)
    K_t1, F_t1  = preds1[:, 2].view(B, Q), preds1[:, 3].view(B, Q)

    w = quad_weights.view(1, Q)
    Exp_Euler = (w * (C_t1 / C_t.clamp(min=1e-5)) ** (-sigma) / Pi_t1.clamp(min=1e-5)).sum(dim=1, keepdim=True)
    Exp_K     = (w * Pi_t1 ** epsilon        * K_t1).sum(dim=1, keepdim=True)
    Exp_F     = (w * Pi_t1 ** (epsilon - 1) * F_t1).sum(dim=1, keepdim=True)

    real_mc_t = ((Y_t * v_t / torch.exp(A_t)) ** varphi * (Y_t / torch.exp(A_t)) * C_t.clamp(min=1e-5) ** (-sigma)).clamp(max=1e4)

    res_euler = beta * R_t * Exp_Euler - 1.0
    res_Pi    = 1.0 - ((1-theta) * p_star_t**(1-epsilon) + theta * Pi_t**(epsilon-1.0))
    res_K     = 1.0 - (real_mc_t + theta * beta * Exp_K) / K_t.clamp(min=1e-5)
    res_F     = 1.0 - (Y_t * C_t.clamp(min=1e-5)**(-sigma) + theta*beta*Exp_F) / F_t.clamp(min=1e-5)

    return res_euler, res_K, res_F, res_Pi

# ==========================================
# 4. Training Loop
# ==========================================
def train_model(target_pi=None, name_label="Model", epochs=15_000):
    net = DSGENet_Limits().to(device)
    optimizer = optim.AdamW(net.parameters(), lr=2e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    nudge_status = f"Nudged to {target_pi:.3f}" if target_pi is not None else "Free (No Nudge)"
    print(f"Training {name_label} [{nudge_status}]...")
    start_time = time.time()
    
    for epoch in range(epochs):
        st = torch.rand(1024, 4, device=device)
        st[:, 0] = (st[:, 0] * 2 - 1) * 0.03
        st[:, 1] = (st[:, 1] * 2 - 1) * 0.01
        st[:, 2] = 1.0 + st[:, 2] * 0.05
        st[:, 3] = 0.01 + st[:, 3] * 0.8
        
        optimizer.zero_grad()
        re, rK, rF, rP = compute_residuals(net, st)
        loss_eq = re.pow(2).mean() + rK.pow(2).mean() + rF.pow(2).mean() + rP.pow(2).mean()

        if target_pi is not None:
            st_origin = torch.zeros(128, 4, device=device)
            st_origin[:, 2] = 1.0
            st_origin[:, 3] = torch.rand(128, device=device) * 0.8 + 0.01
            pi_pred = net(st_origin)[:, 1]
            loss_nudge = (pi_pred - target_pi).pow(2).mean()
            loss = loss_eq + loss_nudge * 0.5
        else:
            loss = loss_eq

        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
    print(f"  -> Done in {time.time()-start_time:.1f}s | Eq Loss: {loss_eq.item():.2e}")
    return net

net_target = train_model(target_pi=Pi_target, name_label="Target Model")
net_trap   = train_model(target_pi=Pi_trap,   name_label="Trap Model")
net_free   = train_model(target_pi=None,      name_label="Free Model")

# ==========================================
# 5. Out-of-Sample (OOS) Performance
# ==========================================
def evaluate_oos(net_dict, n_samples=10_000):
    print("\n" + "="*60)
    print("Out-of-Sample (OOS) Structural Performance (Mean Absolute Error)")
    print("="*60)
    st_oos = torch.rand(n_samples, 4, device=device)
    st_oos[:, 0] = (st_oos[:, 0] * 2 - 1) * 0.04 
    st_oos[:, 1] = (st_oos[:, 1] * 2 - 1) * 0.02 
    st_oos[:, 2] = 1.0 + st_oos[:, 2] * 0.08     
    st_oos[:, 3] = 0.01 + st_oos[:, 3] * 0.90    
    with torch.no_grad():
        for name, net in net_dict.items():
            net.eval()
            re, rK, rF, rP = compute_residuals(net, st_oos)
            mae_euler = re.abs().mean().item()
            log10_euler = np.log10(max(mae_euler, 1e-10))
            print(f"{name:<15} | Log10 Euler: {log10_euler:5.2f} | K: {rK.abs().mean().item():.2e} | Pi: {rP.abs().mean().item():.2e}")

nets = {'Target Nudged': net_target, 'Trap Nudged': net_trap, 'Free (No Nudge)': net_free}
evaluate_oos(nets)

# ==========================================
# 6. Steady-State Multiplicity Mapping
# ==========================================
print("\nMapping Equilibrium Selection at Steady State...")
thetas = np.linspace(0.8, 0.01, 100)
Pi_ss_paths = {name: [] for name in nets}
with torch.no_grad():
    for th in thetas:
        st = torch.tensor([[0.0, 0.0, 1.0, float(th)]], dtype=torch.float32, device=device)
        for name, model in nets.items():
            p_out = model(st)[0]
            Pi_ss_paths[name].append((p_out[1].item() - 1.0) * 400) 

# ==========================================
# 7. Dynamic IRF under Tech Shock (theta = 0.05)
# ==========================================
print("\nSimulating Tech Shock near flexible prices (theta = 0.05)...")
T = 30
theta_flex = 0.05 
Y_paths  = {name: [] for name in nets}
Pi_paths = {name: [] for name in nets}
R_paths  = {name: [] for name in nets}

A_shock = np.zeros(T)
A_shock[0] = 0.02 
for t in range(1, T): A_shock[t] = rho_a * A_shock[t-1]

with torch.no_grad():
    for name, model in nets.items():
        v_current = 1.0
        for t in range(T):
            st = torch.tensor([[A_shock[t], 0.0, v_current, theta_flex]], dtype=torch.float32, device=device)
            preds = model(st)[0]
            Y_t, Pi_t, K_t, F_t = preds[0].item(), preds[1].item(), preds[2].item(), preds[3].item()
            
            Y_paths[name].append((Y_t - Y_ss) * 100)
            Pi_paths[name].append((Pi_t - 1.0) * 400)
            
            R_target = R_ss * (Pi_t / Pi_ss)**phi_pi * (Y_t / Y_ss)**phi_y
            R_t = np.clip(R_target, R_min, R_max)
            R_paths[name].append((R_t - 1.0) * 400)
            
            p_star = max(K_t / max(F_t, 1e-5), 0.1)
            v_current = (1 - theta_flex)*p_star**(-epsilon) + theta_flex*Pi_t**epsilon * v_current

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
fig.suptitle('Equilibrium Selection: Unconstrained Network vs. Nudged Networks', fontsize=16)

colors = {'Target Nudged': '#4575b4', 'Trap Nudged': '#d73027', 'Free (No Nudge)': '#2ca02c'}
styles = {'Target Nudged': '-', 'Trap Nudged': '--', 'Free (No Nudge)': '-.'}

for name in nets.keys(): axes[0].plot(thetas, Pi_ss_paths[name], label=name, color=colors[name], ls=styles[name], lw=2.5)
axes[0].axhline(0, color='black', alpha=0.3)
axes[0].axhline((beta - 1.0) * 400, color='red', alpha=0.3, label='Theoretical Trap')
axes[0].invert_xaxis()
axes[0].set_title('Inflation Limit at Origin')
axes[0].set_xlabel('Price Stickiness ($\\theta$)')
axes[0].set_ylabel('Ann. bps dev from Target')
axes[0].legend()
axes[0].grid(alpha=0.3)

for name in nets.keys(): axes[1].plot(Y_paths[name], label=name, color=colors[name], ls=styles[name], lw=2.5)
axes[1].set_title(f'Output IRF (+2% Tech Shock, $\\theta={theta_flex}$)')
axes[1].set_ylabel('% Deviation from SS')
axes[1].legend()
axes[1].grid(alpha=0.3)

for name in nets.keys(): axes[2].plot(Pi_paths[name], label=name, color=colors[name], ls=styles[name], lw=2.5)
axes[2].set_title(f'Inflation IRF (+2% Tech Shock, $\\theta={theta_flex}$)')
axes[2].set_ylabel('Ann. bps dev from Target')
axes[2].legend()
axes[2].grid(alpha=0.3)

for name in nets.keys(): axes[3].plot(R_paths[name], label=name, color=colors[name], ls=styles[name], lw=2.5)
axes[3].axhline(0, color='red', alpha=0.3, label='Zero Lower Bound ($R=1.0$)')
axes[3].set_title(f'Nominal Rate IRF (+2% Tech Shock, $\\theta={theta_flex}$)')
axes[3].set_ylabel('Ann. bps (0 = ZLB)')
axes[3].legend()
axes[3].grid(alpha=0.3)
plt.tight_layout(); plt.show()

# ==========================================
# 8. High-Resolution Loss Landscape Analysis
# ==========================================
print("\nMapping High-Resolution Loss Landscape (1D Slices and 2D Contours)...")

def calc_loss_landscape(Pi_grid, Y_grid, theta_val=0.50):
    # Bounded Taylor Rule
    R_target = R_ss * (Pi_grid / Pi_ss)**phi_pi * (Y_grid / Y_ss)**phi_y
    s = 0.005
    R_lower = R_min + s * np.log(1.0 + np.exp((R_target - R_min) / s))
    R_val   = R_max - s * np.log(1.0 + np.exp((R_max - R_lower) / s))

    # Euler Residual
    res_euler = beta * R_val / Pi_grid - 1.0

    # Pricing Residuals 
    MC = Y_grid**(1.0 + varphi) * Y_grid**(-sigma)
    denom_K = 1.0 - theta_val * beta * Pi_grid**epsilon
    denom_F = 1.0 - theta_val * beta * Pi_grid**(epsilon - 1.0)

    valid = (denom_K > 0) & (denom_F > 0)
    K_val = np.where(valid, MC / denom_K, 1e5)
    F_val = np.where(valid, Y_grid**(1.0 - sigma) / denom_F, 1e5)
    p_star = K_val / F_val

    res_pi = np.where(valid, 1.0 - ((1.0 - theta_val) * p_star**(1.0 - epsilon) + theta_val * Pi_grid**(epsilon - 1.0)), 1e5)

    Loss = res_euler**2 + res_pi**2
    return np.log10(np.clip(Loss, 1e-12, None))

# --- Generate 1D Cross-Section (Holding Y = 1.0) ---
pi_1d = np.linspace(0.97, 1.03, 1000)
y_1d  = np.ones_like(pi_1d)
loss_1d = calc_loss_landscape(pi_1d, y_1d)

# --- Generate 2D Grids for Contours ---
# Zoomed around Target
pi_targ_grid = np.linspace(0.99, 1.01, 200)
y_targ_grid  = np.linspace(0.99, 1.01, 200)
Pi_targ, Y_targ = np.meshgrid(pi_targ_grid, y_targ_grid)
loss_targ_2d = calc_loss_landscape(Pi_targ, Y_targ)

# Zoomed around Trap
pi_trap_grid = np.linspace(beta - 0.01, beta + 0.01, 200)
y_trap_grid  = np.linspace(0.99, 1.01, 200)
Pi_trap, Y_trap = np.meshgrid(pi_trap_grid, y_trap_grid)
loss_trap_2d = calc_loss_landscape(Pi_trap, Y_trap)

# --- Plotting the Landscape ---
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
fig2.suptitle('Optimization Landscape: Why the Trap is a Gravitational Sink', fontsize=16)

# Subplot 1: 1D Cross Section
axes2[0].plot(pi_1d, loss_1d, color='black', lw=2)
axes2[0].axvline(1.0, color='#4575b4', linestyle='--', label='Target Equilibrium')
axes2[0].axvline(beta, color='#d73027', linestyle='--', label='Trap Equilibrium')
axes2[0].set_title('1D Slice of Loss (Holding Y=1.0)')
axes2[0].set_xlabel('Inflation ($\\Pi$)')
axes2[0].set_ylabel('Log10 MSE Loss')
axes2[0].legend()
axes2[0].grid(alpha=0.3)

# Subplot 2: 2D Contour (Target)
cntr1 = axes2[1].contourf(Pi_targ, Y_targ, loss_targ_2d, levels=50, cmap='viridis')
axes2[1].scatter([1.0], [1.0], color='red', s=50, label='Target Eq', zorder=5)
axes2[1].set_title('Contour Zoom: Target Equilibrium')
axes2[1].set_xlabel('Inflation ($\\Pi$)')
axes2[1].set_ylabel('Output ($Y$)')
axes2[1].legend()
fig2.colorbar(cntr1, ax=axes2[1], fraction=0.046, pad=0.04)

# Subplot 3: 2D Contour (Trap)
cntr2 = axes2[2].contourf(Pi_trap, Y_trap, loss_trap_2d, levels=50, cmap='viridis')
axes2[2].scatter([beta], [1.0], color='red', s=50, label='Trap Eq', zorder=5)
axes2[2].set_title('Contour Zoom: ZLB Trap Equilibrium')
axes2[2].set_xlabel('Inflation ($\\Pi$)')
axes2[2].set_ylabel('Output ($Y$)')
axes2[2].legend()
fig2.colorbar(cntr2, ax=axes2[2], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()

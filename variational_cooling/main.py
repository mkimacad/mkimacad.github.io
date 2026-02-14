import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

torch.set_float32_matmul_precision('high')

# ==========================================
# 1. 3-SAT Problem Generators
# ==========================================

def generate_3sat(M, alpha=4.26):
    """Generates a random 3-SAT instance."""
    C = int(alpha * M)
    clauses = torch.zeros((C, 3), dtype=torch.long)
    for i in range(C):
        clauses[i] = torch.randperm(M)[:3]
    signs = torch.randint(0, 2, (C, 3), dtype=torch.float32)
    return clauses, signs

def get_custom_3sat():
    """
    Users can modify this function to return ANY specific 3-SAT problem.
    
    Default: A 'Planted Solution' instance with M=36.
    It uses a fixed seed to generate heavily constrained clauses (C=165) 
    that all agree with exactly one hidden bitstring: [0, 1, 0, 1, 0, 1...]
    This creates an extremely hard problem with a single narrow ground state.
    """
    M = 36
    C = 165 # Ratio ~4.58 (Overconstrained to eliminate spurious accidental solutions)
    
    # Use a fixed seed for reproducibility of this exact 'known' instance
    g = torch.Generator()
    g.manual_seed(42)
    
    # The planted (hidden) ground state we want the network to find
    hidden_solution = torch.arange(M) % 2 
    
    clauses = torch.zeros((C, 3), dtype=torch.long)
    signs = torch.zeros((C, 3), dtype=torch.float32)
    
    for i in range(C):
        vars_idx = torch.randperm(M, generator=g)[:3]
        clauses[i] = vars_idx
        
        # Randomly assign negation signs
        clause_signs = torch.randint(0, 2, (3,), generator=g).float()
        
        # Evaluate if this random clause is satisfied by our hidden solution
        vals = hidden_solution[vars_idx].float()
        lits = vals * (1.0 - clause_signs) + (1.0 - vals) * clause_signs
        
        # If all 3 literals evaluate to 0, the clause is violated.
        # We must flip one random sign to ensure the planted solution satisfies it.
        if lits.sum() == 0:
            flip_idx = torch.randint(0, 3, (1,), generator=g).item()
            clause_signs[flip_idx] = 1.0 - clause_signs[flip_idx]
            
        signs[i] = clause_signs
        
    print(f"\n[!] Loaded Custom Planted 3-SAT Instance: M={M}, C={C}")
    print(f"[!] Hidden Target Solution: {hidden_solution.tolist()}")
    return clauses, signs, M

def print_3sat_instance(clauses, signs, print_ratio=0.05):
    C = clauses.shape[0]
    num_to_print = max(1, int(C * print_ratio))
    print(f"\n--- Printing {num_to_print} of {C} Clauses ({print_ratio*100:.1f}%) ---")
    formula_str = []
    for i in range(num_to_print):
        literals = [f"~x_{v}" if s == 1.0 else f"x_{v}" for v, s in zip(clauses[i].tolist(), signs[i].tolist())]
        formula_str.append(f"({' v '.join(literals)})")
    print(" ^ \n".join(formula_str))
    if num_to_print < C: print(f" ^ \n... (and {C - num_to_print} more clauses)")
    print("-------------------------------------------------\n")

def compute_3sat_energy(x, clauses, signs):
    vars_in_clauses = x[:, clauses] 
    signs_expanded = signs.unsqueeze(0)
    literals = vars_in_clauses * (1.0 - signs_expanded) + (1.0 - vars_in_clauses) * signs_expanded
    violations = torch.prod(1.0 - literals, dim=-1)
    return violations.sum(dim=-1)

# ==========================================
# 2. ULTRA-FAST STATIC KV CACHE TRANSFORMER
# ==========================================

class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, kv_cache=None, cache_idx=0):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] 
        
        if kv_cache is not None:
            past_k, past_v = kv_cache
            past_k[:, :, cache_idx : cache_idx + T, :] = k
            past_v[:, :, cache_idx : cache_idx + T, :] = v
            k_attn = past_k[:, :, : cache_idx + T, :]
            v_attn = past_v[:, :, : cache_idx + T, :]
            y = F.scaled_dot_product_attention(q, k_attn, v_attn, is_causal=False)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            
        y = y.transpose(1, 2).contiguous().reshape(B, T, C)
        return self.proj(y)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim), nn.GELU(), nn.Linear(4 * embed_dim, embed_dim)
        )

    def forward(self, x, kv_cache=None, cache_idx=0):
        attn_out = self.attn(self.ln_1(x), kv_cache, cache_idx)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x

class FastSpinTransformer(nn.Module):
    def __init__(self, num_vars, embed_dim=128, num_heads=4, num_layers=3):
        super().__init__()
        self.num_vars = num_vars
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.token_embed = nn.Embedding(3, embed_dim)
        self.pos_embed = nn.Embedding(num_vars, embed_dim)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, x):
        B, N = x.shape
        device = x.device
        sos = torch.full((B, 1), 2, dtype=torch.long, device=device)
        x_shifted = torch.cat([sos, x[:, :-1].long()], dim=1)
        
        positions = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
        emb = self.token_embed(x_shifted) + self.pos_embed(positions)
        
        for block in self.blocks:
            emb = block(emb, kv_cache=None)
            
        return self.head(self.ln_f(emb)).squeeze(-1)

    def sample_and_log_prob(self, batch_size, device):
        x_generated = torch.zeros((batch_size, self.num_vars), dtype=torch.long, device=device)
        current_token = torch.full((batch_size, 1), 2, dtype=torch.long, device=device)
        
        kv_caches = []
        for _ in range(self.num_layers):
            k_cache = torch.zeros((batch_size, self.num_heads, self.num_vars, self.head_dim), device=device)
            v_cache = torch.zeros((batch_size, self.num_heads, self.num_vars, self.head_dim), device=device)
            kv_caches.append((k_cache, v_cache))
            
        for i in range(self.num_vars):
            pos = torch.tensor([[i]], device=device).expand(batch_size, 1)
            x = self.token_embed(current_token) + self.pos_embed(pos)
            
            for j, block in enumerate(self.blocks):
                x = block(x, kv_caches[j], cache_idx=i)
                
            logits = self.head(self.ln_f(x)).squeeze(-1)
            probs = torch.sigmoid(logits)
            
            next_spin = torch.bernoulli(probs).long()
            x_generated[:, i] = next_spin.squeeze(-1)
            current_token = next_spin
            
        logits_full = self(x_generated)
        bce = F.binary_cross_entropy_with_logits(logits_full, x_generated.float(), reduction='none')
        log_probs = -bce.sum(dim=-1)
        
        return x_generated.float(), log_probs

# ==========================================
# 3. Plotting Observables
# ==========================================

def plot_physics_metrics(C_ij_tensor, q_batch_tensor):
    C_ij = C_ij_tensor.cpu().numpy()
    q_batch = q_batch_tensor.cpu().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    im = axes[0].imshow(C_ij, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0].set_title('Spin-Spin Correlation Matrix $C_{ij}$', fontsize=14, pad=15)
    axes[0].set_xlabel('Variable Index $j$', fontsize=12)
    axes[0].set_ylabel('Variable Index $i$', fontsize=12)
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label='Correlation Strength')
    
    axes[1].hist(q_batch, bins=50, color='indigo', alpha=0.7, density=True, edgecolor='black')
    axes[1].set_title('Replica Overlap Distribution $P(q)$', fontsize=14, pad=15)
    axes[1].set_xlabel('Overlap $q$', fontsize=12)
    axes[1].set_ylabel('Probability Density $P(q)$', fontsize=12)
    
    q_mean = np.mean(q_batch)
    axes[1].axvline(q_mean, color='red', linestyle='dashed', linewidth=2, label=rf'Mean $\langle q \rangle = {q_mean:.3f}$')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()

# ==========================================
# 4. Main Loop
# ==========================================

def run_experiment(M=50, alpha=4.26, print_ratio=0.05, assignment_print_ratio=0.5, 
                   epochs=1500, batch_size=2048, beta_schedule=(0.1, 15.0), 
                   clip_grad_norm=1.0, use_custom_instance=False):
                   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {device}")

    # --- TOGGLE PROBLEM SOURCE ---
    if use_custom_instance:
        clauses, signs, M = get_custom_3sat() # Note: Overrides M
    else:
        clauses, signs = generate_3sat(M, alpha=alpha)
        print(f"\nGenerated Random 3-SAT Instance: M={M} variables, C={clauses.shape[0]} clauses (Ratio alpha={alpha:.3f})")
        
    print_3sat_instance(clauses, signs, print_ratio=print_ratio)
    clauses, signs = clauses.to(device), signs.to(device)
    
    model = FastSpinTransformer(num_vars=M, embed_dim=128, num_heads=4, num_layers=3).to(device)
    
    if torch.__version__ >= "2.0.0" and torch.cuda.is_available():
        print("Compiling model for H200 (mode='reduce-overhead')...")
        model = torch.compile(model, mode="reduce-overhead")
        
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    beta_start, beta_end = beta_schedule
    
    clip_status = f"Enabled (Max Norm: {clip_grad_norm})" if clip_grad_norm else "DISABLED"
    print(f"\n--- Starting Variational Cooling | Gradient Clipping: {clip_status} ---")
    
    for epoch in range(epochs):
        model.train()
        fraction = epoch / max(1, epochs - 1)
        beta = beta_start * (beta_end / beta_start) ** fraction
        
        with torch.no_grad():
            x_samples, _ = model.sample_and_log_prob(batch_size, device)
            energies = compute_3sat_energy(x_samples, clauses, signs)
            
        logits = model(x_samples)
        bce = F.binary_cross_entropy_with_logits(logits, x_samples, reduction='none')
        log_probs = -bce.sum(dim=-1)
        
        with torch.no_grad():
            local_free_energy = beta * energies + log_probs
            baseline = local_free_energy.mean()
            
        loss = (log_probs * (local_free_energy - baseline)).mean()
        
        optimizer.zero_grad()
        loss.backward()
        
        if clip_grad_norm is not None and clip_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
            
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d} | Beta: {beta:.3f} | Mean E: {energies.mean().item():.4f} | Min E: {energies.min().item():.1f}")

    print("\n--- Cooling Complete. Extracting Metrics ---")
    model.eval()
    with torch.no_grad():
        test_samples, _ = model.sample_and_log_prob(batch_size * 2, device)
        test_energies = compute_3sat_energy(test_samples, clauses, signs)
        
        min_idx = torch.argmin(test_energies)
        best_energy = test_energies[min_idx].item()
        best_assignment = test_samples[min_idx].cpu().numpy().astype(int)
        
        print(f"\nBest Assignment Found: {int(best_energy)} violated clauses out of {clauses.shape[0]}")
        if best_energy == 0:
            print(">> SUCCESS: Found a perfect satisfying assignment!")
        else:
            print(">> FAILED: Reached a local minimum or problem is unsatisfiable.")
            
        num_to_print = max(1, int(M * assignment_print_ratio))
        num_to_print = min(num_to_print, M) 
        
        assignment_str = ", ".join([f"x_{i}={best_assignment[i]}" for i in range(num_to_print)])
        suffix = " ..." if num_to_print < M else ""
        print(f"\nPartial Variable Assignment ({num_to_print} of {M} shown based on ratio {assignment_print_ratio}):\n[{assignment_str}{suffix}]\n")
        
        spins = 2.0 * test_samples - 1.0 
        mean_spins = spins.mean(dim=0)
        spin_spin = torch.matmul(spins.t(), spins) / (batch_size * 2)
        C_ij = spin_spin - torch.outer(mean_spins, mean_spins)

        replica_1, _ = model.sample_and_log_prob(batch_size * 2, device)
        replica_2, _ = model.sample_and_log_prob(batch_size * 2, device)
        q_batch = ((2.0 * replica_1 - 1.0) * (2.0 * replica_2 - 1.0)).mean(dim=-1)
        
        print(f"Mean Overlap <q>: {q_batch.mean().item():.4f} | Variance Var(q): {q_batch.var().item():.6f}")
        plot_physics_metrics(C_ij, q_batch)

if __name__ == "__main__":
    # Toggle 'use_custom_instance' to True to run the extremely hard Planted 3-SAT (M=36).
    # When True, the M=50 and alpha=4.26 arguments are safely ignored.
    run_experiment(
        M=50, 
        alpha=4.26, 
        print_ratio=0.05, 
        assignment_print_ratio=1.0, # Print all 36 variables of the custom instance
        epochs=1500, 
        batch_size=2048, 
        beta_schedule=(0.1, 15.0), 
        clip_grad_norm=1.0,
        use_custom_instance=True 
    )

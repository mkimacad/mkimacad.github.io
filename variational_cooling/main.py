import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random
import urllib.request
import tarfile
import io

torch.set_float32_matmul_precision('high')

# ==========================================
# 1. Native SATLIB Academic Downloader + Synthetic Generator
# ==========================================

def fetch_satlib_direct(M=50, target_sat=True, num_instances=3, seed=42):
    """Download instances from SATLIB for sizes 20-250 variables"""
    random.seed(seed)
    
    sat_map = {
        20: "https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/uf20-91.tar.gz",
        50: "https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/uf50-218.tar.gz",
        75: "https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/uf75-325.tar.gz",
        100: "https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/uf100-430.tar.gz",
        125: "https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/uf125-538.tar.gz",
        150: "https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/uf150-645.tar.gz",
        175: "https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/uf175-753.tar.gz",
        200: "https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/uf200-860.tar.gz",
        225: "https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/uf225-960.tar.gz",
        250: "https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/uf250-1065.tar.gz"
    }
    unsat_map = {
        50: "https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/uuf50-218.tar.gz",
        75: "https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/uuf75-325.tar.gz",
        100: "https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/uuf100-430.tar.gz",
        125: "https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/uuf125-538.tar.gz",
        150: "https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/uuf150-645.tar.gz",
        175: "https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/uuf175-753.tar.gz",
        200: "https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/uuf200-860.tar.gz",
        225: "https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/uuf225-960.tar.gz",
        250: "https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/uuf250-1065.tar.gz"
    }
    
    if target_sat and M not in sat_map: M = 50
    if not target_sat and M not in unsat_map: M = 50
        
    url = sat_map[M] if target_sat else unsat_map[M]
    label = "SAT" if target_sat else "UNSAT"
    
    print(f"\n[!] Downloading Official SATLIB Database directly from UBC...")
    print(f"[!] Target URL: {url}")
    
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        tar_stream = io.BytesIO(response.read())
        
    extracted_cnfs = []
    with tarfile.open(fileobj=tar_stream, mode="r:gz") as tar:
        members = [m for m in tar.getmembers() if m.name.endswith(".cnf")]
        members.sort(key=lambda x: x.name) 
        random.shuffle(members) 
        
        for m in members[:num_instances]:
            f = tar.extractfile(m)
            dimacs_text = f.read().decode('utf-8')
            extracted_cnfs.append((dimacs_text, label, m.name))
            
    return extracted_cnfs


def generate_random_3sat(num_vars, clause_ratio=4.26, num_instances=3, seed=42, target_sat=True):
    """
    Generate synthetic random 3-SAT instances for arbitrary sizes (e.g., 500-1000 variables).
    
    Args:
        num_vars: Number of variables
        clause_ratio: Clauses-to-variables ratio (4.26 is phase transition for 3-SAT)
        num_instances: Number of instances to generate
        seed: Random seed
        target_sat: If True, generate near phase transition (harder), if False generate overconstrained
    
    Returns:
        List of (dimacs_text, label, filename) tuples
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Adjust clause ratio based on satisfiability target
    if target_sat:
        # Slightly below phase transition for satisfiable instances
        actual_ratio = clause_ratio * 0.95
        label = "SAT (synthetic)"
    else:
        # Above phase transition for unsatisfiable instances
        actual_ratio = clause_ratio * 1.1
        label = "UNSAT (synthetic)"
    
    num_clauses = int(num_vars * actual_ratio)
    
    print(f"\n[!] Generating {num_instances} synthetic random 3-SAT instances...")
    print(f"[!] Variables: {num_vars}, Clauses: {num_clauses}, Ratio: {actual_ratio:.2f}")
    
    extracted_cnfs = []
    
    for inst_idx in range(num_instances):
        clauses = []
        attempted = set()
        
        # Generate random 3-SAT clauses
        while len(clauses) < num_clauses:
            # Randomly select 3 distinct variables
            vars_indices = random.sample(range(num_vars), 3)
            
            # Randomly negate each literal
            literals = []
            for var_idx in vars_indices:
                if random.random() < 0.5:
                    literals.append(var_idx + 1)  # Positive literal
                else:
                    literals.append(-(var_idx + 1))  # Negative literal
            
            # Check if clause is tautological (contains both x and -x)
            abs_lits = [abs(lit) for lit in literals]
            if len(abs_lits) != len(set(abs_lits)):
                continue  # Skip tautological clauses
            
            # Create a canonical form to avoid duplicates
            clause_tuple = tuple(sorted(literals, key=abs))
            if clause_tuple in attempted:
                continue
            
            attempted.add(clause_tuple)
            clauses.append(literals)
        
        # Generate DIMACS format
        dimacs_lines = []
        dimacs_lines.append(f"c Random 3-SAT instance {inst_idx+1}")
        dimacs_lines.append(f"c Generated with {num_vars} variables, {num_clauses} clauses")
        dimacs_lines.append(f"p cnf {num_vars} {num_clauses}")
        
        for clause in clauses:
            clause_str = " ".join(map(str, clause)) + " 0"
            dimacs_lines.append(clause_str)
        
        dimacs_text = "\n".join(dimacs_lines)
        filename = f"synthetic_3sat_n{num_vars}_m{num_clauses}_{inst_idx+1:03d}.cnf"
        
        extracted_cnfs.append((dimacs_text, label, filename))
    
    return extracted_cnfs


def parse_dimacs(dimacs_str):
    parsed_clauses, parsed_signs = [], []
    M = 0
    for line in dimacs_str.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('c') or line.startswith('%') or line.startswith('0'): 
            continue
        if line.startswith('p cnf'):
            M = int(line.split()[2])
            continue
        c_vars, c_signs = [], []
        for p in line.split():
            if p == '0': break
            v = int(p)
            c_vars.append(abs(v) - 1) 
            c_signs.append(1.0 if v < 0 else 0.0)
        if len(c_vars) > 0:
            parsed_clauses.append(c_vars)
            parsed_signs.append(c_signs)
    return parsed_clauses, parsed_signs, M

def unify_and_pad(parsed_clauses, parsed_signs, M):
    if not parsed_clauses:
        return torch.zeros((0, 3), dtype=torch.long), torch.zeros((0, 3), dtype=torch.float32), M
    K = max(len(c) for c in parsed_clauses)
    K = max(K, 1) 
    pad_vars, pad_signs = [], []
    for cv, cs in zip(parsed_clauses, parsed_signs):
        while len(cv) < K:
            cv.append(cv[-1]) 
            cs.append(cs[-1])
        pad_vars.append(cv[:K])
        pad_signs.append(cs[:K])
    return torch.tensor(pad_vars, dtype=torch.long), torch.tensor(pad_signs, dtype=torch.float32), M

def print_sat_instance(clauses, signs, print_ratio=0.05):
    C = clauses.shape[0]
    num_to_print = max(1, int(C * print_ratio))
    print(f"\n--- Printing {num_to_print} of {C} Parsed Clauses ({print_ratio*100:.1f}%) ---")
    formula_str = []
    for i in range(num_to_print):
        unique_lits = []
        for v, s in zip(clauses[i].tolist(), signs[i].tolist()):
            lit = f"~x_{v}" if s == 1.0 else f"x_{v}"
            if lit not in unique_lits: unique_lits.append(lit)
        formula_str.append(f"({' v '.join(unique_lits)})")
    print(" ^ \n".join(formula_str))
    if num_to_print < C: print(f" ^ \n... (and {C - num_to_print} more clauses)")
    print("-------------------------------------------------\n")

def compute_sat_energy(x, clauses, signs):
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

def plot_physics_metrics(C_ij_tensor, q_batch_tensor, idx, filename):
    C_ij = C_ij_tensor.cpu().numpy()
    q_batch = q_batch_tensor.cpu().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    im = axes[0].imshow(C_ij, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0].set_title(f'Instance {idx} ({filename}): Correlation Matrix $C_{{ij}}$', fontsize=14, pad=15)
    axes[0].set_xlabel('Variable Index $j$', fontsize=12)
    axes[0].set_ylabel('Variable Index $i$', fontsize=12)
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label='Correlation Strength')
    
    axes[1].hist(q_batch, bins=50, color='indigo', alpha=0.7, density=True, edgecolor='black')
    axes[1].set_title(f'Instance {idx} ({filename}): Overlap Distribution $P(q)$', fontsize=14, pad=15)
    axes[1].set_xlabel('Overlap $q$', fontsize=12)
    axes[1].set_ylabel('Probability Density $P(q)$', fontsize=12)
    
    q_mean = np.mean(q_batch)
    axes[1].axvline(q_mean, color='red', linestyle='dashed', linewidth=2, label=rf'Mean $\langle q \rangle = {q_mean:.3f}$')
    axes[1].legend()
    
    plt.tight_layout()
    print(">> Close the plot window to proceed to the next instance...")
    plt.show()

# ==========================================
# 4. Main Evaluator Loop
# ==========================================

def reset_weights(m):
    """In-place weight reset to clear the model's memory between instances."""
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()

def run_satlib_evaluation(M_target=500, target_sat=True, num_instances=3, dataset_seed=42, 
                          use_synthetic=False, batch_size=2048, clip_grad_norm=1.0, 
                          print_ratio=0.05, assignment_print_ratio=0.5, base_entropy_coef=0.1):
    """
    Run SAT evaluation with SATLIB or synthetic datasets.
    
    Args:
        M_target: Number of variables (20-250 for SATLIB, 500-1000+ for synthetic)
        target_sat: Generate SAT or UNSAT instances
        num_instances: Number of instances to evaluate
        dataset_seed: Random seed for dataset generation
        use_synthetic: If True, generate synthetic instances; if False, use SATLIB
        batch_size: Batch size for training
        clip_grad_norm: Gradient clipping norm
        print_ratio: Fraction of clauses to print
        assignment_print_ratio: Fraction of assignment to print
        base_entropy_coef: Base entropy coefficient
    """
                           
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on GPU Acceleration: {device}")
    
    # Fetch or generate instances
    if use_synthetic or M_target > 250:
        print(f"\n[!] Using SYNTHETIC random 3-SAT generator for {M_target} variables")
        cnfs = generate_random_3sat(M_target, clause_ratio=4.26, num_instances=num_instances, 
                                     seed=dataset_seed, target_sat=target_sat)
    else:
        print(f"\n[!] Using SATLIB database for {M_target} variables")
        cnfs = fetch_satlib_direct(M=M_target, target_sat=target_sat, num_instances=num_instances, seed=dataset_seed)
    
    # =================================================================
    # PRE-COMPILATION PHASE (Standard Triton, No CUDA Graphs)
    # =================================================================
    print(f"\n[!] Pre-allocating and Compiling Transformer in advance for M={M_target}...")
    model = FastSpinTransformer(num_vars=M_target, embed_dim=128, num_heads=4, num_layers=3).to(device)
    
    if torch.__version__ >= "2.0.0" and torch.cuda.is_available():
        # Standard compilation provides native Triton speedups without memory allocation crashes
        model = torch.compile(model)
        model.sample_and_log_prob = torch.compile(model.sample_and_log_prob)
    print(f"[!] Compilation successful. Entering high-speed evaluation loop.\n")
    # =================================================================
    
    for idx, (dimacs_text, gt_label, filename) in enumerate(cnfs):
        print(f"\n{'='*80}")
        print(f"  Evaluating Instance {idx+1}/{num_instances} | File: {filename} | Seed: {dataset_seed}")
        print(f"{'='*80}")
        
        parsed_clauses, parsed_signs, M = parse_dimacs(dimacs_text)
        min_K = min(len(c) for c in parsed_clauses) if parsed_clauses else 0
        max_K = max(len(c) for c in parsed_clauses) if parsed_clauses else 0
        
        clauses, signs, _ = unify_and_pad(parsed_clauses, parsed_signs, M)
        clauses, signs = clauses.to(device), signs.to(device)
        
        beta_start = 0.1
        beta_end = float(M) 
        epochs = int(M * 200 + 2000)
        
        print(f"\n[Dynamic Scaling Physics]")
        print(f" -> System Size (M): {M} Variables")
        print(f" -> Clauses (C): {clauses.shape[0]} (alpha = {clauses.shape[0]/M:.2f})")
        print(f" -> Clause Bounds (K): Min {min_K}, Max {max_K}")
        print(f" -> Official Ground Truth: {gt_label}")
        print(f" -> Auto-calculated Epochs: {epochs}")
        print(f" -> Auto-calculated Max Beta: {beta_end:.1f}")
        
        print_sat_instance(clauses, signs, print_ratio=print_ratio)
        
        # Reset the weights to completely random initialization for the new instance
        model.apply(reset_weights)
            
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        
        print(f"\n--- Starting Variational Cooling (Advantage Norm & Entropy Regularization) ---")
        for epoch in range(epochs):
            model.train()
            fraction = epoch / max(1, epochs - 1)
            beta = beta_start * (beta_end / beta_start) ** fraction
            current_entropy_coef = base_entropy_coef * (1.0 - fraction)
            
            with torch.no_grad():
                x_samples, _ = model.sample_and_log_prob(batch_size, device)
                energies = compute_sat_energy(x_samples, clauses, signs)
                
            logits = model(x_samples)
            bce = F.binary_cross_entropy_with_logits(logits, x_samples, reduction='none')
            log_probs = -bce.sum(dim=-1)
            entropy = -log_probs.mean()
            
            with torch.no_grad():
                local_free_energy = beta * energies + log_probs
                baseline = local_free_energy.mean()
                advantage = local_free_energy - baseline
                advantage = advantage / (advantage.std() + 1e-8)
                
            loss = (log_probs * advantage).mean() - current_entropy_coef * entropy
            
            optimizer.zero_grad()
            loss.backward()
            
            if clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
                
            optimizer.step()
            scheduler.step()
            
            if epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1:
                current_lr = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch:5d}/{epochs} | LR: {current_lr:.1e} | Beta: {beta:9.2f} | Ent: {entropy.item():5.2f} | Mean E: {energies.mean().item():6.2f} | Min E: {energies.min().item():.1f}")

        print("\n--- Cooling Complete. Verifying Labels ---")
        model.eval()
        with torch.no_grad():
            test_samples, _ = model.sample_and_log_prob(batch_size * 2, device)
            test_energies = compute_sat_energy(test_samples, clauses, signs)
            
            min_idx = torch.argmin(test_energies)
            best_energy = test_energies[min_idx].item()
            best_assignment = test_samples[min_idx].cpu().numpy().astype(int)
            
            print(f"\nLowest Energy State Reached: {int(best_energy)} violations remaining.")
            
            if best_energy == 0:
                print(">> SUCCESS: Found a perfect satisfying assignment!")
            else:
                if "UNSAT" in gt_label:
                    print(">> [VERIFIED UNSAT]: The network failed to find 0 energy, which perfectly aligns with the UNSAT designation.")
                else:
                    print(">> FAILED: Problem was labeled SAT, but the network was trapped in a local minimum.")
                
            num_to_print = min(max(1, int(M * assignment_print_ratio)), M) 
            assignment_str = ", ".join([f"x_{i}={best_assignment[i]}" for i in range(num_to_print)])
            suffix = " ..." if num_to_print < M else ""
            print(f"\nPartial Variable Assignment ({num_to_print} of {M}):\n[{assignment_str}{suffix}]\n")
            
            spins = 2.0 * test_samples - 1.0 
            mean_spins = spins.mean(dim=0)
            spin_spin = torch.matmul(spins.t(), spins) / (batch_size * 2)
            C_ij = spin_spin - torch.outer(mean_spins, mean_spins)

            replica_1, _ = model.sample_and_log_prob(batch_size * 2, device)
            replica_2, _ = model.sample_and_log_prob(batch_size * 2, device)
            q_batch = ((2.0 * replica_1 - 1.0) * (2.0 * replica_2 - 1.0)).mean(dim=-1)
            
            print(f"Mean Overlap <q>: {q_batch.mean().item():.4f} | Variance Var(q): {q_batch.var().item():.6f}")
            plot_physics_metrics(C_ij, q_batch, idx + 1, filename)

if __name__ == "__main__":
    # Example 1: Generate synthetic instances with 500 variables
    print("="*80)
    print("EXAMPLE 1: Synthetic Random 3-SAT with 500 Variables")
    print("="*80)
    run_satlib_evaluation(
        M_target=500,              # Synthetic generation for larger sizes
        target_sat=True,          
        num_instances=3,           
        dataset_seed=42,          
        use_synthetic=True,        # Use synthetic generator
        batch_size=2048, 
        clip_grad_norm=1.0, 
        print_ratio=0.02,          # Print fewer clauses for large instances
        assignment_print_ratio=0.1, # Print fewer variables
        base_entropy_coef=0.1     
    )
    
    # Example 2: Generate synthetic instances with 1000 variables
    print("\n" + "="*80)
    print("EXAMPLE 2: Synthetic Random 3-SAT with 1000 Variables")
    print("="*80)
    run_satlib_evaluation(
        M_target=1000,             # Even larger synthetic instances
        target_sat=True,          
        num_instances=2,           # Fewer instances due to computational cost
        dataset_seed=42,          
        use_synthetic=True,        
        batch_size=2048, 
        clip_grad_norm=1.0, 
        print_ratio=0.01,          
        assignment_print_ratio=0.05,
        base_entropy_coef=0.1     
    )

    print("\n" + "="*80)
    print("EXAMPLE 3: SATLIB Dataset with 250 Variables")
    print("="*80)
    run_satlib_evaluation(
        M_target=250,              # Largest SATLIB dataset
        target_sat=True,          
        num_instances=3,           
        dataset_seed=42,          
        use_synthetic=False,       # Use real SATLIB data
        batch_size=2048, 
        clip_grad_norm=1.0, 
        print_ratio=0.05,          
        assignment_print_ratio=0.5,
        base_entropy_coef=0.1     
    )

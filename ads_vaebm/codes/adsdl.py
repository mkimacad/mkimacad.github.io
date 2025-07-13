# ==============================================================================
# AdS/VAEBM: but currently without VAE, currently workable parts.
# ==============================================================================
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import random, jit, value_and_grad
from flax.training import train_state
import flax.linen as nn
import optax
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import collections
from functools import partial

# ===== CONFIGURATION ==========================================================
config = {
    'seed': 42, 'num_points_x': 32, 'num_points_z': 16,
    'z_max': 5.0,   'epsilon': 1e-3, 'num_epochs': 300, 'steps_per_epoch': 150,
    'batch_size': 64,  'latent_dim': 16, 'latent_search_steps': 200,
    'peak_lr': 3e-5, 'gradient_clip_value': 1.0, 'generator_width': 256,

    'phi_reconstruction_weight': 1.0,
    'eom_weight': 100.0,
    'ir_weight': 1.0, 'einstein_weight_final': 0.1, 'einstein_weight_initial': 1e-5,
    'stage1_fraction': 0.1, 'stage2_fraction': 0.4,
    'stage3_fraction': 0.5, 'm_bulk_sq': 0.0
}

# ===== CUSTOM TRAIN STATE ========================================
class CustomTrainState(train_state.TrainState):
    pass

# ===== GRIDS & WEIGHTS =====================================================
z_grid = jnp.linspace(config['epsilon'], config['z_max'], config['num_points_z'])
x_grid = jnp.linspace(-1.0, 1.0, config['num_points_x'])
dz, dx = z_grid[1] - z_grid[0], x_grid[1] - x_grid[0]

# --- NEW: Create a weight mask for the EOM loss ---
# This forces the model to prioritize getting the physics right near the
# boundary (small z), and prevents it from settling on trivial solutions
# like a constant field in the bulk (large z).
eom_z_weight = 1.0 / (z_grid**2 + 0.1) # Add a small constant for stability at z=0
eom_z_weight = eom_z_weight[None, :, None] # Reshape for broadcasting with batch

# ===== MODEL ==========================================================
# (No changes)
class CausalGenerator(nn.Module):
    width: int
    @nn.compact
    def __call__(self, z):
        x = nn.swish(nn.Dense(self.width)(z)); x = nn.swish(nn.Dense(self.width)(x))
        phi = nn.Dense(config['num_points_z']*config['num_points_x'])(x)
        phi = phi.reshape((-1, config['num_points_z'], config['num_points_x']))
        flat = phi.reshape((phi.shape[0], -1)); h = nn.swish(nn.Dense(self.width)(flat))
        h = nn.swish(nn.Dense(self.width)(h)); h = nn.Dense(3*config['num_points_z']*config['num_points_x'])(h)
        h = h.reshape((-1,3,config['num_points_z'],config['num_points_x'])); return phi, h[:,0], h[:,1], h[:,2]

# ===== PHYSICS HELPERS ===============================================
# (No changes to most helpers)
@jit
def first_derivatives(field, dx, dz):
    d_dz = (jnp.roll(field, shift=-1, axis=1) - jnp.roll(field, shift=1, axis=1)) / (2 * dz)
    d_dx = (jnp.roll(field, shift=-1, axis=2) - jnp.roll(field, shift=1, axis=2)) / (2 * dx)
    return d_dx, d_dz
@jit
def laplacian(field, dx, dz):
    d2_dz2 = (jnp.roll(field, shift=-1, axis=1) - 2 * field + jnp.roll(field, shift=1, axis=1)) / dz**2
    d2_dx2 = (jnp.roll(field, shift=-1, axis=2) - 2 * field + jnp.roll(field, shift=1, axis=2)) / dx**2
    return d2_dx2 + d2_dz2
def get_full_metric(h_tt, h_zz, h_xx):
    Z = z_grid[None,:,None]; return (-1.0/Z**2 + h_tt), (1.0/Z**2 + h_zz), (1.0/Z**2 + h_xx)
def calculate_energy_momentum_tensor(phi, g_tt, g_zz, g_xx, m_sq):
    dphi_dx, dphi_dz = first_derivatives(phi, dx, dz)
    lag = (1/g_zz)*dphi_dz**2 + (1/g_xx)*dphi_dx**2; T_tt = -0.5 * g_tt * (lag + m_sq * phi**2)
    T_zz = dphi_dz**2 - 0.5 * g_zz * (lag + m_sq * phi**2); T_xx = dphi_dx**2 - 0.5 * g_xx * (lag + m_sq * phi**2)
    return T_tt, T_zz, T_xx
def calculate_einstein_tensor_linearized(h_tt, h_zz, h_xx):
    return -0.5*laplacian(h_tt, dx, dz), -0.5*laplacian(h_zz, dx, dz), -0.5*laplacian(h_xx, dx, dz)
def loss_free_field_ir(ir_slice):
    centered = ir_slice - jnp.mean(ir_slice, axis=0); cov = centered.T @ centered / (ir_slice.shape[0]-1)
    return jnp.mean((cov - jnp.diag(jnp.diag(cov)))**2)
def loss_einstein_true(h_tt,h_zz,h_xx, phi):
    G = calculate_einstein_tensor_linearized(h_tt,h_zz,h_xx)
    T = calculate_energy_momentum_tensor(phi, *get_full_metric(h_tt,h_zz,h_xx), config['m_bulk_sq'])
    mse_norm = lambda g,t: jnp.mean(((g/jnp.sqrt(jnp.mean(g**2)+1e-8)) - (t/jnp.sqrt(jnp.mean(t**2)+1e-8)))**2)
    return sum(mse_norm(Gi, Ti) for Gi,Ti in zip(G,T))

# --- EOM loss now accepts weights ---
def loss_eom(phi, g_tt, g_zz, g_xx, weights):
    lap_phi = laplacian(phi, dx, dz)
    resid = (1/g_zz) * lap_phi - config['m_bulk_sq'] * phi
    # Apply the z-dependent weights before averaging
    return jnp.mean(weights * resid**2)

# ===== ANALYTICAL TEACHER ====================================================
def create_analytical_teacher_batch(key, batch_size):
    def single(k):
        a,b,c,d = random.split(k,4); amp = random.uniform(a, minval=0.8, maxval=2.0); pos = random.uniform(b, minval=-0.5, maxval=0.5)
        sig = random.uniform(c, minval=0.2, maxval=0.4); decay = random.uniform(d, minval=0.5, maxval=1.5)
        X = amp * jnp.exp(-(x_grid[None,:]-pos)**2/(2*sig**2)); Z = jnp.exp(-decay * z_grid[:,None])
        return Z*X
    keys = random.split(key, batch_size); return jax.vmap(single)(keys)

# ===== TRAIN/EVALUATION STEPS =================================================
@jit
def reconstruction_step(state, key, teacher):
    z = random.normal(key, (config['batch_size'], config['latent_dim']))
    def loss_fn(params):
        phi, *_ = state.apply_fn({'params': params}, z)
        phi = phi.at[:,0,:].set(teacher[:,0,:])
        return jnp.mean((phi - teacher)**2)
    loss_val, grads = value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss_val

@partial(jit, static_argnames=['stage'])
def train_step(state, key, teacher, stage, ein_w):
    z = random.normal(key, (config['batch_size'], config['latent_dim']))
    def loss_fn(params):
        phi, h_tt, h_zz, h_xx = state.apply_fn({'params': params}, z)
        phi = phi.at[:,0,:].set(teacher[:,0,:])
        g_tt, g_zz, g_xx = get_full_metric(h_tt, h_zz, h_xx)

        # --- LOSS CALCULATIONS (EOM is now weighted) ---
        l_rec = jnp.mean((phi - teacher)**2)
        l_eom = loss_eom(phi, g_tt, g_zz, g_xx, eom_z_weight) # Pass weights here

        total_loss = (config['phi_reconstruction_weight'] * l_rec +
                      config['eom_weight'] * l_eom)

        l_ein, l_ir = 0.0, 0.0
        if stage == 3:
            l_ein = loss_einstein_true(h_tt, h_zz, h_xx, phi)
            l_ir = loss_free_field_ir(phi[:, -1, :])
            total_loss += ein_w * l_ein + config['ir_weight'] * l_ir

        metrics = {'rec': l_rec, 'eom': l_eom, 'ein': l_ein, 'ir': l_ir}
        return total_loss, metrics

    (loss, metrics), grads = value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, metrics

# ===== MAIN LOOP ============================================================
if __name__ == '__main__':
    key = random.PRNGKey(config['seed'])
    model = CausalGenerator(config['generator_width'])
    params = model.init(key, jnp.ones((1,config['latent_dim'])))['params']
    total_steps = config['num_epochs'] * config['steps_per_epoch']
    sched = optax.warmup_cosine_decay_schedule(0.0, config['peak_lr'], int(0.1 * total_steps), int(0.9 * total_steps))
    tx = optax.chain(optax.clip_by_global_norm(config['gradient_clip_value']), optax.adamw(sched))
    state = CustomTrainState.create(apply_fn=model.apply, params=params, tx=tx)

    s1_epochs, s2_epochs = int(config['stage1_fraction'] * config['num_epochs']), int(config['stage2_fraction'] * config['num_epochs'])
    s3_epochs = config['num_epochs'] - s1_epochs - s2_epochs

    print(f"--- Starting Stage 1: Supervised Reconstruction ({s1_epochs} epochs) ---")
    for epoch in range(s1_epochs):
        metrics = collections.defaultdict(float)
        pbar = tqdm(range(config['steps_per_epoch']), desc=f"S1 Epoch {epoch+1}/{s1_epochs}")
        for _ in pbar:
            key, sub = random.split(key); teacher = create_analytical_teacher_batch(sub, config['batch_size'])
            state, l_rec = reconstruction_step(state, sub, teacher)
            metrics['rec'] += l_rec / config['steps_per_epoch']
            pbar.set_postfix({'rec': f"{metrics['rec']:.6f}"})

    print(f"\n--- Starting Stage 2: Fast Adaptive EOM Training ({s2_epochs} epochs) ---")
    for epoch in range(s2_epochs):
        metrics = collections.defaultdict(float)
        pbar = tqdm(range(config['steps_per_epoch']), desc=f"S2 Epoch {epoch+1}/{s2_epochs}")
        for _ in pbar:
            key, sub = random.split(key); teacher = create_analytical_teacher_batch(sub, config['batch_size'])
            state, mets = train_step(state, sub, teacher, stage=2, ein_w=0.0)
            for k,v in mets.items(): metrics[k] += v/config['steps_per_epoch']
            pbar.set_postfix({'rec': f"{metrics['rec']:.5f}", 'eom': f"{metrics['eom']:.2e}"})

    print(f"\n--- Starting Stage 3: Full Physics Training ({s3_epochs} epochs) ---")
    for epoch in range(s3_epochs):
        anneal_prog = min(1.0, epoch / (s3_epochs * 0.9))
        ein_w = config['einstein_weight_initial'] + (config['einstein_weight_final'] - config['einstein_weight_initial']) * anneal_prog
        metrics = collections.defaultdict(float)
        pbar = tqdm(range(config['steps_per_epoch']), desc=f"S3 Epoch {epoch+1}/{s3_epochs}")
        for _ in pbar:
            key, sub = random.split(key); teacher = create_analytical_teacher_batch(sub, config['batch_size'])
            state, mets = train_step(state, sub, teacher, stage=3, ein_w=ein_w)
            for k,v in mets.items(): metrics[k] += v/config['steps_per_epoch']
            pbar.set_postfix({
                'rec': f"{metrics['rec']:.4f}", 'eom': f"{metrics['eom']:.2e}",
                'ein': f"{metrics['ein']:.4f}"
            })

    print("\nTraining complete.")

    # --- ANALYSIS ---
    print("Analyzing final model...")
    key, tkey = jax.random.split(key)
    phi_t = create_analytical_teacher_batch(tkey, 1)[0]
    def find_latent(state, teacher):
        z = jnp.zeros((1,config['latent_dim']),dtype=jnp.float64)
        opt_z = optax.adam(0.05)
        opt_state = opt_z.init(z)
        @jax.jit
        def loss_fn(z): phi, *_ = state.apply_fn({'params':state.params}, z); return jnp.mean((phi[0] - teacher)**2)
        @jax.jit
        def step(z, opt_state):
            loss, grads = jax.value_and_grad(loss_fn)(z)
            updates, opt_state = opt_z.update(grads, opt_state)
            z = optax.apply_updates(z, updates)
            return z, opt_state, loss
        for _ in tqdm(range(config['latent_search_steps']), desc="Latent search"): z, opt_state, _ = step(z, opt_state)
        return z
    z_star = find_latent(state, phi_t)
    phi_model, h_tt, h_zz, h_xx = state.apply_fn({'params':state.params}, z_star)
    phi_model = phi_model[0]
    fig, axs = plt.subplots(1, 5, figsize=(30,5))
    im0 = axs[0].imshow(phi_t, origin='lower', extent=[x_grid[0],x_grid[-1],z_grid[0],z_grid[-1]], aspect='auto')
    axs[0].set_title('Teacher φ'); fig.colorbar(im0, ax=axs[0])
    im1 = axs[1].imshow(phi_model, origin='lower', extent=[x_grid[0],x_grid[-1],z_grid[0],z_grid[-1]], aspect='auto')
    axs[1].set_title('Model φ'); fig.colorbar(im1, ax=axs[1])
    for i,(h,title) in enumerate([(h_tt,'h_tt'),(h_zz,'h_zz'),(h_xx,'h_xx')], start=2):
        im = axs[i].imshow(h[0], origin='lower', extent=[x_grid[0],x_grid[-1],z_grid[0],z_grid[-1]], aspect='auto', cmap='RdBu_r')
        axs[i].set_title(title); fig.colorbar(im, ax=axs[i])
    plt.tight_layout(); plt.show()

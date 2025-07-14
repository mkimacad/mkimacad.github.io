# ===================================================================
# --- AdSP/CFTP: no longer AdS/VAEBM. VAE no longer needed ---
# --- Still matches UV and IR partition functions by scoring ---
# --- Plots still missing ---
# ===================================================================
# --- Part 1: Setup and Dependencies ---
# ===================================================================
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state
from dataclasses import dataclass, field
from functools import partial
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

# ===================================================================
# --- Part 2: Tunable Configuration ---
# ===================================================================
@dataclass(frozen=True)
class HolographicConfig:
    """A single, comprehensive configuration for the entire experiment."""
    # --- Physics Parameters ---
    n_x: int = 64; n_t: int = 64; n_z: int = 100; z_ir: float = 10.0
    num_ir_params: int = 1
    # --- Optimizer Settings ---
    learning_rate: float = 3e-4; gradient_clip_norm: float = 1.0
    # --- Loss Weights (Used only if dynamic_weighting is False) ---
    score_loss_weight: float = 1.0; action_min_weight: float = 0.1
    path_smoothness_weight: float = 1.0; bulk_amplitude_weight: float = 0.1
    perturb_regularization: float = 0.01
    # --- Network Architecture ---
    action_nn_features: int = 128; action_nn_depth: int = 2
    path_solver_features: int = 256; path_solver_depth: int = 3
    # --- Training Parameters ---
    num_epochs: int = 10000; batch_size: int = 32
    # --- Toggles for Advanced Training Techniques ---
    use_dynamic_weighting: bool = True
    use_lr_scheduler: bool = True
    # --- Hyperparameters for Advanced Techniques ---
    dynamic_weighting_ema_alpha: float = 0.99
    lr_warmup_epochs: int = 500

# ===================================================================
# --- Part 3: Physical System Definition ---
# ===================================================================
@partial(jax.jit, static_argnums=(0, 2))
def generate_cft_samples(config: HolographicConfig, key: jax.random.PRNGKey, batch_size: int):
    shape = (batch_size, config.n_x, config.n_t); kx = jnp.fft.fftfreq(config.n_x) * config.n_x; kt = jnp.fft.fftfreq(config.n_t) * config.n_t
    k_grid_x, k_grid_t = jnp.meshgrid(kx, kt, indexing='ij'); k_squared = k_grid_x**2 + k_grid_t**2
    k_squared = k_squared.at[0, 0].set(1.0); propagator = 1.0 / jnp.sqrt(k_squared)
    propagator = propagator.at[0, 0].set(0.0); white_noise = jax.random.normal(key, shape)
    f_white_noise = jnp.fft.fft2(white_noise, axes=(1, 2)); f_correlated_noise = f_white_noise * propagator
    correlated_samples = jnp.fft.ifft2(f_correlated_noise, axes=(1, 2)).real
    return correlated_samples
def free_scalar_action(phi, dx, dt):
    grad_x_phi, grad_t_phi = jnp.gradient(phi, dx, dt); integrand = 0.5 * (grad_x_phi**2 + grad_t_phi**2); return jnp.sum(integrand * dx * dt)
def get_target_theories(config: HolographicConfig):
    dx = 1.0 / config.n_x; dt = 1.0 / config.n_t; cft_action_fn = partial(free_scalar_action, dx=dx, dt=dt); cft_score_fn = jax.grad(cft_action_fn)
    def ir_perturbed_action_fn(phi, ir_params):
        mass_term = ir_params[0] * 0.5 * jnp.sum(phi**2) * dx * dt; return cft_action_fn(phi) + mass_term
    ir_perturbed_score_fn = jax.grad(ir_perturbed_action_fn); return cft_action_fn, cft_score_fn, ir_perturbed_action_fn, ir_perturbed_score_fn

# ===================================================================
# --- Part 4: ResNet-based Neural Network Architecture ---
# ===================================================================
class ResNetBlock(nn.Module):
    features: int
    @nn.compact
    def __call__(self, x):
        y = nn.Dense(features=self.features)(x); y = nn.relu(y)
        y = nn.Dense(features=self.features)(y)
        return x + y

class BulkActionNN(nn.Module):
    config: HolographicConfig
    @nn.compact
    def __call__(self, phi1, phi2, ir_params):
        input_vec = jnp.concatenate([phi1.ravel(), phi2.ravel(), ir_params])
        x = nn.Dense(features=self.config.action_nn_features)(input_vec); x = nn.relu(x)
        for _ in range(self.config.action_nn_depth): x = ResNetBlock(features=self.config.action_nn_features)(x)
        x = nn.relu(x); x = nn.Dense(features=1)(x)
        return nn.softplus(x.squeeze())

class ClassicalPathSolverNN(nn.Module):
    config: HolographicConfig
    @nn.compact
    def __call__(self, z_coords, phi_uv, phi_ir):
        def generate_slice(z, phi_uv, phi_ir):
            z_norm = z / self.config.z_ir
            linear_path = (1 - z_norm) * phi_uv + z_norm * phi_ir
            z_embedding = nn.Dense(features=self.config.path_solver_features // 4)(jnp.array([z_norm, 1-z_norm, z_norm**2, z_norm*(1-z_norm)]))
            boundary_embedding = nn.Dense(features=self.config.path_solver_features)(jnp.concatenate([phi_uv.ravel(), phi_ir.ravel()]))
            initial_vec = jnp.concatenate([z_embedding, boundary_embedding])
            x = nn.Dense(features=self.config.path_solver_features)(initial_vec); x = nn.relu(x)
            for _ in range(self.config.path_solver_depth): x = ResNetBlock(features=self.config.path_solver_features)(x)
            x = nn.relu(x)
            deviation = nn.Dense(features=phi_uv.size)(x).reshape(phi_uv.shape)
            return linear_path + z * (1 - z / self.config.z_ir) * deviation
        return jax.vmap(generate_slice, in_axes=(0, None, None))(z_coords, phi_uv, phi_ir)

class HolographicModel(nn.Module):
    config: HolographicConfig; cft_action_fn: callable
    def setup(self):
        self.action_nn = BulkActionNN(self.config, name="action_net")
        self.path_solver = ClassicalPathSolverNN(self.config, name="path_solver")
    @nn.compact
    def __call__(self, phi_uv, phi_ir, ir_params):
        z_coords = jnp.linspace(0, self.config.z_ir, self.config.n_z + 1); phi_cl_path = self.path_solver(z_coords, phi_uv, phi_ir)
        def single_step_action(phi_i, phi_i_plus_1):
            original_action = self.cft_action_fn((phi_i + phi_i_plus_1) / 2.0); learned_perturbation = self.action_nn(phi_i, phi_i_plus_1, ir_params)
            return original_action + learned_perturbation
        total_action = jnp.sum(jax.vmap(single_step_action)(phi_cl_path[:-1], phi_cl_path[1:])); return total_action, phi_cl_path

# ===================================================================
# --- Part 5: Loss Function and Training State ---
# ===================================================================
class CustomTrainState(train_state.TrainState):
    loss_emas: dict = field(default_factory=dict)

def create_loss_fn(config: HolographicConfig, model: nn.Module, target_fns: tuple):
    cft_action_fn, cft_score_fn, ir_perturbed_action_fn, ir_perturbed_score_fn = target_fns
    get_total_action_and_path_fn = lambda params, phi_uv, phi_ir, ir_params: model.apply({'params': params}, phi_uv, phi_ir, ir_params)
    get_action_for_grad = lambda *args: get_total_action_and_path_fn(*args)[0]
    total_action_scorer = jax.grad(get_action_for_grad, argnums=(1, 2))
    score_loss_normalizer = (config.n_x)**4
    def loss_fn(params, batch, loss_emas):
        phi_uv_batch, phi_ir_batch, ir_params_batch = batch
        total_actions, phi_cl_paths = jax.vmap(get_total_action_and_path_fn, in_axes=(None, 0, 0, 0))(params, phi_uv_batch, phi_ir_batch, ir_params_batch)
        model_scores_uv, model_scores_ir = jax.vmap(total_action_scorer, in_axes=(None, 0, 0, 0))(params, phi_uv_batch, phi_ir_batch, ir_params_batch)
        target_scores_uv = jax.vmap(cft_score_fn)(phi_uv_batch); target_scores_ir = jax.vmap(ir_perturbed_score_fn)(phi_ir_batch, ir_params_batch)
        loss_score = (jnp.mean(jnp.sum((model_scores_uv - target_scores_uv)**2, axis=(1, 2))) + jnp.mean(jnp.sum((model_scores_ir - target_scores_ir)**2, axis=(1, 2)))) / score_loss_normalizer
        loss_action = jnp.mean(total_actions)
        loss_smoothness = jnp.mean((phi_cl_paths[:, :-1, :, :] - phi_cl_paths[:, 1:, :, :])**2)
        loss_amplitude = jnp.mean(phi_cl_paths**2)
        def get_single_perturbation(phi1, phi2, ir_params):
            return model.apply({'params': params}, phi1, phi2, ir_params, method=lambda mdl, p1, p2, p_ir: mdl.action_nn(p1, p2, p_ir))
        loss_reg = jnp.mean(jax.vmap(get_single_perturbation)(jnp.zeros_like(phi_uv_batch), jnp.zeros_like(phi_ir_batch), jnp.zeros_like(ir_params_batch))**2)
        unweighted_losses = {'score': loss_score, 'action': loss_action, 'smoothness': loss_smoothness, 'amplitude': loss_amplitude, 'reg': loss_reg}
        if config.use_dynamic_weighting:
            new_emas = {k: config.dynamic_weighting_ema_alpha * loss_emas[k] + (1 - config.dynamic_weighting_ema_alpha) * unweighted_losses[k] for k in unweighted_losses}
            weights = {k: 1.0 / (new_emas[k] + 1e-8) for k in new_emas}; norm_factor = sum(weights.values()); final_weights = {k: v / norm_factor for k, v in weights.items()}
            total_loss = sum(final_weights[k] * unweighted_losses[k] for k in unweighted_losses)
        else:
            weights = {'score': config.score_loss_weight, 'action': config.action_min_weight, 'smoothness': config.path_smoothness_weight, 'amplitude': config.bulk_amplitude_weight, 'reg': config.perturb_regularization}
            total_loss = sum(weights[k] * unweighted_losses[k] for k in unweighted_losses); new_emas = loss_emas
        return total_loss, {"unweighted": unweighted_losses, "updated_emas": new_emas}
    return loss_fn

# ===================================================================
# --- Part 6: Main Execution Block ---
# ===================================================================
if __name__ == '__main__':
    config = HolographicConfig(
        n_z=100, z_ir=10.0, action_nn_features=256, action_nn_depth=3,
        path_solver_features=512, path_solver_depth=4, num_epochs=10000,
        use_dynamic_weighting=True, use_lr_scheduler=True, learning_rate=3e-4,
        score_loss_weight=1.0, action_min_weight=1.0, path_smoothness_weight=1.0,
        bulk_amplitude_weight=1.0, perturb_regularization=1.0
    )
    print(f"--- Running Experiment with Config: ---\n{config}\n")
    key = jax.random.PRNGKey(42)
    target_fns = get_target_theories(config)
    cft_action_fn, _, _, _ = target_fns
    model = HolographicModel(config, cft_action_fn=cft_action_fn)
    key, init_key = jax.random.split(key)
    dummy_phi = jnp.zeros((config.n_x, config.n_t)); dummy_ir_params = jnp.zeros((config.num_ir_params,))
    params = model.init(init_key, dummy_phi, dummy_phi, dummy_ir_params)['params']
    if config.use_lr_scheduler:
        print("Using learning rate scheduler with warmup.")
        lr_schedule = optax.warmup_cosine_decay_schedule(init_value=0.0, peak_value=config.learning_rate, warmup_steps=config.lr_warmup_epochs, decay_steps=config.num_epochs - config.lr_warmup_epochs, end_value=config.learning_rate / 10.0)
        optimizer = optax.chain(optax.clip_by_global_norm(config.gradient_clip_norm), optax.adamw(lr_schedule))
    else:
        print("Using fixed learning rate.")
        optimizer = optax.chain(optax.clip_by_global_norm(config.gradient_clip_norm), optax.adamw(config.learning_rate))
    initial_emas = {'score': 1.0, 'action': 1.0, 'smoothness': 1.0, 'amplitude': 1.0, 'reg': 1.0}
    state = CustomTrainState.create(apply_fn=model.apply, params=params, tx=optimizer, loss_emas=initial_emas)
    loss_calculator = create_loss_fn(config, model, target_fns)
    @jax.jit
    def train_step(state, batch):
        (loss, aux), grads = jax.value_and_grad(loss_calculator, argnums=0, has_aux=True)(state.params, batch, state.loss_emas)
        state = state.apply_gradients(grads=grads)
        state = state.replace(loss_emas=aux['updated_emas'])
        metrics = {'total_loss': loss, **aux['unweighted']}
        return state, metrics
    print("Starting training...")
    history = {k: [] for k in ['total_loss', 'score', 'action', 'smoothness', 'amplitude', 'reg']}
    pbar = tqdm(range(config.num_epochs), desc="Training Progress")
    for epoch in pbar:
        key, uv_key, ir_key, params_key = jax.random.split(key, 4)
        phi_uv_batch = generate_cft_samples(config, uv_key, config.batch_size)
        phi_ir_batch = generate_cft_samples(config, ir_key, config.batch_size)
        ir_params_batch = jax.random.uniform(params_key, (config.batch_size, config.num_ir_params))
        batch = (phi_uv_batch, phi_ir_batch, ir_params_batch)
        state, metrics = train_step(state, batch)
        for k in history.keys(): history[k].append(metrics[k])
        if (epoch + 1) % 200 == 0:
            pbar.set_postfix({'Loss': f"{metrics['total_loss']:.3f}", 'Score': f"{metrics['score']:.4f}", 'Action': f"{metrics['action']:.2f}", 'Amp': f"{metrics['amplitude']:.4f}"})
    print("Training finished.")
    # (Verification and Visualization part is unchanged)
    print("\n--- Running Verification and Visualization ---")
    trained_params = state.params; key, uv_key, ir_key = jax.random.split(key, 3)
    test_phi_uv = generate_cft_samples(config, uv_key, 1)[0]; test_phi_ir = generate_cft_samples(config, ir_key, 1)[0]
    print("\n[1] Verifying the p=0 (unperturbed) case...")
    p_zero = jnp.zeros((config.num_ir_params,)); _, path_p_zero = model.apply({'params': trained_params}, test_phi_uv, test_phi_ir, p_zero)
    z_coords = jnp.linspace(0, config.z_ir, config.n_z + 1)
    ideal_path = jax.vmap(lambda z: (1 - z/config.z_ir) * test_phi_uv + (z/config.z_ir) * test_phi_ir)(z_coords)
    p0_path_deviation = jnp.mean((path_p_zero - ideal_path)**2)
    print(f"--> Deviation from ideal straight-line path: {p0_path_deviation:.6f}")
    if p0_path_deviation < 1e-3: print("--> SUCCESS: The learned path for p=0 is a nearly perfect straight line.")
    else: print("--> WARNING: The path for p=0 deviates significantly from a straight line.")
    print("\n[2] Visualizing the RG flow for p > 0...")
    p_one = jnp.ones((config.num_ir_params,)); _, path_p_one = model.apply({'params': trained_params}, test_phi_uv, test_phi_ir, p_one)
    indices_to_plot = jnp.array([0, config.n_z // 3, 2 * config.n_z // 3, config.n_z]); slices_to_plot = path_p_one[indices_to_plot]
    vmin = slices_to_plot.min(); vmax = slices_to_plot.max(); fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    for i, slice_idx in enumerate(indices_to_plot):
        ax = axes[i]; im = ax.imshow(slices_to_plot[i], cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
        z_val = z_coords[slice_idx]; ax.set_title(f"RG Flow at z = {z_val:.2f}"); ax.set_xlabel("x"); ax.set_ylabel("t")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6); plt.suptitle("Holographic RG Flow: High-frequency modes are smoothed out from UV (z=0) to IR", fontsize=16); plt.show()
    print("\n[3] Probing the bulk response to the perturbation parameter 'p'...")
    p_values = jnp.linspace(0, 2.0, 20)
    @jax.jit
    def get_s_perturb(params, phi_uv, phi_ir, p_val):
        _, path = model.apply({'params': params}, phi_uv, phi_ir, p_val)
        def get_step_perturb(phi1, phi2):
            return model.apply({'params': params}, phi1, phi2, p_val, method=lambda mdl, p1, p2, p_ir: mdl.action_nn(p1, p2, p_ir))
        return jnp.sum(jax.vmap(get_step_perturb)(path[:-1], path[1:]))
    s_perturb_values = []
    for p_val in tqdm(p_values, desc="Probing geometry"):
        s_perturb_values.append(get_s_perturb(trained_params, test_phi_uv, test_phi_ir, jnp.array([p_val])))
    plt.figure(figsize=(8, 6)); plt.plot(p_values, s_perturb_values, 'o-'); plt.xlabel("Perturbation strength 'p' (squared mass m²)"); plt.ylabel("Total Perturbation Action S_perturb"); plt.title("Emergent Bulk Geometry Response"); plt.grid(True); plt.show()

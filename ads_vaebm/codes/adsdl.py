# =================================================================== #
# --- AdS/PINN: no longer AdS/VAEBM. VAE no longer needed ---
# --- Still matches UV and IR partition functions by scoring ---
# =================================================================== #
import os
import sys
import jax
import jax.numpy as jnp
from jax import tree_util as jax_tree
import flax.linen as nn
import optax
from flax.training import train_state, checkpoints
from flax.traverse_util import path_aware_map
from dataclasses import dataclass, field
from functools import partial
from tqdm.notebook import tqdm
from typing import Tuple, Any, Dict
import matplotlib.pyplot as plt

# ---------------- CONFIGURATION (with Training Rounds) ----------------
@dataclass(frozen=True)
class HolographicConfig:
    # New parameter to control training repetitions
    num_training_rounds: int = 2

    n_x: int = 30
    n_t: int = 30
    n_z: int = 10
    z_ir: float = 10.0
    num_ir_params: int = 1

    gradient_clip_norm: float = 1.0
    
    pretrain_p_steps: int = 5
    pretrain_epochs_per_p: int = 2000
    
    finetune_p_steps: int = 10
    finetune_epochs_per_p: int = 5000
    
    pretrain_path_solver_lr: float = 5e-4
    pretrain_action_net_lr: float = 1e-5
    
    finetune_path_solver_lr: float = 5e-5 
    finetune_action_net_lr: float = 1e-6

    score_weight: float = 1.0
    action_weight: float = 0.0
    dev_smoothness_weight: float = 0.01
    dev_amplitude_weight: float = 0.00
    p0_regularization_weight: float = 0.05

    path_solver_features: int = 768
    path_solver_depth: int = 6
    path_cnn_features: Tuple[int, ...] = field(default_factory=lambda: (24, 48, 96, 192))
    
    action_nn_features: int = 256
    action_nn_depth: int = 3
    action_cnn_features: Tuple[int, ...] = field(default_factory=lambda: (16, 32))
    cnn_kernel_size: int = 5

    batch_size: int = 8
    training_p_max: float = 10.0
    log_frequency: int = 200
    checkpoint_base_dir: str = 'checkpoints'
    plot_base_dir: str = 'plots'
    validation_seed: int = 0

# ---------------- INITIALIZERS / GENERATOR / THEORIES (No changes) ----------------
def small_kick_init(key, shape, dtype=jnp.float32): return jax.random.normal(key, shape, dtype) * 1e-2
class CFTSampleGenerator:
    def __init__(self, config: HolographicConfig):
        self.config = config; kx = jnp.fft.fftfreq(config.n_x) * config.n_x; kt = jnp.fft.fftfreq(config.n_t) * config.n_t
        kx, kt = jnp.meshgrid(kx, kt, indexing='ij'); k2 = kx**2 + kt**2; k2 = k2.at[0,0].set(1.0)
        propagator = 1.0 / jnp.sqrt(k2); self.propagator = propagator.at[0,0].set(0.0)
        self._gen = jax.jit(self._gen_impl, static_argnums=1)
    def _gen_impl(self, key, batch_size: int):
        shape = (batch_size, self.config.n_x, self.config.n_t); wn = jax.random.normal(key, shape)
        f = jnp.fft.fft2(wn, axes=(1,2)) * self.propagator; return jnp.fft.ifft2(f, axes=(1,2)).real
    def generate(self, key, batch_size): return self._gen(key, batch_size)
@jax.jit
def free_scalar_action(phi, dx, dt): grad_x, grad_t = jnp.gradient(phi, dx, dt); return jnp.sum(0.5 * (grad_x**2 + grad_t**2)) * dx * dt
def get_target_theories(cfg: HolographicConfig):
    dx, dt = 1.0/cfg.n_x, 1.0/cfg.n_t; cft_act = partial(free_scalar_action, dx=dx, dt=dt)
    cft_score = jax.jit(jax.grad(lambda p: cft_act(p)))
    @jax.jit
    def ir_act(phi, ir): return cft_act(phi) + ir[0] * 0.5 * jnp.sum(phi**2) * dx * dt
    ir_score = jax.jit(jax.grad(ir_act, argnums=0)); return cft_act, cft_score, ir_act, ir_score

# ---------------- MODEL DEFINITION (No changes) ----------------
class ActionPerturbationCNN(nn.Module):
    cfg: HolographicConfig
    @nn.compact
    def __call__(self, p1, p2, ir):
        x = jnp.stack([p1, p2], -1)
        for i, f in enumerate(self.cfg.action_cnn_features):
            x = nn.Conv(f, (self.cfg.cnn_kernel_size,)*2, padding='SAME')(x); x = nn.gelu(x)
            if i < len(self.cfg.action_cnn_features)-1: x = nn.avg_pool(x, (2,2), (2,2))
        x = jnp.mean(x, axis=(1,2)); v = jnp.concatenate([x, ir], -1)
        for _ in range(self.cfg.action_nn_depth): v = nn.Dense(self.cfg.action_nn_features)(v); v = nn.gelu(v)
        return nn.Dense(1, kernel_init=small_kick_init)(v).squeeze()
class ResNetBlock(nn.Module):
    features: int
    @nn.compact
    def __call__(self, x):
        y = nn.GroupNorm(min(32, self.features))(x); y = nn.gelu(y); y = nn.Dense(self.features)(y)
        y = nn.GroupNorm(min(32, self.features))(y); y = nn.gelu(y); y = nn.Dense(self.features)(y)
        return x + y
class PathSolverCNN(nn.Module):
    cfg: HolographicConfig
    @nn.compact
    def __call__(self, phi):
        x = phi[..., None]
        for f in self.cfg.path_cnn_features:
            x = nn.Conv(f, (self.cfg.cnn_kernel_size,)*2, strides=(2,2))(x); x = nn.GroupNorm(min(8, f))(x); x = nn.gelu(x)
        return x.ravel()
class StabilizedClassicalPathSolverNN(nn.Module):
    cfg: HolographicConfig
    @nn.compact
    def __call__(self, z_coords, p_uv, p_ir, ir):
        encoder = PathSolverCNN(self.cfg); eu = encoder(p_uv); ei = encoder(p_ir); be = jnp.concatenate([eu, ei], -1)
        zn = z_coords / self.cfg.z_ir; zf = jnp.stack([zn, 1-zn, zn**2, zn*(1-zn)], -1)
        ze = nn.Dense(self.cfg.path_solver_features // 4, name="z_embedding")(zf)
        num_z_pts = z_coords.shape[0]; be_b = jnp.broadcast_to(be, (num_z_pts, be.shape[-1])); ir_b = jnp.broadcast_to(ir, (num_z_pts, ir.shape[-1]))
        v0 = jnp.concatenate([ze, be_b, ir_b], -1)
        v = nn.Dense(self.cfg.path_solver_features, name="input_dense")(v0); v = nn.gelu(v)
        for i in range(self.cfg.path_solver_depth): v = ResNetBlock(self.cfg.path_solver_features, name=f"resnet_{i}")(v)
        v = nn.gelu(v); dev = nn.Dense(p_uv.size, kernel_init=small_kick_init, name="output_dense")(v)
        dev = dev.reshape((num_z_pts,) + p_uv.shape); zn_reshaped = zn.reshape(-1, 1, 1)
        path_envelope = self.cfg.z_ir * zn_reshaped * (1 - zn_reshaped)
        path = (1 - zn_reshaped) * p_uv + zn_reshaped * p_ir + path_envelope * dev; return path, dev
class HolographicModel(nn.Module):
    cfg: HolographicConfig; cft_act: Any
    def setup(self): self.anet = ActionPerturbationCNN(self.cfg); self.psol = StabilizedClassicalPathSolverNN(self.cfg)
    def get_action_pert(self, p1, p2, ir): return self.anet(p1, p2, ir)
    @nn.compact
    def __call__(self, puv, pir, ir, use_path=True) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        z = jnp.linspace(0, self.cfg.z_ir, self.cfg.n_z + 1)
        if use_path: path, dev = self.psol(z, puv, pir, ir)
        else: path = (1-z[:,None,None]/self.cfg.z_ir)*puv + (z[:,None,None]/self.cfg.z_ir)*pir; dev = jnp.zeros_like(path)
        def step(a, b): c_act = self.cft_act((a + b) / 2); p_act = self.anet(a, b, ir); return c_act + p_act, p_act
        tot_acts, pert_acts = jax.vmap(step)(path[:-1], path[1:])
        return jnp.sum(tot_acts), path, jnp.sum(pert_acts), dev

# ---------------- LOSS & SCALING (No changes) ----------------
def create_loss_fn(cfg: HolographicConfig, model: nn.Module, fns: tuple):
    _, cft_s, _, ir_s = fns; norm = float(cfg.n_x * cfg.n_t)
    def loss_fn(params, batch, loss_scales: Dict[str, float]) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        puv, pir, ir = batch
        def model_apply_fn(p, u, v, z): return model.apply({'params': p}, u, v, z, use_path=True)
        _, _, pert_actions, devs = jax.vmap(model_apply_fn, in_axes=(None, 0, 0, 0))(params, puv, pir, ir)
        grad_target_fn = lambda p, u, v, z: model_apply_fn(p, u, v, z)[0]
        ms_uv = jax.vmap(jax.grad(grad_target_fn, argnums=1), in_axes=(None, 0, 0, 0))(params, puv, pir, ir)
        ms_ir = jax.vmap(jax.grad(grad_target_fn, argnums=2), in_axes=(None, 0, 0, 0))(params, puv, pir, ir)
        ts_uv = jax.vmap(cft_s)(puv); ts_ir = jax.vmap(ir_s)(pir, ir)
        def p0_pert_fn(p, u, v): return model.apply({'params': p}, u, v, jnp.zeros((cfg.num_ir_params,)), method=HolographicModel.get_action_pert)
        p0_perts = jax.vmap(p0_pert_fn, in_axes=(None, 0, 0))(params, puv, pir)
        losses = {
            'score': (jnp.mean(jnp.sum((ms_uv-ts_uv)**2, (1, 2))) + jnp.mean(jnp.sum((ms_ir-ts_ir)**2, (1, 2)))) / norm,
            'p0_reg': jnp.mean(p0_perts**2), 'pert_action': jnp.mean(pert_actions**2),
            'dev_smoothness': jnp.mean((devs[:, :-1] - devs[:, 1:])**2), 'dev_amplitude': jnp.mean(devs**2)
        }
        return sum(loss_scales[k] * v for k, v in losses.items()), losses
    return loss_fn
def calculate_gradient_scales(cfg, loss_fn, params, batch, is_finetune=False):
    base_weights = {
        'score': cfg.score_weight, 'p0_reg': cfg.p0_regularization_weight, 'pert_action': cfg.action_weight, 
        'dev_smoothness': cfg.dev_smoothness_weight, 'dev_amplitude': cfg.dev_amplitude_weight
    }
    if not is_finetune:
        print("Using score-only loss for pre-training.")
        return {k: (1.0 if k == 'score' else 0.0) for k in base_weights}
    
    print("Calculating gradient scales for fine-tuning loss balancing...")
    loss_scales = {}
    for loss_name in base_weights:
        scales_for_one_loss = {k: (1.0 if k == loss_name else 0.0) for k in base_weights}
        if base_weights[loss_name] == 0:
            loss_scales[loss_name] = 0.0; print(f"  - Grad norm for '{loss_name}': SKIPPED (weight is zero)"); continue
        grad_fn = jax.grad(lambda p, b: loss_fn(p, b, scales_for_one_loss)[0])
        grads = grad_fn(params, batch)
        total_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax_tree.tree_leaves(grads)))
        loss_scales[loss_name] = base_weights[loss_name] / (total_norm + 1e-8)
        print(f"  - Grad norm for '{loss_name}': {total_norm:.4f}, final scale: {loss_scales[loss_name]:.4f}")
    return loss_scales

# ---------------- ENVIRONMENT SETUP (No changes) ----------------
def setup_environment(cfg):
    in_colab = 'google.colab' in sys.modules
    if in_colab:
        print("--- Detected Google Colab environment. Mounting Google Drive. ---")
        from google.colab import drive
        drive.mount('/content/drive')
        gdrive_path = '/content/drive/MyDrive/holographic_model'
        checkpoint_dir = os.path.join(gdrive_path, cfg.checkpoint_base_dir)
        plot_dir = os.path.join(gdrive_path, cfg.plot_base_dir)
    else:
        print("--- Detected local environment. Using current directory. ---")
        checkpoint_dir = f'./{cfg.checkpoint_base_dir}'
        plot_dir = f'./{cfg.plot_base_dir}'
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    return checkpoint_dir, plot_dir

# ----------------- TRAINING ROUND LOGIC -----------------
def run_training_round(cfg: HolographicConfig, round_num: int, checkpoint_dir: str):
    """Executes one full round of training, from pre-training to fine-tuning."""
    
    # Checkpoint path is specific to the training round
    current_checkpoint_path = os.path.join(checkpoint_dir, f'holography_checkpoint_round_{round_num}.flax')
    previous_checkpoint_path = os.path.join(checkpoint_dir, f'holography_checkpoint_round_{round_num-1}.flax')

    master_key = jax.random.PRNGKey(42 + round_num) # Use round number to vary data
    training_keys, val_key_base = jax.random.split(master_key)
    
    total_pretrain_steps = cfg.pretrain_p_steps * cfg.pretrain_epochs_per_p
    total_finetune_steps = cfg.finetune_p_steps * cfg.finetune_epochs_per_p
    total_steps_per_round = total_pretrain_steps + total_finetune_steps
    training_keys = jax.random.split(training_keys, total_steps_per_round + 1)

    generator = CFTSampleGenerator(cfg)
    target_fns = get_target_theories(cfg)
    model = HolographicModel(cfg, cft_act=target_fns[0])
    
    # --- Robust State Initialization for Rounds and Resumption ---
    
    # 1. Check if we are resuming an interrupted run of the CURRENT round.
    if os.path.exists(current_checkpoint_path):
        print(f"--- Resuming Round {round_num} from checkpoint: {current_checkpoint_path} ---")
        restored_data = checkpoints.restore_checkpoint(current_checkpoint_path, target=None)
        start_step = restored_data['step']
        params = restored_data['params']
        opt_state = restored_data['opt_state']
        print(f"--- Resuming from step {start_step} ---")
    
    # 2. If not, start this round fresh, loading parameters from the PREVIOUS round if available.
    else:
        start_step = 0
        opt_state = None # Always start a new round with a fresh optimizer state
        if round_num > 0 and os.path.exists(previous_checkpoint_path):
            print(f"--- Starting Round {round_num}. Loading parameters from previous round: {previous_checkpoint_path} ---")
            params = checkpoints.restore_checkpoint(previous_checkpoint_path, target=None)['params']
        else:
            print(f"--- Starting Round {round_num} from scratch. ---")
            init_key, _ = jax.random.split(master_key)
            params = model.init(init_key, jnp.zeros((cfg.n_x, cfg.n_t)), jnp.zeros((cfg.n_x, cfg.n_t)), jnp.zeros((cfg.num_ir_params,)), True)['params']

    # 3. Define the optimizer and create the TrainState for the CURRENT phase of this round.
    #    The LR schedule is now correctly calculated based on the progress within this round.
    if start_step < total_pretrain_steps:
        print("--- Configuring for Phase 1: Pre-training ---")
        lr_schedule = optax.cosine_decay_schedule(cfg.pretrain_path_solver_lr, total_pretrain_steps)
        tx = optax.chain(optax.clip_by_global_norm(cfg.gradient_clip_norm), optax.adam(lr_schedule))
    else:
        print("--- Configuring for Phase 2: Fine-tuning ---")
        finetune_schedule_step = start_step - total_pretrain_steps
        lr_schedule = optax.cosine_decay_schedule(cfg.finetune_path_solver_lr, total_finetune_steps, initial_step=finetune_schedule_step)
        tx = optax.chain(optax.clip_by_global_norm(cfg.gradient_clip_norm), optax.adam(lr_schedule))

    # 4. Create the state.
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    
    # 5. If we restored an interrupted run, restore the optimizer state and step count.
    if opt_state is not None:
        state = state.replace(step=start_step, opt_state=opt_state)
    
    loss_fn = create_loss_fn(cfg, model, target_fns)
    
    @jax.jit
    def train_step(st, batch, scales):
        (loss, loss_dict), grads = jax.value_and_grad(loss_fn, has_aux=True)(st.params, batch, scales)
        st = st.apply_gradients(grads=grads)
        return st, loss, loss_dict

    @jax.jit
    def eval_step(params, batch, scales):
        return loss_fn(params, batch, scales)

    pbar = tqdm(total=total_steps_per_round, desc=f'Round {round_num} Progress', initial=start_step)
    training_history = {'train_loss': [], 'val_loss': []}
    global_step = start_step
    
    while global_step < total_steps_per_round:
        if global_step < total_pretrain_steps:
            # --- PHASE 1: PRE-TRAINING ---
            phase_name = "Pre-train"
            is_finetune = False
            p_curriculum_steps = cfg.pretrain_p_steps
            epochs_per_p = cfg.pretrain_epochs_per_p
            phase_base_step = 0
            loss_scales = calculate_gradient_scales(cfg, loss_fn, state.params, None, is_finetune=False)
        else:
            # --- PHASE 2: FINE-TUNING ---
            phase_name = "Fine-tune"
            if global_step == total_pretrain_steps: # First step of fine-tuning
                print("\n--- Transitioning to Phase 2: Fine-tuning. Resetting optimizer. ---")
                lr_schedule = optax.cosine_decay_schedule(cfg.finetune_path_solver_lr, total_finetune_steps)
                finetune_tx = optax.chain(optax.clip_by_global_norm(cfg.gradient_clip_norm), optax.adam(lr_schedule))
                state = state.replace(tx=finetune_tx, opt_state=finetune_tx.init(params))
            is_finetune = True
            p_curriculum_steps = cfg.finetune_p_steps
            epochs_per_p = cfg.finetune_epochs_per_p
            phase_base_step = total_pretrain_steps

        phase_step = global_step - phase_base_step
        p_step = phase_step // epochs_per_p
        current_p = (p_step / (p_curriculum_steps - 1)) * cfg.training_p_max if p_curriculum_steps > 1 else 0.0

        if is_finetune and phase_step % epochs_per_p == 0:
            print(f"\n--- Fine-tuning at p={current_p:.2f} (step {global_step}): Re-balancing ---")
            scale_batch = (generator.generate(training_keys[global_step], cfg.batch_size),
                           generator.generate(training_keys[global_step]+1, cfg.batch_size),
                           jnp.full((cfg.batch_size, cfg.num_ir_params), current_p))
            loss_scales = calculate_gradient_scales(cfg, loss_fn, state.params, scale_batch, is_finetune=True)

        p_batch = jnp.full((cfg.batch_size, cfg.num_ir_params), current_p, dtype=jnp.float32)
        batch = (generator.generate(training_keys[global_step], cfg.batch_size),
                 generator.generate(training_keys[global_step]+1, cfg.batch_size), p_batch)
        
        state, loss, loss_dict_train = train_step(state, batch, loss_scales)
        training_history['train_loss'].append(loss)

        if (global_step + 1) % cfg.log_frequency == 0:
            checkpoints.save_checkpoint(current_checkpoint_path, state, step=state.step, overwrite=True, keep=1)
            
            val_key, val_batch_key = jax.random.split(val_key_base); val_key_base = val_key
            val_batch = (generator.generate(val_batch_key, cfg.batch_size),
                         generator.generate(jax.random.split(val_batch_key)[0], cfg.batch_size),
                         jax.random.uniform(val_key, shape=(cfg.batch_size, cfg.num_ir_params), minval=0.0, maxval=cfg.training_p_max))
            val_total_loss, val_loss_dict = eval_step(state.params, val_batch, loss_scales)
            training_history['val_loss'].append(val_total_loss)
            pbar.set_postfix({'phase': phase_name, 'p': f'{current_p:.2f}', 'val_score': f'{float(val_loss_dict["score"]):.4f}', 'step': state.step})

        global_step += 1
        pbar.update(1)
        
    pbar.close()
    return state, training_history

# ----------------- VISUALIZATION SUITE -----------------
def visualize_results(cfg, state, training_history, generator, plot_dir, round_num):
    print(f"\n--- Generating Visualizations for Round {round_num} (saving to {plot_dir}) ---")
    
    if not training_history['train_loss']:
        print("No training history to visualize.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(training_history['train_loss'], label='Total Train Loss', alpha=0.7)
    if training_history['val_loss']:
        val_steps = jnp.linspace(0, len(training_history['train_loss']), len(training_history['val_loss']))
        ax.plot(val_steps, training_history['val_loss'], label='Total Validation Loss', lw=2)
    ax.set_yscale('log')
    ax.set_title(f'Training & Validation Loss (Round {round_num})')
    ax.set_xlabel('Training Step in Round')
    ax.set_ylabel('Log Loss')
    ax.legend()
    ax.grid(True, which='both', linestyle='--', alpha=0.6)
    # Save plot with round-specific name
    fig.savefig(os.path.join(plot_dir, f'loss_history_round_{round_num}.png'))
    plt.show()

    @jax.jit
    def get_model_output(params, puv, pir, ir):
        _, _, _, dev = state.apply_fn({'params': params}, puv, pir, ir)
        return dev

    vis_key = jax.random.PRNGKey(cfg.validation_seed + 1 + round_num)
    p_uv_sample = generator.generate(vis_key, 1)[0]
    p_ir_sample = generator.generate(jax.random.split(vis_key)[0], 1)[0]
    ir_sample = jnp.array([[cfg.training_p_max / 2.0]])[0]
    
    dev_sample = get_model_output(state.params, p_uv_sample, p_ir_sample, ir_sample)
    dev_mid_z = dev_sample[cfg.n_z // 2]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    im1 = axes[0].imshow(p_uv_sample, cmap='viridis', origin='lower'); axes[0].set_title('Input Field (p_uv)'); fig.colorbar(im1, ax=axes[0])
    im2 = axes[1].imshow(p_ir_sample, cmap='viridis', origin='lower'); axes[1].set_title('Target Boundary Field (p_ir)'); fig.colorbar(im2, ax=axes[1])
    im3 = axes[2].imshow(dev_mid_z, cmap='RdBu_r', origin='lower'); axes[2].set_title(f'Learned Deviation at z={cfg.n_z//2}'); fig.colorbar(im3, ax=axes[2])
    fig.suptitle(f'Example of Model Inputs and Learned Deviation (Round {round_num})'); fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f'field_visualization_round_{round_num}.png'))
    plt.show()

if __name__=='__main__':
    # Example: Run 2 full training rounds. The second round will start
    # with the model trained in the first round.
    config = HolographicConfig(num_training_rounds=2)
    
    checkpoint_directory, plot_directory = setup_environment(config)
    generator = CFTSampleGenerator(config)

    for i in range(config.num_training_rounds):
        print(f"\n{'='*25} STARTING TRAINING ROUND {i} {'='*25}")
        
        final_state, training_history = run_training_round(
            cfg=config, 
            round_num=i, 
            checkpoint_dir=checkpoint_directory
        )
        
        print(f"\n{'='*25} COMPLETED TRAINING ROUND {i} {'='*25}")
        
        if training_history['train_loss']:
            print(f"Final training loss for round {i}: {float(training_history['train_loss'][-1]):.4f}")
        if training_history['val_loss']:
            print(f"Final validation loss for round {i}: {float(training_history['val_loss'][-1]):.4f}")

        visualize_results(config, final_state, training_history, generator, plot_directory, round_num=i)

    print("\nAll training rounds completed.")

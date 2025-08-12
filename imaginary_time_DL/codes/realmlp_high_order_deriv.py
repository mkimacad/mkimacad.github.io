import jax
import jax.numpy as jnp
from jax import jit, vmap, random, lax
from jax import custom_vjp
from functools import partial
import flax.linen as nn
from dataclasses import dataclass
from flax.serialization import to_bytes, from_bytes, msgpack_restore, from_state_dict
from flax.serialization import to_state_dict
import optax
import matplotlib.pyplot as plt
import time
import os
from typing import Callable, Any, NamedTuple, List
from jax.experimental.jet import jet

# ================== GLOBAL CONFIGURATION ==================
RUN_NAME = "realmlp_forward_on_reverse_order_25_1"

@dataclass(frozen=True)
class Config:
    RUN_NAME: str = RUN_NAME
    ENABLE_PLOTTING: bool = True; SAVE_MODELS: bool = True; SEED: int = 42
    MODEL_WIDTH: int = 512; MODEL_DEPTH: int = 5
    PEAK_LR: float = 1e-1
    DERIV_ORDER: int = 25
    DERIVATIVE_LOSS_WEIGHT: float = 1.0
    USE_NORMALIZATION: bool = True
    TARGET_N_FREQUENCIES: int = 6; TARGET_MIN_FREQ: float = 0.05; TARGET_MAX_FREQ: float = 1.0
    TARGET_AMP_MIN: float = 0.5; TARGET_AMP_MAX: float = 5.0
    PLATEAU_PATIENCE: int = 4; PLATEAU_FACTOR: float = 0.5
    PLATEAU_MIN_LR: float = 1e-10
    PLATEAU_IMPROVEMENT_REL_THRESHOLD: float = 1e-1
    TOTAL_TRAINING_STEPS: int = 480_000
    LOG_EVERY_N_STEPS: int = 5000; GRADIENT_CLIP_VALUE: float = 1.0
    N_TRAINING_SAMPLES: int = 20
    INTERPOLATION_HALF_WIDTH: float = 1.0; EXTRAPOLATION_HALF_WIDTH: float = 8.0

# ================== HELPERS ==================
@custom_vjp
def safe_logcosh(x):
    threshold = 15.0
    return jnp.where(jnp.abs(x) > threshold, jnp.abs(x) - jnp.log(2.0), jnp.log(jnp.cosh(x)))
def _safe_logcosh_fwd(x): return safe_logcosh(x), x
def _safe_logcosh_bwd(x, g): return (g * jnp.tanh(x),)
safe_logcosh.defvjp(_safe_logcosh_fwd, _safe_logcosh_bwd)

class NormalizationStats(NamedTuple):
    centers: List[jnp.ndarray]
    scales: List[jnp.ndarray]

@jit
def compute_normalization_stats(deriv_tensors: List[jnp.ndarray]) -> NormalizationStats:
    centers = [jnp.median(d, axis=0) for d in deriv_tensors]
    scales = [jnp.median(jnp.abs(d - c), axis=0) * 1.4826 + 1e-8 for d, c in zip(deriv_tensors, centers)]
    return NormalizationStats(centers=centers, scales=scales)

# ================== FAdam (Fromage-like) IMPLEMENTATION ==================
class FAdamState(NamedTuple):
    count: jnp.ndarray

def fadam(learning_rate=1e-3, eps=1e-8) -> optax.GradientTransformation:
    def init_fn(params): return FAdamState(count=jnp.zeros([], jnp.int32))
    def update_fn(updates, state, params=None):
        global_grad_norm = optax.global_norm(updates)
        normalized_updates = jax.tree_util.tree_map(lambda g: g / (global_grad_norm + eps), updates)
        scaled_updates = jax.tree_util.tree_map(lambda g: -learning_rate * g, normalized_updates)
        return scaled_updates, FAdamState(count=state.count + 1)
    return optax.GradientTransformation(init_fn, update_fn)

# ================== CORE COMPONENTS ==================
class TargetFunctionParams(NamedTuple):
    ks: jnp.ndarray; amps: jnp.ndarray

class ReduceLROnPlateauState(NamedTuple):
    lr: float; best_loss: float; patience_counter: int

class TrainState(NamedTuple):
    params: Any; opt_state: Any; key: random.PRNGKey; step: int
    lr_plateau_state: ReduceLROnPlateauState; target_params: TargetFunctionParams; norm_stats: NormalizationStats

class RealMLP(nn.Module):
    cfg: Config
    def setup(self):
        self.layers = [nn.Dense(self.cfg.MODEL_WIDTH, name=f"dense_{i}") for i in range(self.cfg.MODEL_DEPTH - 1)]
        self.out_layer = nn.Dense(1, name="dense_out")
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        core = x
        for layer in self.layers:
            core = layer(core)
            core = nn.gelu(core)
        out = self.out_layer(core)
        return jnp.squeeze(out, axis=-1)

def build_target_function(target_params: TargetFunctionParams) -> Callable:
    ks, amps = target_params.ks, target_params.amps
    def _f(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(amps * jnp.cos(ks * x), axis=-1)
    return _f

def get_target_derivs(f: Callable, x_data: jnp.ndarray, max_order: int) -> List[jnp.ndarray]:
    def get_series(x_scalar):
        series_in = (x_scalar,) + (jnp.ones_like(x_scalar),) + tuple(jnp.zeros_like(x_scalar) for _ in range(max_order))
        primal, series = jet(f, (x_scalar,), (series_in,))
        return (primal,) + tuple(series)
    all_derivs_series = vmap(get_series)(x_data.flatten())
    return [d.reshape(-1, 1) for d in all_derivs_series]

# ================== TRAINING STEP AND UTILITIES ==================
def save_checkpoint(path, state: TrainState):
    with open(path, "wb") as f: f.write(to_bytes(to_state_dict(state._replace(opt_state=None))))

def load_checkpoint_state(path, dummy_state: TrainState):
    if not os.path.exists(path): return None
    try:
        with open(path, "rb") as f: content = f.read()
        return from_state_dict(dummy_state, msgpack_restore(content))
    except Exception as e:
        print(f"-> Checkpoint read failed: {e}. Starting from scratch.")
        return None

@partial(jit, static_argnames=['cfg'])
def update_lr_on_plateau(state: TrainState, loss: float, cfg: Config) -> TrainState:
    lr_state = state.lr_plateau_state
    is_better = loss < lr_state.best_loss * (1.0 - cfg.PLATEAU_IMPROVEMENT_REL_THRESHOLD)
    new_counter = jnp.where(is_better, 0, lr_state.patience_counter + 1)
    reduce_cond = (new_counter >= cfg.PLATEAU_PATIENCE)
    new_lr = jnp.where(reduce_cond, jnp.maximum(lr_state.lr * cfg.PLATEAU_FACTOR, cfg.PLATEAU_MIN_LR), lr_state.lr)
    lax.cond(new_lr < lr_state.lr, lambda: jax.debug.print(">>> LRPlateau: Reducing LR from {lr:.2e} to {nlr:.2e}", lr=lr_state.lr, nlr=new_lr), lambda: None)
    return state._replace(lr_plateau_state=lr_state._replace(lr=new_lr, best_loss=jnp.where(is_better, loss, lr_state.best_loss), patience_counter=jnp.where(reduce_cond, 0, new_counter)))

@partial(jit, static_argnames=['model', 'cfg', 'get_target_derivs_fn'])
def training_step(model: nn.Module, state: TrainState, _, cfg: Config, get_target_derivs_fn: Callable):
    key, data_key = random.split(state.key)
    x_train = random.uniform(data_key, (cfg.N_TRAINING_SAMPLES, 1), minval=-cfg.INTERPOLATION_HALF_WIDTH, maxval=cfg.INTERPOLATION_HALF_WIDTH)
    target_derivs = get_target_derivs_fn(x_train)

    def loss_fn(params):
        def get_model_derivs_series(x_scalar):
            def f_scalar(x):
                # Apply the model with the given parameters for a single point.
                return model.apply({'params': params}, jnp.reshape(x, (1, 1)))[0]
            
            series_in = (x_scalar,) + (jnp.ones_like(x_scalar),) + tuple(jnp.zeros_like(x_scalar) for _ in range(cfg.DERIV_ORDER))
            primal, series = jet(f_scalar, (x_scalar,), (series_in,))
            return (primal,) + tuple(series)
        
        # Vmap over the training batch to get all derivatives for all points..
        model_derivs_series = vmap(get_model_derivs_series)(x_train.flatten())
        
        total_loss = 0.0
        for k in range(cfg.DERIV_ORDER + 1):
            pred_k = model_derivs_series[k].reshape(-1, 1)
            target_k = target_derivs[k]
            
            pred_norm = (pred_k - state.norm_stats.centers[k]) / state.norm_stats.scales[k]
            target_norm = (target_k - state.norm_stats.centers[k]) / state.norm_stats.scales[k]
            
            loss_k = jnp.mean(jnp.sum(safe_logcosh(pred_norm - target_norm), axis=-1))
            weight = 1.0 if k == 0 else cfg.DERIVATIVE_LOSS_WEIGHT
            total_loss += weight * loss_k
            
        return total_loss

    total_loss, grads = jax.value_and_grad(loss_fn)(state.params)

    gnorm = optax.global_norm(grads)
    clip_coef = jnp.minimum(1.0, cfg.GRADIENT_CLIP_VALUE / (gnorm + 1e-12))
    grads = jax.tree_map(lambda g: g * clip_coef, grads)
    leaves = jax.tree_leaves(grads)
    grads_finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(l)) for l in leaves if l is not None]))
    grads = lax.cond(grads_finite, lambda g: g, lambda g: jax.tree_map(jnp.zeros_like, g), grads)

    current_lr = state.lr_plateau_state.lr
    temp_fadam_optimizer = fadam(learning_rate=current_lr)
    updates, new_opt_state = temp_fadam_optimizer.update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)

    return state._replace(params=new_params, opt_state=new_opt_state, key=key, step=state.step + 1), total_loss

# ============================
# RUN TRAINING
# ============================
def run_training(key, cfg: Config, target_params, save_dir):
    model = RealMLP(cfg)
    key, init_key, norm_key = random.split(key, 3)
    optimizer = fadam()
    dummy_params = model.init(init_key, jnp.ones((1, 1)))['params']
    target_func = build_target_function(target_params)
    get_target_derivs_fn = jit(partial(get_target_derivs, target_func, max_order=cfg.DERIV_ORDER))

    print("--- Computing normalization stats for derivatives... ---")
    x_stats = random.uniform(norm_key, (20000, 1), minval=-cfg.INTERPOLATION_HALF_WIDTH, maxval=cfg.INTERPOLATION_HALF_WIDTH)
    chunk_size = 2000
    parts = [get_target_derivs_fn(x_stats[i:i+chunk_size]) for i in range(0, x_stats.shape[0], chunk_size)]
    target_derivs_for_norm = [jnp.concatenate([p[k] for p in parts], axis=0) for k in range(cfg.DERIV_ORDER + 1)]
    print("  -> Finished computing derivatives for normalization stats.")

    norm_stats = compute_normalization_stats(target_derivs_for_norm)
    if not jnp.all(jnp.isfinite(jnp.concatenate([c.reshape(-1) for c in norm_stats.centers]))):
        norm_stats = NormalizationStats([jnp.nan_to_num(c, 0.0) for c in norm_stats.centers], [jnp.nan_to_num(s, 1.0) for s in norm_stats.scales])

    initial_lr_state = ReduceLROnPlateauState(lr=cfg.PEAK_LR, best_loss=jnp.inf, patience_counter=0)
    dummy_state = TrainState(dummy_params, None, key, 0, initial_lr_state, target_params, norm_stats)
    checkpoint_path = os.path.join(save_dir, "model_checkpoint.msgpack")
    state = load_checkpoint_state(checkpoint_path, dummy_state)
    if state is None:
        print("--- No checkpoint found. Creating initial state... ---")
        state = dummy_state._replace(opt_state=optimizer.init(dummy_params))
    else:
        print(f"--- Loaded state from checkpoint. Resuming at step {state.step}. ---")
        state = state._replace(opt_state=optimizer.init(state.params), norm_stats=norm_stats, target_params=target_params)

    # Pass the model as a static argument to the training step.
    scan_step_fn = partial(training_step, model, cfg=cfg, get_target_derivs_fn=get_target_derivs_fn)

    print(f"\n--- Starting Training (Main Loss, DERIV_ORDER={cfg.DERIV_ORDER}) ---")
    start_step = state.step
    while state.step < cfg.TOTAL_TRAINING_STEPS:
        steps_to_run = min(cfg.LOG_EVERY_N_STEPS, cfg.TOTAL_TRAINING_STEPS - state.step)
        if steps_to_run <= 0: break
        state, loss_history = lax.scan(scan_step_fn, state, jnp.arange(steps_to_run))
        state = state._replace(key=random.split(state.key)[0])
        state = update_lr_on_plateau(state, jnp.mean(loss_history), cfg)
        print(f"  Steps {start_step}-{state.step}/{cfg.TOTAL_TRAINING_STEPS} | Avg Loss: {jnp.mean(loss_history):.3e} | LR: {state.lr_plateau_state.lr:.2e}")
        start_step = state.step
        if cfg.SAVE_MODELS: save_checkpoint(checkpoint_path, state)
    return model, state

# ================== PLOTTING & MAIN ==================
def generate_info_text(cfg: Config, state: TrainState):
    lines = [f"Run: {cfg.RUN_NAME}", f"Optimizer: FAdam", f"NN: Real GELU-MLP {cfg.MODEL_WIDTH}x{cfg.MODEL_DEPTH}", f"Loss: Value + Derivatives (Order ≤ {cfg.DERIV_ORDER})", f"Deriv Weight: {cfg.DERIVATIVE_LOSS_WEIGHT}", f"Final Step: {state.step}"]
    return "\n".join(lines)

def plot_result(model, state, save_dir, cfg: Config):
    if not cfg.ENABLE_PLOTTING: return
    print("  -> Generating final plot...")
    target_func = build_target_function(state.target_params)
    apply_fn = jit(lambda params, x: model.apply({'params': params}, x))
    x_plot = jnp.linspace(-cfg.EXTRAPOLATION_HALF_WIDTH, cfg.EXTRAPOLATION_HALF_WIDTH, 2000)
    x_plot_data = x_plot.reshape(-1, 1)
    y_truth, y_pred = vmap(target_func)(x_plot_data), apply_fn(state.params, x_plot_data)
    fig, axs = plt.subplots(2, 1, figsize=(18, 12), sharex=True)
    axs[0].plot(x_plot, y_truth, 'k--', lw=2, label='Ground Truth f(x)')
    axs[0].plot(x_plot, y_pred, 'r', lw=2, alpha=0.8, label='Predicted f(x)')
    axs[0].axvspan(-cfg.INTERPOLATION_HALF_WIDTH, cfg.INTERPOLATION_HALF_WIDTH, color='gray', alpha=0.2, label='Training Domain')
    axs[0].set_title(f"Model Performance - {cfg.RUN_NAME}"); axs[0].set_ylabel("f(x)"); axs[0].grid(True, linestyle=':'); axs[0].legend()
    axs[0].text(0.02, 0.98, generate_info_text(cfg, state), transform=axs[0].transAxes, fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='aliceblue', alpha=0.8))
    axs[1].plot(x_plot, jnp.abs(y_truth - y_pred), 'b', lw=2, label='Absolute Error |f_true - f_pred|')
    axs[1].axvspan(-cfg.INTERPOLATION_HALF_WIDTH, cfg.INTERPOLATION_HALF_WIDTH, color='gray', alpha=0.2)
    axs[1].set_xlabel("x-axis"); axs[1].set_ylabel("Absolute Error"); axs[1].set_yscale('log'); axs[1].grid(True, which="both", linestyle=':'); axs[1].legend()
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, f"plot_final_{cfg.RUN_NAME}.png")); plt.show()

def main():
    start_time = time.time()
    cfg = Config()
    try:
        from google.colab import drive; drive.mount('/content/drive', force_remount=True)
        base_dir = '/content/drive/MyDrive/realmlp_deriv_order_25_1'
    except (ImportError, ModuleNotFoundError):
        base_dir = 'realmlp_deriv_order_25_1'
    save_dir = os.path.join(base_dir, cfg.RUN_NAME); os.makedirs(save_dir, exist_ok=True)
    jax.config.update("jax_enable_x64", False)
    print(f"JAX backend: {jax.default_backend()} | Precision: 32-bit")
    print(f"### RUNNING: {cfg.RUN_NAME} ###"); print(f"Results will be saved in: {save_dir}")
    key = random.PRNGKey(cfg.SEED); print(f"--- Using Seed: {cfg.SEED} ---")
    key, target_key, train_key = random.split(key, 3)
    target_params_path = os.path.join(save_dir, "target_params.msgpack")
    if os.path.exists(target_params_path):
        print("--- Loading existing target function parameters ---")
        with open(target_params_path, 'rb') as f:
            target_params = from_bytes(TargetFunctionParams(ks=None, amps=None), f.read())
    else:
        print("--- Generating new target function parameters ---")
        freq_key, amp_key = random.split(target_key)
        ks = random.uniform(freq_key, (cfg.TARGET_N_FREQUENCIES,), minval=cfg.TARGET_MIN_FREQ, maxval=cfg.TARGET_MAX_FREQ)
        amps = random.uniform(amp_key, (cfg.TARGET_N_FREQUENCIES,), minval=cfg.TARGET_AMP_MIN, maxval=cfg.TARGET_AMP_MAX)
        target_params = TargetFunctionParams(ks=ks, amps=amps)
        if cfg.SAVE_MODELS:
            with open(target_params_path, 'wb') as f: f.write(to_bytes(target_params))

    model, final_state = run_training(train_key, cfg, target_params, save_dir)
    if final_state is not None:
        plot_result(model, final_state, save_dir, cfg)
        print("\n" + "=" * 60); print(">>> TRAINING COMPLETE <<<"); print("=" * 60 + "\n")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds.")

if __name__ == '__main__':
    main()

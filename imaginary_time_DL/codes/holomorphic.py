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
RUN_NAME = "holomorphicMLP_order_0_2D_1"

@dataclass(frozen=True)
class Config:
    RUN_NAME: str = RUN_NAME
    ENABLE_PLOTTING: bool = True; SAVE_MODELS: bool = True; SEED: int = 42
    MODEL_WIDTH: int = 512; MODEL_DEPTH: int = 5
    PEAK_LR: float = 1e-1
    DERIV_ORDER: int = 0
    DERIVATIVE_LOSS_WEIGHT: float = 1.0
    USE_NORMALIZATION: bool = True
    TARGET_N_FREQUENCIES: int = 6; TARGET_MIN_FREQ: float = 0.05; TARGET_MAX_FREQ: float = 1.0
    TARGET_AMP_MIN: float = 0.5; TARGET_AMP_MAX: float = 5.0
    PLATEAU_PATIENCE: int = 4; PLATEAU_FACTOR: float = 0.5
    PLATEAU_MIN_LR: float = 1e-10
    PLATEAU_IMPROVEMENT_REL_THRESHOLD: float = 1e-1
    TOTAL_TRAINING_STEPS: int = 700_000
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

def fadam(learning_rate=1e-3, eps=1e-20) -> optax.GradientTransformation:
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

class HolomorphicMLP(nn.Module):
    cfg: Config
    @nn.compact
    def __call__(self, z_real: jnp.ndarray) -> jnp.ndarray:
        z_complex = z_real[..., 0] + 1j * z_real[..., 1]
        core_features = z_complex[..., None]
        dtype = jnp.complex64
        kernel_init = nn.initializers.glorot_normal(dtype=dtype)
        for i in range(self.cfg.MODEL_DEPTH - 1):
            core_features = nn.Dense(self.cfg.MODEL_WIDTH, dtype=dtype, kernel_init=kernel_init, name=f"complex_dense_{i}")(core_features)
            core_features = jnp.tanh(core_features)
        holo_output = nn.Dense(1, dtype=jnp.complex64, name="complex_dense_out")(core_features)
        return jnp.squeeze(holo_output, axis=-1)

def build_target_function(target_params: TargetFunctionParams) -> Callable:
    ks, amps = target_params.ks, target_params.amps
    def _w(z_real: jnp.ndarray) -> jnp.ndarray:
        z = z_real[..., 0] + 1j * z_real[..., 1]
        w_complex = jnp.sum(amps * jnp.cos(ks * z))
        return w_complex
    return _w

def get_target_derivs(f: Callable, z_data: jnp.ndarray, max_order: int) -> List[jnp.ndarray]:
    def f_wrapped(z_real):
        w_complex = f(z_real)
        return jnp.stack([w_complex.real, w_complex.imag], axis=-1)

    def get_series(z_scalar):
        direction = jnp.array([1.0, 0.0])
        series_in = (z_scalar,) + (direction,) + tuple(jnp.zeros_like(z_scalar) for _ in range(max_order))
        primal, series = jet(f_wrapped, (z_scalar,), (series_in,))
        return (primal,) + tuple(series)

    all_derivs_series = vmap(get_series)(z_data)
    return [d for d in all_derivs_series]

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

    # FIXED: Sample from the 2D complex plane (real and imaginary parts) instead of just the real axis.
    z_train = random.uniform(data_key, (cfg.N_TRAINING_SAMPLES, 2),
                              minval=-cfg.INTERPOLATION_HALF_WIDTH,
                              maxval=cfg.INTERPOLATION_HALF_WIDTH)

    target_derivs = get_target_derivs_fn(z_train)

    def loss_fn(params):
        def get_model_derivs_series(z_scalar):
            def f_scalar_model(z_point):
                w_complex = model.apply({'params': params}, jnp.reshape(z_point, (1, 2)))[0]
                return jnp.stack([w_complex.real, w_complex.imag], axis=-1)

            direction = jnp.array([1.0, 0.0])
            series_in = (z_scalar,) + (direction,) + tuple(jnp.zeros_like(z_scalar) for _ in range(cfg.DERIV_ORDER))
            primal, series = jet(f_scalar_model, (z_scalar,), (series_in,))
            return (primal,) + tuple(series)

        model_derivs_series = vmap(get_model_derivs_series)(z_train)

        total_loss = 0.0
        for k in range(cfg.DERIV_ORDER + 1):
            pred_k = model_derivs_series[k]
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
    model = HolomorphicMLP(cfg)
    key, init_key, norm_key = random.split(key, 3)
    optimizer = fadam()
    dummy_params = model.init(init_key, jnp.ones((1, 2)))['params']
    target_func = build_target_function(target_params)
    get_target_derivs_fn = jit(partial(get_target_derivs, target_func, max_order=cfg.DERIV_ORDER))

    print("--- Computing normalization stats for derivatives... ---")
    # FIXED: Sample from the 2D domain for normalization to match the training distribution.
    z_stats = random.uniform(norm_key, (20000, 2),
                             minval=-cfg.INTERPOLATION_HALF_WIDTH,
                             maxval=cfg.INTERPOLATION_HALF_WIDTH)
    chunk_size = 2000
    parts = [get_target_derivs_fn(z_stats[i:i+chunk_size]) for i in range(0, z_stats.shape[0], chunk_size)]
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

    scan_step_fn = partial(training_step, model, cfg=cfg, get_target_derivs_fn=get_target_derivs_fn)

    print(f"\n--- Starting Training (HoloMLP, DERIV_ORDER={cfg.DERIV_ORDER}) ---")
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
    lines = [f"Run: {cfg.RUN_NAME}", f"Optimizer: FAdam", f"NN: Holo Tanh-MLP {cfg.MODEL_WIDTH}x{cfg.MODEL_DEPTH}", f"Loss: Value + Derivatives (Order ≤ {cfg.DERIV_ORDER})", f"Deriv Weight: {cfg.DERIVATIVE_LOSS_WEIGHT}", f"Final Step: {state.step}"]
    return "\n".join(lines)

def plot_result(model, state, save_dir, cfg: Config):
    if not cfg.ENABLE_PLOTTING: return
    print("  -> Generating final plot...")
    target_func = build_target_function(state.target_params)
    apply_fn = jit(lambda params, z: model.apply({'params': params}, z))

    x_plot = jnp.linspace(-cfg.EXTRAPOLATION_HALF_WIDTH, cfg.EXTRAPOLATION_HALF_WIDTH, 2000)
    z_plot_real = jnp.stack([x_plot, jnp.zeros_like(x_plot)], axis=-1)

    w_truth = vmap(target_func)(z_plot_real)
    uv_truth = jnp.stack([w_truth.real, w_truth.imag], axis=-1)

    w_pred = apply_fn(state.params, z_plot_real)
    uv_pred = jnp.stack([w_pred.real, w_pred.imag], axis=-1)

    fig, axs = plt.subplots(2, 1, figsize=(18, 12), sharex=True)

    axs[0].plot(x_plot, uv_truth[:, 0], 'k--', lw=2, label='Ground Truth U(x, 0)')
    axs[0].plot(x_plot, uv_pred[:, 0], 'r', lw=2, alpha=0.8, label='Predicted U(x, 0)')
    axs[0].axvspan(-cfg.INTERPOLATION_HALF_WIDTH, cfg.INTERPOLATION_HALF_WIDTH, color='gray', alpha=0.2, label='Training Domain')
    axs[0].set_title(f"Model Performance (Real Part) - {cfg.RUN_NAME}"); axs[0].set_ylabel("U component"); axs[0].grid(True, linestyle=':'); axs[0].legend()
    axs[0].text(0.02, 0.98, generate_info_text(cfg, state), transform=axs[0].transAxes, fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='aliceblue', alpha=0.8))

    error_u = jnp.abs(uv_truth[:, 0] - uv_pred[:, 0])
    axs[1].plot(x_plot, error_u, 'b', lw=2, label='Absolute Error |U_true - U_pred|')
    axs[1].axvspan(-cfg.INTERPOLATION_HALF_WIDTH, cfg.INTERPOLATION_HALF_WIDTH, color='gray', alpha=0.2)
    axs[1].set_xlabel("x-axis (z = x + 0i)"); axs[1].set_ylabel("Absolute Error"); axs[1].set_yscale('log'); axs[1].grid(True, which="both", linestyle=':'); axs[1].legend()

    plt.tight_layout(); plt.savefig(os.path.join(save_dir, f"plot_final_{cfg.RUN_NAME}.png")); plt.show()

def main():
    start_time = time.time()
    cfg = Config()
    try:
        from google.colab import drive; drive.mount('/content/drive', force_remount=True)
        base_dir = '/content/drive/MyDrive/holomorphic_1'
    except (ImportError, ModuleNotFoundError):
        base_dir = 'holomorphic_1'
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

import jax
import jax.numpy as jnp
from jax import jit, vmap, value_and_grad, custom_vjp, random
from jax.experimental.jet import jet
import optax
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
import math
import time
import os
from typing import Callable, Tuple, List, NamedTuple

# ================== GLOBAL CONFIGURATION ==================
jax.config.update("jax_enable_x64", False) # Default precision: 32-bit

RUN_NAME = "run_1d"
ENABLE_PLOTTING = True
SEED = 42 # Randomness also in target function generation

# --- Target Function Configuration ---
TARGET_N_FREQUENCIES = 6
TARGET_MIN_FREQ = 0.05
TARGET_MAX_FREQ = 1.0
TARGET_AMP_MIN = 0.5
TARGET_AMP_MAX = 5.0

# --- Training Configuration ---
SCHEDULER_TYPE = 'reduce_on_plateau'  # Options: 'warmup_cosine_decay', 'reduce_on_plateau'
LOSS_FUNCTION_TYPE = 'log_cosh' # log_cosh or mse, no reason to use mse though
USE_DERIVATIVE_DATA_LOSS = True
DERIVATIVE_DATA_WEIGHT = 1.0
PEAK_LR = 3e-4
TOTAL_TRAINING_STEPS = 200000
LOG_EVERY_N_STEPS = 2000
BATCH_SIZE = 128
GRADIENT_CLIP_VALUE = 1.0

# --- ReduceLROnPlateau Specific Settings ---
PLATEAU_PATIENCE_CHECKS = 100
PLATEAU_CHECK_EVERY_N_STEPS = 100
PLATEAU_REDUCTION_FACTOR = 0.25
PLATEAU_MIN_LR = 1e-10

# --- Architecture & Domain Configuration ---
ACTIVATION_FUNCTION = 'gelu'
MODEL_WIDTH = 512
MODEL_DEPTH = 6
INTERPOLATION_HALF_WIDTH = 1.0
EXTRAPOLATION_HALF_WIDTH = 8.0

print(f"JAX is using: {jax.default_backend()} with {'64-bit' if jax.config.jax_enable_x64 else '32-bit'} precision.")
print(f"### RUNNING 1D FUNCTION APPROXIMATION (REFERENCE-BASED HYBRID) ###")
print(f"Scheduler: {SCHEDULER_TYPE.replace('_', ' ').title()}")

# ================== 1. NETWORK & HELPERS ==================

@custom_vjp
def safe_logcosh(x):
    threshold = 15.0
    return jnp.where(jnp.abs(x) > threshold, jnp.abs(x) - jnp.log(2.0), jnp.log(jnp.cosh(x)))

def _safe_logcosh_fwd(x): return safe_logcosh(x), x
def _safe_logcosh_bwd(x, g): return (g * jnp.tanh(x),)
safe_logcosh.defvjp(_safe_logcosh_fwd, _safe_logcosh_bwd)

def init_mlp_params(layer_widths, key):
    params = []
    for i in range(len(layer_widths) - 1):
        key, subkey = random.split(key)
        # Using Glorot/Xavier initialization from reference
        std_dev = jnp.sqrt(2.0 / (layer_widths[i] + layer_widths[i+1]))
        W = random.normal(subkey, (layer_widths[i+1], layer_widths[i])) * std_dev
        b = jnp.zeros((layer_widths[i+1],))
        params.append({'W': W, 'b': b})
    return params

def mlp_forward(params, x, activation_fn):
    # Ensure input is at least 1D
    x = jnp.atleast_1d(x)
    *hidden_layers, last_layer = params
    for layer in hidden_layers:
        x = jnp.dot(layer['W'], x) + layer['b']
        x = activation_fn(x)
    x = jnp.dot(last_layer['W'], x) + last_layer['b']
    return x[0]

# ================== 2. TARGET FUNCTION ==================

def generate_random_target(key, n_freq, min_freq, max_freq, amp_min, amp_max) -> Tuple[Callable, Callable]:
    freq_key, amp_key = random.split(key)
    ks = random.uniform(freq_key, (n_freq,), minval=min_freq, maxval=max_freq)
    amps = random.uniform(amp_key, (n_freq,), minval=amp_min, maxval=amp_max)
    
    @jit
    def _u_fn(x):
        return jnp.sum(amps * jnp.cos(ks * x), -1)
    
    @jit
    def _derivs_fn(x):
        return jnp.sum(-amps * ks * jnp.sin(ks * x), -1)
        
    return _u_fn, _derivs_fn

# ================== 3. NORMALIZATION ==================

@partial(jit)
def compute_normalization_stats(y_data):
    center = jnp.median(y_data)
    mad = jnp.median(jnp.abs(y_data - center))
    scale = mad * 1.4826 + 1e-8
    return center, scale

@partial(jit)
def normalize_data(y, center, scale): return (y - center) / scale

@partial(jit)
def denormalize_data(y_norm, center, scale): return y_norm * scale + center

# ================== 4. LOSS FUNCTION ==================

def create_loss_function(target_fns, activation_fn, norm_stats_u, norm_stats_du):
    center_u, scale_u = norm_stats_u
    center_du, scale_du = norm_stats_du

    # Pre-calculate initial conditions at x=0 from the true functions
    u_ic = target_fns[0](0.0)
    du_ic = target_fns[1](0.0)

    def loss_fn(params, x_batch):
        f_nn_scalar = lambda x: mlp_forward(params, x, activation_fn)

        # --- Initial Condition Loss (The Anchor) ---
        primals_ic = (0.0,)
        series_ic = (1.0, 1.0) # For f(0) and f'(0)
        f_val_at_0, taylor_coeffs_at_0 = jet(f_nn_scalar, primals_ic, (series_ic,))
        
        # Error in function value at x=0 (in original scale)
        error_ic_u = f_val_at_0 - u_ic
        
        # Error in derivative at x=0 (in original scale)
        pred_ic_du = taylor_coeffs_at_0[0]
        error_ic_du = pred_ic_du - du_ic

        ic_loss = safe_logcosh(error_ic_u) + safe_logcosh(error_ic_du)

        # --- Data Loss (in normalized space) ---
        def single_point_loss(x):
            # Value loss
            pred_u_raw = f_nn_scalar(x)
            target_u_raw = target_fns[0](x)
            pred_u_norm = normalize_data(pred_u_raw, center_u, scale_u)
            target_u_norm = normalize_data(target_u_raw, center_u, scale_u)
            u_loss = safe_logcosh(pred_u_norm - target_u_norm)
            
            # Derivative loss
            du_loss = 0.0
            if USE_DERIVATIVE_DATA_LOSS:
                primals_d = (x,)
                series_d = (1.0, 1.0)
                _, taylor_coeffs_d = jet(f_nn_scalar, primals_d, (series_d,))
                pred_du_raw = taylor_coeffs_d[0]
                target_du_raw = target_fns[1](x)
                pred_du_norm = normalize_data(pred_du_raw, center_du, scale_du)
                target_du_norm = normalize_data(target_du_raw, center_du, scale_du)
                du_loss = safe_logcosh(pred_du_norm - target_du_norm)

            return u_loss + DERIVATIVE_DATA_WEIGHT * du_loss

        data_loss = jnp.mean(vmap(single_point_loss)(x_batch))
        
        ic_weight = 1e-2 # Weighting for the initial condition anchor
        return data_loss + ic_weight * ic_loss
    
    return jit(loss_fn)

# ================== 5. TRAINING & PLOTTING ==================

@partial(jit, static_argnums=(2, 3))
def train_step(params, opt_state, loss_fn, optimizer, x_batch):
    loss, grads = value_and_grad(loss_fn)(params, x_batch)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

class ReduceLROnPlateauState(NamedTuple):
    lr: float
    best_loss: float
    patience_counter: int

def run_training(key, target_fns, save_dir):
    activation_fn = {'gelu': jax.nn.gelu, 'tanh': jnp.tanh}.get(ACTIVATION_FUNCTION)

    print("--- Computing normalization statistics... ---")
    key, subkey = random.split(key)
    x_stats = random.uniform(subkey, (10000, 1), minval=-INTERPOLATION_HALF_WIDTH, maxval=INTERPOLATION_HALF_WIDTH)
    norm_stats_u = compute_normalization_stats(vmap(target_fns[0])(x_stats))
    norm_stats_du = compute_normalization_stats(vmap(target_fns[1])(x_stats))
    print(f"  U-Value Stats: Center={norm_stats_u[0]:.2f}, Scale={norm_stats_u[1]:.2f}")
    if USE_DERIVATIVE_DATA_LOSS: print(f"  dU/dx Stats: Center={norm_stats_du[0]:.2f}, Scale={norm_stats_du[1]:.2f}")

    loss_function = create_loss_function(target_fns, activation_fn, norm_stats_u, norm_stats_du)

    layer_widths = [1] + [MODEL_WIDTH] * (MODEL_DEPTH - 1) + [1]
    key, subkey = random.split(key)
    params = init_mlp_params(layer_widths, subkey)

    if SCHEDULER_TYPE == 'warmup_cosine_decay':
        schedule = optax.warmup_cosine_decay_schedule(init_value=0.0, peak_value=PEAK_LR, warmup_steps=int(TOTAL_TRAINING_STEPS*0.1), decay_steps=int(TOTAL_TRAINING_STEPS*0.9))
        optimizer = optax.chain(optax.clip_by_global_norm(GRADIENT_CLIP_VALUE), optax.adamw(learning_rate=schedule))
    else: # 'reduce_on_plateau'
        # MODIFICATION 1: Wrap adamw with inject_hyperparams to access the learning_rate later.
        # This makes the learning rate part of the optimizer's state.
        optimizer = optax.chain(
            optax.clip_by_global_norm(GRADIENT_CLIP_VALUE),
            optax.inject_hyperparams(optax.adamw)(learning_rate=PEAK_LR)
        )
        plateau_state = ReduceLROnPlateauState(lr=PEAK_LR, best_loss=jnp.inf, patience_counter=0)

    opt_state = optimizer.init(params)

    print(f"--- Starting Main Training ({TOTAL_TRAINING_STEPS} steps) ---")
    for step in range(1, TOTAL_TRAINING_STEPS + 1):
        key, data_key = random.split(key)
        x_batch = random.uniform(data_key, (BATCH_SIZE,), minval=-INTERPOLATION_HALF_WIDTH, maxval=INTERPOLATION_HALF_WIDTH)

        # MODIFICATION 2: Correctly update the learning rate in the optimizer state.
        # opt_state is a tuple: (ClipState, AdamState). We modify the AdamState (at index 1).
        # Since JAX states are immutable, we create a new state with the updated LR.
        if SCHEDULER_TYPE == 'reduce_on_plateau':
            # The state for inject_hyperparams is a named tuple containing the hyperparams dict.
            # We create a new state tuple by replacing the adamw state with an updated version.
            adam_state = opt_state[1] # Get the adamw state from the chain's state tuple
            new_adam_state = adam_state._replace(
                hyperparams={'learning_rate': plateau_state.lr}
            )
            opt_state = (opt_state[0], new_adam_state) # Reconstruct the full state tuple

        params, opt_state, loss = train_step(params, opt_state, loss_function, optimizer, x_batch)

        if not jnp.isfinite(loss):
            print(f"!!! Step {step}: NaN loss detected. Halting training. !!!"); return None, None

        if SCHEDULER_TYPE == 'reduce_on_plateau' and step % PLATEAU_CHECK_EVERY_N_STEPS == 0:
            key, val_key = random.split(key)
            x_val = random.uniform(val_key, (1024,), minval=-INTERPOLATION_HALF_WIDTH, maxval=INTERPOLATION_HALF_WIDTH)
            val_loss = loss_function(params, x_val)
            if val_loss < plateau_state.best_loss:
                plateau_state = plateau_state._replace(best_loss=val_loss, patience_counter=0)
            else:
                plateau_state = plateau_state._replace(patience_counter=plateau_state.patience_counter + 1)
            if plateau_state.patience_counter >= PLATEAU_PATIENCE_CHECKS:
                new_lr = max(plateau_state.lr * PLATEAU_REDUCTION_FACTOR, PLATEAU_MIN_LR)
                if new_lr < plateau_state.lr:
                    print(f"\n>>> Step {step}: Val loss stagnant. Reducing LR from {plateau_state.lr:.2e} to {new_lr:.2e} <<<\n")
                    # Update the plateau state with the new LR. This will be picked up at the start of the next loop.
                    plateau_state = plateau_state._replace(lr=new_lr, patience_counter=0, best_loss=jnp.inf)

        if step % LOG_EVERY_N_STEPS == 0 or step == TOTAL_TRAINING_STEPS:
            current_lr = schedule(step) if SCHEDULER_TYPE == 'warmup_cosine_decay' else plateau_state.lr
            print(f"  Step {step}/{TOTAL_TRAINING_STEPS} | TrainLoss={loss:.3e} | LR={current_lr:.2e}")

    return params, norm_stats_u

def generate_info_text():
    lr_sched_info = f"Scheduler: {SCHEDULER_TYPE.replace('_', ' ').title()}"
    lr_sched_info += f" (Peak LR={PEAK_LR:.1e})" if SCHEDULER_TYPE == 'warmup_cosine_decay' else f" (Start LR={PEAK_LR:.1e})"
    lines = [
        f"NN: {MODEL_WIDTH}x{MODEL_DEPTH} (Functional MLP, Glorot Init)",
        f"Training: Normalized Data + Initial Condition Loss",
        lr_sched_info,
        f"Loss: {'U(x) + dU/dx' if USE_DERIVATIVE_DATA_LOSS else 'U(x)'} ({LOSS_FUNCTION_TYPE.upper()})",
        f"TargetFn: {TARGET_N_FREQUENCIES} Freqs=[{TARGET_MIN_FREQ},{TARGET_MAX_FREQ}], Amp=[{TARGET_AMP_MIN},{TARGET_AMP_MAX}]"
    ]
    return "\n".join(lines)

def plot_final_result(params, target_fns, save_dir, norm_stats_u):
    print("  -> Generating final 1D interpolation-extrapolation plot...")
    center_u, scale_u = norm_stats_u
    activation_fn = {'gelu': jax.nn.gelu, 'tanh': jnp.tanh}.get(ACTIVATION_FUNCTION)
    
    x_plot = jnp.linspace(-EXTRAPOLATION_HALF_WIDTH, EXTRAPOLATION_HALF_WIDTH, 1000)
    
    # Use the pure functional forward pass
    pred_u_raw = vmap(partial(mlp_forward, params, activation_fn=activation_fn))(x_plot)
    
    # Denormalize predictions for plotting (since network predicts raw values that are normalized in loss)
    # The network itself doesn't output normalized values, the loss function normalizes them.
    # Therefore, we just plot the raw network output.
    u_pred = pred_u_raw
    u_truth = vmap(target_fns[0])(x_plot)

    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    ax.plot(x_plot, u_truth, 'k--', lw=2, label='Ground Truth U(x)')
    ax.plot(x_plot, u_pred, 'r', lw=2, alpha=0.8, label='Predicted U(x)')
    ax.axvspan(-EXTRAPOLATION_HALF_WIDTH, -INTERPOLATION_HALF_WIDTH, color='orange', alpha=0.1, label='Extrapolation Domain')
    ax.axvspan(INTERPOLATION_HALF_WIDTH, EXTRAPOLATION_HALF_WIDTH, color='orange', alpha=0.1)
    ax.axvspan(-INTERPOLATION_HALF_WIDTH, INTERPOLATION_HALF_WIDTH, color='gray', alpha=0.2, label='Interpolation Domain')
    ax.set_title(f"Final Model Performance - 1D Function Extrapolation")
    ax.grid(True, linestyle=':'); ax.legend(loc='upper right'); ax.set_xlabel("x"); ax.set_ylabel("u(x)")
    
    info_text = generate_info_text()
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9, va='top', bbox=dict(boxstyle='round', facecolor='aliceblue', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(save_dir, f"final_1d_plot.png"))
    plt.show()

def main():
    start_time = time.time()
    SAVE_DIR = os.path.join(os.getcwd(), RUN_NAME); os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Results will be saved in: {SAVE_DIR}")
    key = random.PRNGKey(SEED)

    key, subkey = random.split(key)
    target_fns = generate_random_target(key=subkey, n_freq=TARGET_N_FREQUENCIES, min_freq=TARGET_MIN_FREQ, max_freq=TARGET_MAX_FREQ, amp_min=TARGET_AMP_MIN, amp_max=TARGET_AMP_MAX)
    
    key, subkey = random.split(key)
    params, norm_stats_u = run_training(subkey, target_fns, SAVE_DIR)

    if params is not None:
        plot_final_result(params, target_fns, SAVE_DIR, norm_stats_u)
            
    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds.")

if __name__ == '__main__':
    main()

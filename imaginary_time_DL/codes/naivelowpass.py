# First, install necessary libraries
# pip install "jax[cuda12_pip]" -U # Or cuda11_pip, or cpu
# pip install flax optax matplotlib

import jax
import jax.numpy as jnp
import jax.nn as jnn
from jax import jit, vmap, random
from functools import partial
import flax.linen as nn
import optax
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Callable

# ================== MODEL CONFIGURATION ==================
# --- Architecture ---
MODEL_WIDTH = 10
MODEL_DEPTH = 10
ACTIVATION_FUNCTION = jnn.tanh

# --- Training Data & Target Function ---
N_TRAINING_SAMPLES = 5000
INTERPOLATION_HALF_WIDTH = 25.0 # The [-X, X] domain for training data points
# Parameters for the cosine sum target function
K_FQ = 1.0
FREQ_GAP = 20.0
AMPLITUDE_DOMINANCE_FACTOR = 10.0

# --- Derivative Constraints ---
# The number of points in the constraint domain to enforce the derivative constraint
N_CONSTRAINT_POINTS = 1000
# The domain for derivative constraints will be [-X*factor, X*factor]
CONSTRAINT_DOMAIN_FACTOR = 2.0
DERIVATIVE_WEIGHT = 1.0
USE_DYNAMIC_WEIGHTING = True
DYNAMIC_WEIGHT_WARMUP_EPOCHS = 0
ADAPTIVE_WEIGHT_SMOOTHING_FACTOR = 0.5
NORMALIZE_DERIVATIVE_PENALTY = False
DERIVATIVE_PENALTY_TYPE = 'mse' # 'mse' or 'log_quadratic'

# --- Optimizer & Training ---
EPOCHS = 20000
PEAK_LR = 1e-4
GRADIENT_CLIP_VALUE = 1.0
NUMERICAL_STABILITY_EPSILON = 1e-16

# --- Scheduler Configuration ---
SCHEDULER_CONFIG = {
    'USE_SCHEDULER': True,
    'PEAK_LR': PEAK_LR,
    'WARMUP_RATIO': 0.1,
    'END_LR_RATIO': 0.1,
}

print(f"JAX is using: {jax.default_backend()}")
print(f"Using Activation Function: {ACTIVATION_FUNCTION.__name__}")

# ================== NEURAL NETWORK DEFINITION ==================
class SimpleMLP(nn.Module):
    """A simple Multi-Layer Perceptron."""
    width: int
    depth: int
    activation_fn: Callable

    @nn.compact
    def __call__(self, x):
        for i in range(self.depth - 1):
            x = nn.Dense(features=self.width, name=f'Dense_{i}')(x)
            x = self.activation_fn(x)
        # Final layer to output a single value
        return nn.Dense(features=1, name=f'Dense_{self.depth-1}')(x)


# ================== DERIVATIVE & TARGET FUNCTIONS ==================

def compute_val_and_grad_jvp(model: SimpleMLP, params, x: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the model's output and its first derivative at a single point x using jax.jvp.
    `x` is expected to have a shape like (1,).
    """
    model_fn = lambda x_in: model.apply(params, x_in)
    tangent_in = (jnp.ones_like(x),)
    y, y_dot = jax.jvp(model_fn, (x,), tangent_in)
    return jnp.stack([y.squeeze(), y_dot.squeeze()])

def generate_cosine_sum_function(key, n_freq=12, r_low_to_high=0.25, k_fq=1.0, freq_gap=20.0, amplitude_dominance_factor=10.0):
    """Generates the target function (sum of cosines) and its low-frequency component."""
    n_low_freq = int(n_freq * r_low_to_high); n_high_freq = n_freq - n_low_freq
    low_freq_upper_bound = 1.0 / (2 * np.pi * np.e * k_fq)
    low_freqs_np = np.linspace(0.01 / k_fq, low_freq_upper_bound, n_low_freq)
    high_freq_start = low_freq_upper_bound + freq_gap; high_freq_end = high_freq_start + 25.0
    high_freqs_np = np.linspace(high_freq_start, high_freq_end, n_high_freq)
    key, low_key, high_key = random.split(key, 3)
    raw_low_amps = jax.random.uniform(low_key, (n_low_freq,), minval=0.5, maxval=1.5)
    raw_high_amps = jax.random.uniform(high_key, (n_high_freq,), minval=0.5, maxval=1.5)
    high_amps = raw_high_amps/jnp.sum(raw_high_amps) if jnp.sum(raw_high_amps)>0 else raw_high_amps
    low_amps = (raw_low_amps/jnp.sum(raw_low_amps))*amplitude_dominance_factor if jnp.sum(raw_low_amps)>0 else raw_low_amps
    all_freqs, all_amps = jnp.concatenate([low_freqs_np, high_freqs_np]), jnp.concatenate([low_amps, high_amps])

    def target_function(x): return jnp.sum(all_amps * jnp.cos(jnp.pi * x * all_freqs), axis=-1, keepdims=True)
    def low_freq_only_fn(x): return jnp.sum(low_amps * jnp.cos(jnp.pi * x * low_freqs_np), axis=-1, keepdims=True)
    return target_function, low_freq_only_fn, jnp.array(low_amps), jnp.array(low_freqs_np)

def calculate_target_val_and_grad(x, amps, freqs):
    """Calculates the 0th and 1st derivative of the cosine sum target function."""
    w_x = jnp.pi * x * freqs
    val = jnp.sum(amps * jnp.cos(w_x))
    grad = jnp.sum(amps * (-jnp.pi * freqs) * jnp.sin(w_x))
    return jnp.stack([val, grad])

def calculate_first_derivative_bound_at_zero(low_amps, low_freqs):
    """Calculates the max bound for the first derivative based on low-frequency components."""
    term = low_amps * (np.pi * low_freqs)
    return jnp.sum(jnp.abs(term))


# ================== TRAINING & OPTIMIZATION ==================
def create_optimizer_and_schedule(epochs, scheduler_config):
    if scheduler_config['USE_SCHEDULER']:
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=scheduler_config['PEAK_LR'],
            warmup_steps=int(epochs * scheduler_config['WARMUP_RATIO']),
            decay_steps=epochs - int(epochs * scheduler_config['WARMUP_RATIO']),
            end_value=scheduler_config['PEAK_LR'] * scheduler_config['END_LR_RATIO']
        )
    else:
        schedule = scheduler_config['PEAK_LR']
    optimizer = optax.chain(
        optax.clip_by_global_norm(GRADIENT_CLIP_VALUE),
        optax.adam(learning_rate=schedule)
    )
    return optimizer, schedule if callable(schedule) else (lambda e: schedule)

def train_model(
    model, init_params, key, epochs, scheduler_config, x_train, y_train, low_amps, low_freqs
):
    print(f"\n--- Starting Model Training ---")
    optimizer, schedule_fn = create_optimizer_and_schedule(epochs, scheduler_config)
    opt_state = optimizer.init(init_params)

    # --- Prepare Constraint Data ---
    constraint_domain_max = INTERPOLATION_HALF_WIDTH * CONSTRAINT_DOMAIN_FACTOR
    constraint_points = jnp.linspace(-constraint_domain_max, constraint_domain_max, N_CONSTRAINT_POINTS).reshape(-1, 1)

    vmapped_model_derivatives_fn = vmap(
        lambda p, x_point: compute_val_and_grad_jvp(model, p, x_point),
        in_axes=(None, 0) # Don't map over params (p), map over points (x_point)
    )

    vmapped_target_derivatives_fn = vmap(
        partial(calculate_target_val_and_grad, amps=low_amps, freqs=low_freqs),
        in_axes=0
    )
    target_derivs = vmapped_target_derivatives_fn(constraint_points.squeeze())
    M1_bound = calculate_first_derivative_bound_at_zero(low_amps, low_freqs)

    # --- Loss and Update Step ---
    def combined_loss_fn(params):
        y_pred = model.apply(params, x_train)
        data_loss = jnp.mean((y_pred - y_train) ** 2)

        model_derivs = vmapped_model_derivatives_fn(params, constraint_points)
        deriv_diff_sq = (model_derivs[:, 1] - target_derivs[:, 1]) ** 2

        penalty = deriv_diff_sq / jnp.maximum(M1_bound**2, NUMERICAL_STABILITY_EPSILON) if NORMALIZE_DERIVATIVE_PENALTY else deriv_diff_sq
        deriv_loss = jnp.log(1 + jnp.mean(penalty)) if DERIVATIVE_PENALTY_TYPE == 'log_quadratic' else jnp.mean(penalty)

        return data_loss, jnp.nan_to_num(deriv_loss, nan=1e6)

    @jit
    def update_step(params, opt_s, w_deriv_current):
        def total_loss_fn(p):
            data_l, deriv_l = combined_loss_fn(p)
            return data_l + w_deriv_current * deriv_l, (data_l, deriv_l)

        (loss, (data, deriv)), grads = jax.value_and_grad(total_loss_fn, has_aux=True)(params)
        updates, new_opt_s = optimizer.update(grads, opt_s, params)
        params = optax.apply_updates(params, updates)
        return params, new_opt_s, loss, data, deriv

    # --- Training Loop ---
    params = init_params
    w_deriv_state = 1.0
    for epoch in range(epochs):
        current_weight_for_step = w_deriv_state if USE_DYNAMIC_WEIGHTING and epoch >= DYNAMIC_WEIGHT_WARMUP_EPOCHS else (DERIVATIVE_WEIGHT if not USE_DYNAMIC_WEIGHTING else 0.0)
        params, opt_state, loss, data_loss, deriv_loss = update_step(params, opt_state, current_weight_for_step)

        if USE_DYNAMIC_WEIGHTING and epoch >= DYNAMIC_WEIGHT_WARMUP_EPOCHS:
            data_item, deriv_item = data_loss.item(), deriv_loss.item()
            if deriv_item > 0 and data_item > 0:
                target_w = data_item / (deriv_item + NUMERICAL_STABILITY_EPSILON)
                w_deriv_state = (1 - ADAPTIVE_WEIGHT_SMOOTHING_FACTOR) * w_deriv_state + ADAPTIVE_WEIGHT_SMOOTHING_FACTOR * target_w
            w_deriv_state = jnp.clip(w_deriv_state, 0, 1e5)

        if (epoch + 1) % 1000 == 0:
            active_weight = w_deriv_state if USE_DYNAMIC_WEIGHTING else DERIVATIVE_WEIGHT
            print(f"Epoch {epoch+1:5d}: Loss={loss.item():.2e} "
                  f"(Data: {data_loss.item():.2e}, Deriv: {current_weight_for_step*deriv_loss.item():.2e}) "
                  f"W_deriv={active_weight:.2e} LR={schedule_fn(epoch):.2e}")

    print("Finished model training.")
    return params


# ================== MAIN EXPERIMENT ==================
if __name__ == '__main__':
    start_time = time.time()
    key = random.PRNGKey(42)

    key, subkey = random.split(key)
    target_func, low_freq_func, low_amps, low_freqs = generate_cosine_sum_function(
        subkey, k_fq=K_FQ, freq_gap=FREQ_GAP, amplitude_dominance_factor=AMPLITUDE_DOMINANCE_FACTOR
    )
    pool_x = jnp.linspace(-INTERPOLATION_HALF_WIDTH, INTERPOLATION_HALF_WIDTH, 4000).reshape(-1, 1)
    key, subkey = random.split(key)
    indices = random.permutation(subkey, pool_x.shape[0])[:N_TRAINING_SAMPLES]
    x_train, y_train = pool_x[indices], target_func(pool_x)[indices]

    print("\n" + "="*50 + "\n" + " " * 15 + "RUNNING MODEL" + "\n" + "="*50 + "\n")
    model = SimpleMLP(width=MODEL_WIDTH, depth=MODEL_DEPTH, activation_fn=ACTIVATION_FUNCTION)
    key, subkey = random.split(key)
    init_params = model.init(subkey, jnp.ones((1,1)))

    trained_params = train_model(
        model, init_params, key,
        epochs=EPOCHS, scheduler_config=SCHEDULER_CONFIG,
        x_train=x_train, y_train=y_train,
        low_amps=low_amps, low_freqs=low_freqs,
    )

    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds.")

    PLOT_WIDTH = (CONSTRAINT_DOMAIN_FACTOR + 0.5) * INTERPOLATION_HALF_WIDTH
    x_plot = jnp.linspace(-PLOT_WIDTH, PLOT_WIDTH, 2000).reshape(-1, 1)
    y_true_plot = target_func(x_plot)
    y_low_freq_plot = low_freq_func(x_plot)
    y_pred_plot = model.apply(trained_params, x_plot)

    plt.figure(figsize=(14, 8))
    plt.title(f"NN with {ACTIVATION_FUNCTION.__name__} and JVP-based Derivative Constraints", fontsize=16)
    plt.plot(x_plot, y_true_plot, label='Ground Truth', color='black', lw=2.5, alpha=0.8)
    plt.plot(x_plot, y_low_freq_plot, label='Low-Freq Target', color='limegreen', ls=':', lw=2.5)
    plt.plot(x_plot, y_pred_plot, label='Final NN Prediction', color='deepskyblue', ls='-', lw=2.5)
    plt.axvspan(-INTERPOLATION_HALF_WIDTH, INTERPOLATION_HALF_WIDTH, color='gray', alpha=0.2, label='Training Domain')

    y_on_fit_domain = y_pred_plot[(x_plot >= -INTERPOLATION_HALF_WIDTH) & (x_plot <= INTERPOLATION_HALF_WIDTH)]
    margin = (jnp.max(y_on_fit_domain) - jnp.min(y_on_fit_domain)) * 0.2 if y_on_fit_domain.shape[0] > 0 else 5
    plt.ylim(jnp.min(y_on_fit_domain) - margin, jnp.max(y_on_fit_domain) + margin)
    plt.xlim(-PLOT_WIDTH, PLOT_WIDTH)
    plt.legend(loc='upper right'); plt.grid(True, linestyle=':'); plt.xlabel("x"); plt.ylabel("y")
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.show()

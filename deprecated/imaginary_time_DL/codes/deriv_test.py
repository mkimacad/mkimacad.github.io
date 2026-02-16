import jax
import jax.numpy as jnp
from jax import jit, vmap, value_and_grad, custom_vjp
from jax.experimental.jet import jet
import optax
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
import math

# --- 1. Helper Functions and Network Definition ---

@custom_vjp
def safe_logcosh_with_stable_grad(x):
    """
    Computes log(cosh(x)) with a stable forward pass and a custom,
    stable gradient (VJP) for the backward pass.
    """
    threshold = 15.0
    return jnp.where(
        jnp.abs(x) > threshold,
        jnp.abs(x) - jnp.log(2.0),
        jnp.log(jnp.cosh(x))
    )

def _safe_logcosh_fwd(x):
    return safe_logcosh_with_stable_grad(x), x

def _safe_logcosh_bwd(x, g):
    # The true gradient is tanh(x), which is always stable.
    return (g * jnp.tanh(x),)

safe_logcosh_with_stable_grad.defvjp(_safe_logcosh_fwd, _safe_logcosh_bwd)


def init_mlp_params(layer_widths, key):
    params = []
    for i in range(len(layer_widths) - 1):
        key, subkey = jax.random.split(key)
        std_dev = jnp.sqrt(2.0 / (layer_widths[i] + layer_widths[i+1]))
        W = jax.random.normal(subkey, (layer_widths[i+1], layer_widths[i])) * std_dev
        b = jnp.zeros((layer_widths[i+1],))
        params.append({'W': W, 'b': b})
    return params

def mlp_forward(params, x, activation_fn):
    *hidden_layers, last_layer = params
    for layer in hidden_layers:
        x = jnp.dot(layer['W'], x) + layer['b']
        x = activation_fn(x)
    x = jnp.dot(last_layer['W'], x) + last_layer['b']
    return x[0]

def target_func(x, A1, w1, A2, w2):
    return A1 * jnp.cos(w1 * x) + A2 * jnp.cos(w2 * x)

def get_analytical_nth_derivative(order, A1, w1, A2, w2):
    def deriv_component(A, w, x, order):
        if order % 4 == 0: return A * (w**order) * jnp.cos(w * x)
        elif order % 4 == 1: return -A * (w**order) * jnp.sin(w * x)
        elif order % 4 == 2: return -A * (w**order) * jnp.cos(w * x)
        else: return A * (w**order) * jnp.sin(w * x)
    def full_derivative(x):
        return deriv_component(A1, w1, x, order) + deriv_component(A2, w2, x, order)
    return full_derivative

# --- 2. JIT-Compatible Normalization Functions ---
def compute_normalization_stats(y_data, deriv_order, robust=True):
    """
    Compute normalization statistics with options for robust estimation.
    Returns only JAX arrays (no strings) for JIT compatibility.
    
    Args:
        y_data: Target derivative values
        deriv_order: Order of derivative (for informative printing)
        robust: If True, use robust statistics (median, MAD) instead of mean, std
    
    Returns:
        center, scale: JAX arrays for normalization
    """
    if robust:
        # Robust normalization using median and median absolute deviation (MAD)
        center = jnp.median(y_data)
        mad = jnp.median(jnp.abs(y_data - center))
        # Scale MAD to approximate standard deviation for normal distribution
        scale = mad * 1.4826  # 1/Φ^(-1)(3/4) where Φ is standard normal CDF
        # Avoid division by zero
        scale = jnp.maximum(scale, 1e-8)
        print(f"Robust normalization for derivative order {deriv_order}:")
        print(f"  Raw range: [{jnp.min(y_data):.2e}, {jnp.max(y_data):.2e}]")
        print(f"  Median: {center:.2e}, MAD: {mad:.2e}")
        print(f"  Normalized scale: {scale:.2e}")
    else:
        # Standard normalization using mean and standard deviation
        center = jnp.mean(y_data)
        scale = jnp.std(y_data)
        # Avoid division by zero
        scale = jnp.maximum(scale, 1e-8)
        print(f"Standard normalization for derivative order {deriv_order}:")
        print(f"  Raw range: [{jnp.min(y_data):.2e}, {jnp.max(y_data):.2e}]")
        print(f"  Mean: {center:.2e}, Std: {scale:.2e}")
    
    return center, scale

def normalize_data(y_data, center, scale):
    """Apply normalization using precomputed statistics."""
    return (y_data - center) / scale

def denormalize_data(y_normalized, center, scale):
    """Reverse normalization using precomputed statistics."""
    return y_normalized * scale + center

# --- 3. Loss Function with JIT-Compatible Normalization ---
def create_loss_function_taylor_normalized(deriv_order, initial_conditions, activation_fn, loss_fn_name, center, scale):
    """
    Creates a loss function with careful normalization handling.
    Uses separate center and scale parameters instead of a dict for JIT compatibility.
    """
    def loss_fn(params, x_data, y_data_normalized):
        f_nn_scalar = lambda x: mlp_forward(params, jnp.array([x]), activation_fn)

        # Initial condition loss (work in original scale)
        ic_loss = 0.0
        if deriv_order > 0:
            primals_ic = (0.0,)
            series_ic_len = max(1, deriv_order)
            series_ic = (jnp.ones_like(primals_ic[0]),) + (jnp.zeros_like(primals_ic[0]),) * (series_ic_len - 1)
            f_val_at_0, taylor_coeffs_at_0 = jet(f_nn_scalar, primals_ic, (series_ic,))

            # Function value IC (always in original scale)
            error_ic_0 = f_val_at_0 - initial_conditions[0]
            if loss_fn_name == 'mse': ic_loss += error_ic_0**2
            elif loss_fn_name == 'logcosh': ic_loss += safe_logcosh_with_stable_grad(error_ic_0)

            # Derivative ICs (convert to normalized scale for consistency)
            for k in range(1, deriv_order):
                pred_ic_k_raw = taylor_coeffs_at_0[k-1] * math.factorial(k)
                target_ic_k_raw = initial_conditions[k]
                
                # Normalize both prediction and target for this derivative order
                if k == deriv_order:
                    # Use the main normalization stats for the target derivative order
                    pred_ic_k_norm = normalize_data(pred_ic_k_raw, center, scale)
                    target_ic_k_norm = normalize_data(target_ic_k_raw, center, scale)
                else:
                    # For other derivative orders, use a simple scaling to avoid large mismatches
                    scale_factor = jnp.maximum(jnp.abs(target_ic_k_raw), 1e-8)
                    pred_ic_k_norm = pred_ic_k_raw / scale_factor
                    target_ic_k_norm = target_ic_k_raw / scale_factor
                
                error_ic_k = pred_ic_k_norm - target_ic_k_norm
                if loss_fn_name == 'mse': ic_loss += error_ic_k**2
                elif loss_fn_name == 'logcosh': ic_loss += safe_logcosh_with_stable_grad(error_ic_k)

        # Data loss (work in normalized scale)
        def single_point_data_loss(x, y_normalized):
            primals_data = (x,)
            series_data_len = max(1, deriv_order + 1)
            series_data = (jnp.ones_like(x),) + (jnp.zeros_like(x),) * (series_data_len - 1)
            _, taylor_coeffs_data = jet(f_nn_scalar, primals_data, (series_data,))

            # Get raw derivative prediction
            y_pred_raw = taylor_coeffs_data[deriv_order-1] * math.factorial(deriv_order)
            
            # Normalize the prediction to match the normalized target
            y_pred_normalized = normalize_data(y_pred_raw, center, scale)

            error = y_pred_normalized - y_normalized
            if loss_fn_name == 'mse': return error**2
            elif loss_fn_name == 'logcosh': return safe_logcosh_with_stable_grad(error)

        data_loss = jnp.mean(vmap(single_point_data_loss)(x_data, y_data_normalized))
        
        # Weight the IC loss appropriately
        ic_weight = 1e-3 if deriv_order >= 2 else 1e-6
        return data_loss + ic_weight * ic_loss
    
    return loss_fn

# --- 4. Training and Plotting with Normalization ---
@partial(jit, static_argnums=(4, 5))
def train_step_normalized(params, opt_state, x_data, y_data, loss_fn, optimizer):
    """JIT-compatible training step with simplified arguments."""
    loss, grads = value_and_grad(loss_fn)(params, x_data, y_data)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

def plot_results_normalized(nn_params, A1, w1, A2, w2, deriv_order, x_train, y_train_unscaled, 
                          center, scale, activation_fn):
    """Plotting function with separate center/scale parameters."""
    x_plot = np.linspace(-2, 2, 500)
    y_pred_plot, y_deriv_pred_plot_unscaled = np.zeros_like(x_plot), np.zeros_like(x_plot)
    f_nn_scalar_final = lambda x: mlp_forward(nn_params, jnp.array([x]), activation_fn)
    
    for i, x_val in enumerate(x_plot):
        primals, series_len = (x_val,), max(1, deriv_order + 1)
        series = (1.0,) + (0.0,) * (series_len - 1)
        f_val, taylor_coeffs = jet(f_nn_scalar_final, primals, (series,))
        y_pred_plot[i] = f_val
        
        # Get raw derivative and denormalize for plotting
        y_deriv_pred_raw = taylor_coeffs[deriv_order-1] * math.factorial(deriv_order)
        y_deriv_pred_plot_unscaled[i] = y_deriv_pred_raw  # Already in original scale
    
    y_true_plot = vmap(partial(target_func, A1=A1, w1=w1, A2=A2, w2=w2))(x_plot)
    true_deriv_func = get_analytical_nth_derivative(deriv_order, A1, w1, A2, w2)
    y_deriv_true_plot = vmap(true_deriv_func)(x_plot)
    
    # Compute error metrics
    train_error_raw = jnp.mean(jnp.abs(y_deriv_pred_plot_unscaled - y_deriv_true_plot))
    train_error_relative = train_error_raw / jnp.maximum(jnp.std(y_deriv_true_plot), 1e-8)
    
    plt.figure(figsize=(16, 6))
    
    plt.subplot(1, 3, 1)
    plt.plot(x_plot, y_true_plot, 'b-', label='True f(x)', linewidth=2)
    plt.plot(x_plot, y_pred_plot, 'r--', label='NN Approx f(x)', linewidth=2)
    plt.title('Function Comparison')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(x_plot, y_deriv_true_plot, 'b-', label=f'True f^({deriv_order})(x)', linewidth=2)
    plt.plot(x_plot, y_deriv_pred_plot_unscaled, 'r--', label=f'NN f^({deriv_order})(x)', linewidth=2)
    plt.scatter(x_train, y_train_unscaled, color='green', s=20, zorder=5, label='Last Epoch Training Data', alpha=0.6)
    plt.title(f'Derivative Comparison (Order {deriv_order})')
    plt.xlabel('x')
    plt.ylabel(f'f^({deriv_order})(x)')
    plt.legend()
    plt.grid(True)
    
    # Error plot
    plt.subplot(1, 3, 3)
    error_plot = y_deriv_pred_plot_unscaled - y_deriv_true_plot
    plt.plot(x_plot, error_plot, 'r-', linewidth=2)
    plt.title(f'Prediction Error\nMAE: {train_error_raw:.2e}, Relative: {train_error_relative:.2e}')
    plt.xlabel('x')
    plt.ylabel(f'Error in f^({deriv_order})(x)')
    plt.grid(True)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Error Statistics:")
    print(f"  Mean Absolute Error: {train_error_raw:.2e}")
    print(f"  Relative Error: {train_error_relative:.2e}")
    print(f"  Max Absolute Error: {jnp.max(jnp.abs(error_plot)):.2e}")

# --- 5. Main Training Function with True Resampling ---
def train_model_normalized(
    A1=1.0, w1=2.0, A2=0.5, w2=5.0,
    depth=3, width=64,
    activation_fn_name='tanh',
    deriv_order=2, num_samples=100,
    learning_rate=1e-3, num_epochs=80000,
    batch_size=32,
    gradient_clip_value=1.0, loss_fn_name='mse',
    use_scheduler=True,
    use_robust_normalization=True,
    use_x64=False, seed=42,
    resample_every_epoch=True
):
    if use_x64: jax.config.update("jax_enable_x64", True)
    else: jax.config.update("jax_enable_x64", False)
    
    activation_fns = {'tanh': jnp.tanh, 'gelu': jax.nn.gelu, 'relu': jax.nn.relu}
    activation_fn = activation_fns.get(activation_fn_name)
    if activation_fn is None: raise ValueError(f"Activation function '{activation_fn_name}' not recognized.")
    
    loss_fns = ['mse', 'logcosh']
    if loss_fn_name not in loss_fns: raise ValueError(f"Loss function '{loss_fn_name}' not recognized.")
    
    print(f"--- Starting Training with Normalization ---")
    print(f"Target function: {A1}*cos({w1}*x) + {A2}*cos({w2}*x)")
    print(f"Derivative order: {deriv_order}")
    print(f"Normalization: {'Robust (median/MAD)' if use_robust_normalization else 'Standard (mean/std)'}")
    print(f"Precision: {'x64' if use_x64 else 'x32'}, Activation: {activation_fn_name}, Loss: {loss_fn_name}")
    print(f"Resampling: {'Every epoch' if resample_every_epoch else 'Fixed samples'}")
    
    key = jax.random.PRNGKey(seed)
    
    # Generate initial training data to compute normalization statistics
    key, subkey = jax.random.split(key)
    x_initial = jax.random.uniform(subkey, (num_samples * 10,), minval=-1.0, maxval=1.0)  # More samples for better stats
    true_deriv_func = get_analytical_nth_derivative(deriv_order, A1, w1, A2, w2)
    y_initial_unscaled = vmap(true_deriv_func)(x_initial)
    
    # Compute normalization statistics once
    center, scale = compute_normalization_stats(y_initial_unscaled, deriv_order, use_robust_normalization)
    print(f"Normalization center: {center:.2e}, scale: {scale:.2e}")

    # Initial conditions
    initial_conditions = []
    if deriv_order > 0:
        for k in range(deriv_order):
            initial_conditions.append(get_analytical_nth_derivative(k, A1, w1, A2, w2)(0.0))
    
    # Network setup
    layer_widths = [1] + [width] * (depth - 1) + [1]
    key, subkey = jax.random.split(key)
    nn_params = init_mlp_params(layer_widths, subkey)
    
    # Optimizer setup
    if use_scheduler:
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0, peak_value=learning_rate, 
            warmup_steps=int(num_epochs*0.1), 
            decay_steps=int(num_epochs*0.9), 
            end_value=1e-7
        )
        optimizer = optax.chain(optax.clip(gradient_clip_value), optax.adam(learning_rate=schedule))
    else:
        optimizer = optax.chain(optax.clip(gradient_clip_value), optax.adam(learning_rate=learning_rate))
    
    opt_state = optimizer.init(nn_params)
    loss_function = create_loss_function_taylor_normalized(deriv_order, initial_conditions, activation_fn, loss_fn_name, center, scale)

    # Training loop with true resampling
    for epoch in range(num_epochs):
        # Generate new training data each epoch (true resampling)
        if resample_every_epoch or epoch == 0:
            key, subkey = jax.random.split(key)
            x_train = jax.random.uniform(subkey, (num_samples,), minval=-1.0, maxval=1.0)
            y_train_unscaled = vmap(true_deriv_func)(x_train)
            y_train_normalized = normalize_data(y_train_unscaled, center, scale)
        
        # Shuffle the current epoch's data for batch processing
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, num_samples)
        x_train_shuffled = x_train[perm]
        y_train_normalized_shuffled = y_train_normalized[perm]

        num_batches = num_samples // batch_size
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_x = x_train_shuffled[start_idx:end_idx]
            batch_y = y_train_normalized_shuffled[start_idx:end_idx]

            nn_params, opt_state, loss = train_step_normalized(
                nn_params, opt_state, batch_x, batch_y, loss_function, optimizer
            )

        if (epoch + 1) % 10000 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Last Batch Loss: {loss:.2e}")

    print("--- Training Finished ---")
    # Plot with the final epoch's training data
    plot_results_normalized(nn_params, A1, w1, A2, w2, deriv_order, x_train, y_train_unscaled, center, scale, activation_fn)
    
    return nn_params, center, scale

# --- 6. Main Execution Block ---
if __name__ == '__main__':
    train_model_normalized(
        A1=1.0, w1=1.0, A2=0.5, w2=2.0,
        depth=5,
        width=256,
        deriv_order=10,  
        learning_rate=1e-5,
        num_epochs=100000,
        batch_size=16,
        gradient_clip_value=0.5,
        use_scheduler=True,
        activation_fn_name='gelu',
        use_x64=True,
        loss_fn_name='logcosh',
        use_robust_normalization=True,
        num_samples=16,  
        resample_every_epoch=True
    )

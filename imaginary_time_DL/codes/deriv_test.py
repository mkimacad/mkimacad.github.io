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

# --- 2. Loss Function ---
def create_loss_function_taylor(deriv_order, initial_conditions, activation_fn, loss_fn_name):
    def loss_fn(params, x_data, y_data_unscaled):
        f_nn_scalar = lambda x: mlp_forward(params, jnp.array([x]), activation_fn)

        # Initial condition loss
        ic_loss = 0.0
        if deriv_order > 0:
            primals_ic = (0.0,)
            series_ic_len = max(1, deriv_order)
            series_ic = (jnp.ones_like(primals_ic[0]),) + (jnp.zeros_like(primals_ic[0]),) * (series_ic_len - 1)
            f_val_at_0, taylor_coeffs_at_0 = jet(f_nn_scalar, primals_ic, (series_ic,))

            error_ic_0 = f_val_at_0 - initial_conditions[0]
            if loss_fn_name == 'mse': ic_loss += error_ic_0**2
            elif loss_fn_name == 'logcosh': ic_loss += safe_logcosh_with_stable_grad(error_ic_0)

            for k in range(1, deriv_order):
                pred_ic_k = taylor_coeffs_at_0[k-1] * math.factorial(k)
                error_ic_k = pred_ic_k - initial_conditions[k]
                if loss_fn_name == 'mse': ic_loss += error_ic_k**2
                elif loss_fn_name == 'logcosh': ic_loss += safe_logcosh_with_stable_grad(error_ic_k)

        def single_point_data_loss(x, y_unscaled):
            primals_data = (x,)
            series_data_len = max(1, deriv_order + 1)
            series_data = (jnp.ones_like(x),) + (jnp.zeros_like(x),) * (series_data_len - 1)
            _, taylor_coeffs_data = jet(f_nn_scalar, primals_data, (series_data,))

            y_pred_unscaled = taylor_coeffs_data[-1] * math.factorial(deriv_order)

            error = y_pred_unscaled - y_unscaled
            if loss_fn_name == 'mse': return error**2
            elif loss_fn_name == 'logcosh': return safe_logcosh_with_stable_grad(error)

        data_loss = jnp.mean(vmap(single_point_data_loss)(x_data, y_data_unscaled))
        return data_loss + 1e-6 * ic_loss
    return loss_fn

# --- 3. Training and Plotting ---
@partial(jit, static_argnums=(4, 5))
def train_step(params, opt_state, x_data, y_data, loss_fn, optimizer):
    loss, grads = value_and_grad(loss_fn)(params, x_data, y_data)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

def plot_results(nn_params, A1, w1, A2, w2, deriv_order, x_train, y_train_unscaled, activation_fn):
    x_plot = np.linspace(-2, 2, 500)
    y_pred_plot, y_deriv_pred_plot_unscaled = np.zeros_like(x_plot), np.zeros_like(x_plot)
    f_nn_scalar_final = lambda x: mlp_forward(nn_params, jnp.array([x]), activation_fn)
    for i, x_val in enumerate(x_plot):
        primals, series_len = (x_val,), max(1, deriv_order + 1)
        series = (1.0,) + (0.0,) * (series_len - 1)
        f_val, taylor_coeffs = jet(f_nn_scalar_final, primals, (series,))
        y_pred_plot[i], y_deriv_pred_plot_unscaled[i] = f_val, taylor_coeffs[-1] * math.factorial(deriv_order)
    y_true_plot = vmap(partial(target_func, A1=A1, w1=w1, A2=A2, w2=w2))(x_plot)
    true_deriv_func = get_analytical_nth_derivative(deriv_order, A1, w1, A2, w2)
    y_deriv_true_plot = vmap(true_deriv_func)(x_plot)
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(x_plot, y_true_plot, 'b-', label='True f(x)'); plt.plot(x_plot, y_pred_plot, 'r--', label='NN Approx f(x)'); plt.title('Function Comparison'); plt.legend(); plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(x_plot, y_deriv_true_plot, 'b-', label=f'True f^({deriv_order})(x)'); plt.plot(x_plot, y_deriv_pred_plot_unscaled, 'r--', label=f'NN f^({deriv_order})(x)'); plt.scatter(x_train, y_train_unscaled, color='green', s=10, zorder=5, label='Training Data'); plt.title(f'Derivative Comparison (Order {deriv_order})'); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.show()

# --- 4. Main Training Function ---
def train_model(
    A1=1.0, w1=2.0, A2=0.5, w2=5.0,
    depth=3, width=64,
    activation_fn_name='tanh',
    deriv_order=2, num_samples=100,
    learning_rate=1e-3, num_epochs=80000,
    batch_size=32,
    gradient_clip_value=1.0, loss_fn_name='mse',
    use_scheduler=True,
    use_x64=False, seed=42
):
    if use_x64: jax.config.update("jax_enable_x64", True)
    else: jax.config.update("jax_enable_x64", False)
    activation_fns = {'tanh': jnp.tanh, 'gelu': jax.nn.gelu, 'relu': jax.nn.relu}; activation_fn = activation_fns.get(activation_fn_name)
    if activation_fn is None: raise ValueError(f"Activation function '{activation_fn_name}' not recognized.")
    loss_fns = ['mse', 'logcosh'];
    if loss_fn_name not in loss_fns: raise ValueError(f"Loss function '{loss_fn_name}' not recognized.")
    print(f"--- Starting Training ---"); print(f"Precision: {'x64' if use_x64 else 'x32'}, Activation: {activation_fn_name}, Loss: {loss_fn_name}, Scheduler: {use_scheduler}")
    key = jax.random.PRNGKey(seed); key, subkey = jax.random.split(key)
    x_train = jax.random.uniform(subkey, (num_samples,), minval=-1.0, maxval=1.0)
    true_deriv_func = get_analytical_nth_derivative(deriv_order, A1, w1, A2, w2); y_train_unscaled = vmap(true_deriv_func)(x_train)

    initial_conditions = []
    if deriv_order > 0:
        for k in range(deriv_order): initial_conditions.append(get_analytical_nth_derivative(k, A1, w1, A2, w2)(0.0))
    layer_widths = [1] + [width] * (depth - 1) + [1]; key, subkey = jax.random.split(key); nn_params = init_mlp_params(layer_widths, subkey)
    if use_scheduler: schedule = optax.warmup_cosine_decay_schedule(init_value=0.0, peak_value=learning_rate, warmup_steps=int(num_epochs*0.1), decay_steps=int(num_epochs*0.9), end_value=1e-7); optimizer = optax.chain(optax.clip(gradient_clip_value), optax.adam(learning_rate=schedule))
    else: optimizer = optax.chain(optax.clip(gradient_clip_value), optax.adam(learning_rate=learning_rate))
    opt_state = optimizer.init(nn_params)

    loss_function = create_loss_function_taylor(deriv_order, initial_conditions, activation_fn, loss_fn_name)

    for epoch in range(num_epochs):
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, num_samples)
        x_train_shuffled = x_train[perm]
        y_train_unscaled_shuffled = y_train_unscaled[perm] # Use unscaled y

        num_batches = num_samples // batch_size
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_x = x_train_shuffled[start_idx:end_idx]
            batch_y = y_train_unscaled_shuffled[start_idx:end_idx] # Use unscaled y

            nn_params, opt_state, loss = train_step(
                nn_params, opt_state, batch_x, batch_y, loss_function, optimizer
            )

        if (epoch + 1) % 10000 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Last Batch Loss: {loss:.2e}")

    print("--- Training Finished ---")
    # Pass the unscaled training data to the plotting function
    plot_results(nn_params, A1, w1, A2, w2, deriv_order, x_train, y_train_unscaled, activation_fn)


# --- 5. Main Execution Block ---
if __name__ == '__main__':
    train_model(
        depth=5,
        width=256,
        deriv_order=1,
        learning_rate=1e-4,
        num_epochs=400000,
        batch_size=32,
        gradient_clip_value=1.0,
        use_scheduler=True,
        activation_fn_name='gelu',
        use_x64=True,
        loss_fn_name='logcosh',
        w1=0.08,
        w2=0.2
    )

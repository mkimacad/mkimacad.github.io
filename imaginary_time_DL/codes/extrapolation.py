# First, install necessary libraries
# pip install "jax[cuda12_pip]" -U # Or cuda11_pip, or cpu
# pip install flax optax matplotlib msgpack

import jax
import jax.numpy as jnp
import jax.nn as jnn
from jax import jit, vmap, random, grad, tree_util
from functools import partial
import flax.linen as nn
from flax.serialization import to_bytes, from_bytes
import optax
import matplotlib.pyplot as plt
import time
import os
import itertools
import json
import math
from typing import Callable, Tuple, Dict, Any, List

# --- Enable 32-bit for compatibility and speed ---
jax.config.update("jax_enable_x64", False)

# ================== GLOBAL CONFIGURATION ==================
GROUPS_TO_RUN = ['UV2', 6, 'U_ONLY']
RUN_DYNAMIC_WEIGHTING_CONFIGS = True
SAVE_MODELS = True

# --- Curriculum Control Parameters ---
TOTAL_CURRICULUM_REPETITIONS = 3
EPOCHS_PER_PHASE = 15000
FINAL_POLISH_EPOCHS = 30000
LOG_EVERY_N_EPOCHS = max(EPOCHS_PER_PHASE//3, 2000) # This now only controls the print frequency within a phase

# --- Data-Fitting Curriculum Configuration ---
USE_SEQUENTIAL_TRAINING = True
SEQUENTIAL_INTERVAL_SIZE = 0.5 # Interval size for exploring the [-1, 1] box

# --- Physics Finetuning & Constraint Configuration ---
USE_PHYSICS_FINETUNING_CURRICULUM = True
PHYSICS_SHELL_STEP_SIZE = 2.0
N_CONSTRAINT_POINTS_OVERRIDE = 1000
# Number of extra points from the extrapolation shell to add to every physics phase
N_EXTRAPOLATION_CONSTRAINT_POINTS = N_CONSTRAINT_POINTS_OVERRIDE
USE_KINK_PENALTY = True # Global switch for the new higher-order penalty
KINK_PENALTY_WEIGHT = 20.0 # Weight for the kink penalty term

# --- Kink Penalty Point Generation Strategy ---
FOCUS_KINK_PENALTY_ON_BOUNDARY = True # If True, kink penalty points are sampled from the data boundary.
N_KINK_PENALTY_BOUNDARY_POINTS = N_CONSTRAINT_POINTS_OVERRIDE # Number of points for this focused penalty.

# --- Model and Domain Configuration ---
ACTIVATION_FUNCTION = jnn.gelu
INTERPOLATION_HALF_WIDTH = 1.0  # Interpolation data is in [-1, 1]^2 box. Not tunable.
EXTRAPOLATION_HALF_WIDTH = 10.0 # Tunable half-width for the extrapolation boundary.
MODEL_WIDTH_OVERRIDE = 64
MODEL_DEPTH_OVERRIDE = 6
N_TRAINING_SAMPLES_OVERRIDE = 2000

# --- Hyperparameters ---
UV_DATA_WEIGHT = 1.0; POTENTIAL_DATA_WEIGHT = 1.0; LAPLACE_WEIGHT = 50.0
CR_PENALTY_WEIGHT = 50.0; U_DERIV_DATA_WEIGHT = 0.5
DYNAMIC_WEIGHT_EMA_ALPHA = 0.99
PEAK_LR = 1e-4; GRADIENT_CLIP_VALUE = 1.0

print(f"JAX is using: {jax.default_backend()} with 32-bit precision.")
print(f"### Using 'Circling Explore-Consolidate' with True Curriculum Resumption ###")
print(f"### Data domain is fixed to [-1, 1]^2. Extrapolation domain is [{-EXTRAPOLATION_HALF_WIDTH}, {EXTRAPOLATION_HALF_WIDTH}]^2. ###")
if USE_KINK_PENALTY:
    strategy = "the data domain boundary" if FOCUS_KINK_PENALTY_ON_BOUNDARY else "the general physics domain"
    print(f"### KINK PENALTY ENABLED (w={KINK_PENALTY_WEIGHT}), focused on {strategy}. ###")


# ================== NEURAL NETWORK ARCHITECTURES AND CORE HELPERS ==================
class ScalarPotentialMLP(nn.Module):
    width: int; depth: int; activation_fn: Callable
    @nn.compact
    def __call__(self, z): x=z; [x := self.activation_fn(nn.Dense(self.width)(x)) for _ in range(self.depth-1)]; return nn.Dense(1)(x).squeeze()
class DirectUVMLP(nn.Module):
    width: int; depth: int; activation_fn: Callable
    @nn.compact
    def __call__(self, z): x=z; [x := self.activation_fn(nn.Dense(self.width)(x)) for _ in range(self.depth-1)]; return nn.Dense(2)(x)
class UOnlyMLP(nn.Module):
    width: int; depth: int; activation_fn: Callable
    @nn.compact
    def __call__(self, z): x=z; [x := self.activation_fn(nn.Dense(self.width)(x)) for _ in range(self.depth-1)]; return nn.Dense(1)(x).squeeze()
def get_uv_from_potential(model, params, z): grad_psi = grad(lambda z_in: model.apply(params, z_in))(z); return jnp.stack([grad_psi[0], -grad_psi[1]])
def get_u_and_du_dx(model, params, z): u = model.apply(params, z); du_dx = grad(lambda z_in: model.apply(params, z_in))(z)[0]; return u, du_dx
def get_uv_and_cr_grads(model, params, z): u_fn = lambda z_in: model.apply(params, z_in)[0]; v_fn = lambda z_in: model.apply(params, z_in)[1]; du_dx, du_dy = grad(u_fn)(z); dv_dx, dv_dy = grad(v_fn)(z); return (du_dx - dv_dy)**2 + (du_dy + dv_dx)**2

# --- HELPERS FOR KINK PENALTY ---
@partial(jit, static_argnames=['model'])
def get_laplacian_and_grad_laplacian_sq_norm(model, params, z):
    potential_fn = lambda z_in: model.apply(params, z_in)
    def laplacian_fn(z_in): return jnp.trace(jax.hessian(potential_fn)(z_in))
    laplacian_at_z = laplacian_fn(z)
    grad_laplacian_at_z = jax.grad(laplacian_fn)(z)
    return laplacian_at_z, jnp.sum(grad_laplacian_at_z**2)

@partial(jit, static_argnames=['model'])
def get_cr_and_grad_cr_sq_norm(model, params, z):
    u_fn = lambda z_in: model.apply(params, z_in)[0]
    v_fn = lambda z_in: model.apply(params, z_in)[1]
    def cr_residual_sq_fn(z_in):
        du_dx_i, du_dy_i = grad(u_fn)(z_in)
        dv_dx_i, dv_dy_i = grad(v_fn)(z_in)
        return (du_dx_i - dv_dy_i)**2 + (du_dy_i + dv_dx_i)**2
    cr_residual_sq_at_z = cr_residual_sq_fn(z)
    grad_cr_residual_sq_at_z = jax.grad(cr_residual_sq_fn)(z)
    return cr_residual_sq_at_z, jnp.sum(grad_cr_residual_sq_at_z**2)
    
def generate_initial_target(key, k_fq_scale=0.1, amp_dom_factor=10.0, n_freq=6) -> Tuple[Callable, Callable, Callable]:
    ks = jnp.linspace(0.5, k_fq_scale, n_freq); raw_amps = random.uniform(random.split(key)[1], (n_freq,), minval=0.5, maxval=1.5); amps = (raw_amps / jnp.sum(raw_amps)) * amp_dom_factor
    def _uv_fn(z): x, y = z[..., 0:1], z[..., 1:2]; u = jnp.sum(amps * jnp.cos(ks * x) * jnp.cosh(ks * y), axis=-1); v = jnp.sum(-amps * jnp.sin(ks * x) * jnp.sinh(ks * y), axis=-1); return jnp.stack([u, v], axis=-1)
    def _psi_fn(z): x, y = z[..., 0:1], z[..., 1:2]; return jnp.sum((amps / (ks + 1e-8)) * jnp.sin(ks * x) * jnp.cosh(ks * y), axis=-1)
    def _du_dx_fn(z): x, y = z[..., 0:1], z[..., 1:2]; return jnp.sum(-amps * ks * jnp.sin(ks * x) * jnp.cosh(ks * y), axis=-1)
    return _uv_fn, _psi_fn, _du_dx_fn
@jit
def get_grad_norm_statistic(grads): leaves = tree_util.tree_leaves(grads); norms = [jnp.max(jnp.abs(leaf)) for leaf in leaves if jnp.issubdtype(leaf.dtype, jnp.number)]; return jnp.max(jnp.stack(norms)) if norms else 0.0
@partial(jit, static_argnames=['optimizer', 'loss_names', 'loss_functions'])
def static_training_step(params, opt_state, optimizer, loss_names: Tuple, loss_functions: Tuple, weights: Dict):
    def total_loss_fn(p): losses = {name: fn(p) for name, fn in zip(loss_names, loss_functions)}; total = jnp.sum(jnp.array([weights.get(name, 0.0) * losses[name] for name in loss_names])); return total, losses
    (loss, losses), grads = jax.value_and_grad(total_loss_fn, has_aux=True)(params); updates, new_opt_s = optimizer.update(grads, opt_state, params); params = optax.apply_updates(params, updates); return params, new_opt_s, loss, losses
@partial(jit, static_argnames=['optimizer', 'loss_names', 'loss_functions', 'data_loss_names', 'physics_loss_names'])
def dynamic_training_step(params, opt_state, dw_state, optimizer, loss_names: Tuple, loss_functions: Tuple, data_loss_names: Tuple, physics_loss_names: Tuple):
    grads, losses = {}, {}; [ (losses.__setitem__(name, res[0]), grads.__setitem__(name, res[1])) for name, fn in zip(loss_names, loss_functions) for res in [jax.value_and_grad(fn)(params)] ]
    ref_grad_stat = jnp.mean(jnp.array([get_grad_norm_statistic(grads[name]) for name in data_loss_names if name in grads and name in grads])); ref_grad_stat = jax.lax.stop_gradient(ref_grad_stat)
    total_grads = tree_util.tree_map(jnp.zeros_like, params); [ total_grads := tree_util.tree_map(lambda x,y: x+y, total_grads, grads[name]) for name in data_loss_names if name in grads]
    total_loss = jnp.sum(jnp.array([losses[name] for name in data_loss_names if name in losses]))
    new_dw_state = {}
    for name in physics_loss_names:
        if name in grads:
            stat = get_grad_norm_statistic(grads[name]); w_raw = ref_grad_stat / (stat + 1e-8); w = DYNAMIC_WEIGHT_EMA_ALPHA * dw_state.get(name, 1.0) + (1 - DYNAMIC_WEIGHT_EMA_ALPHA) * w_raw; new_dw_state[name] = w
            total_grads = tree_util.tree_map(lambda g,h: g + w*h, total_grads, grads[name]); total_loss += w * losses[name]
    updates, new_opt_s = optimizer.update(total_grads, opt_state, params); params = optax.apply_updates(params, updates); return params, new_opt_s, new_dw_state, total_loss, losses
def calculate_run_parameters():
    # Since INTERPOLATION_HALF_WIDTH is fixed, these calculations are now constant unless overridden.
    model_width = MODEL_WIDTH_OVERRIDE; model_depth = MODEL_DEPTH_OVERRIDE; n_training_samples = N_TRAINING_SAMPLES_OVERRIDE; n_constraint_points = N_CONSTRAINT_POINTS_OVERRIDE
    print(f"  -> Using Width={model_width}, Depth={model_depth}, Train Samples={n_training_samples}, Constraint Points={n_constraint_points}")
    return {'width': model_width, 'depth': model_depth,'n_train': n_training_samples, 'n_constraint': n_constraint_points}

# =========================================================================================
# --- CHECKPOINTING & CURRICULUM HELPERS ---
# =========================================================================================
def save_checkpoint(path: str, params, opt_state, dw_state, phase_index: int, key):
    """Saves the full training state. Serializes the entire state dict in one go."""
    # The dictionary should contain the raw JAX PyTrees and the key as an array.
    state_dict = {
        'params': params,
        'opt_state': opt_state,
        'dw_state': dw_state,
        'phase_index': phase_index,
        'key': key.__array__() # Save key as its array representation for robustness
    }
    # Use flax.serialization.to_bytes on the entire dictionary.
    serialized_state = to_bytes(state_dict)
    with open(path, "wb") as f:
        f.write(serialized_state)

def load_checkpoint(path: str, init_params, init_opt_state, init_dw_state, init_key):
    """Loads the training state. Handles both new and old (faulty) checkpoint formats."""
    if not os.path.exists(path):
        return init_params, init_opt_state, init_dw_state, 0, init_key

    with open(path, "rb") as f:
        loaded_bytes = f.read()

    # Define the structure for deserialization, using initial states as templates.
    # The key template should be its array form.
    template_state_new = {
        'params': init_params,
        'opt_state': init_opt_state,
        'dw_state': init_dw_state,
        'phase_index': 0,
        'key': init_key.__array__()
    }

    try:
        # --- Attempt 1: Try to load as the NEW, correct format ---
        state_dict = from_bytes(template_state_new, loaded_bytes)
        # Re-constitute the key from its array representation
        key = jnp.asarray(state_dict['key'], dtype=jnp.uint32)
        print(f"  -> Checkpoint found (new format). Resuming from phase {state_dict['phase_index']}.")
        return state_dict['params'], state_dict['opt_state'], state_dict['dw_state'], state_dict['phase_index'], key

    except Exception as e_new:
        # --- Attempt 2: If new format fails, try to load as the OLD, faulty format ---
        print(f"  -> Could not load as new format ({e_new}). Trying old format...")
        try:
            # Define the template for the old, doubly-serialized format
            template_state_old = {
                'params': b'', 'opt_state': b'', 'dw_state': init_dw_state,
                'phase_index': 0, 'key': init_key.__array__()
            }
            outer_dict = from_bytes(template_state_old, loaded_bytes)

            # Deserialize the inner PyTrees from their raw bytes
            params = from_bytes(init_params, outer_dict['params'])
            opt_state = from_bytes(init_opt_state, outer_dict['opt_state'])
            dw_state = outer_dict['dw_state']
            phase_index = outer_dict['phase_index']
            
            # **THE FIX IS HERE**: Directly use the loaded array as the key.
            # Do not try to re-create it with PRNGKey().
            key = jnp.asarray(outer_dict['key'], dtype=jnp.uint32)

            print(f"  -> Checkpoint found (old format). Resuming from phase {phase_index}.")
            # Save the checkpoint in the new, correct format immediately
            print("  -> Resaving checkpoint in the new, corrected format...")
            save_checkpoint(path, params, opt_state, dw_state, phase_index, key)

            return params, opt_state, dw_state, phase_index, key

        except Exception as e_old:
            print(f"  -> WARNING: Could not load checkpoint from '{path}' using either new or old format. Error: {e_old}. Starting from scratch.")
            return init_params, init_opt_state, init_dw_state, 0, init_key

def _generate_circling_curriculum(half_width: float, interval_size: float) -> List[List[Tuple[Tuple[float, float], Tuple[float, float]]]]:
    s = interval_size / 2.0; L_max = int(jnp.floor((half_width / s - 1) / 2.0))
    def get_interval(k, s_val):
        if k > 0: return ((2 * k - 1) * s_val, (2 * k + 1) * s_val)
        if k < 0: return ((2 * k + 1) * s_val, (2 * k - 1) * s_val)
        return (-s_val, s_val)
    all_shells = []
    for L in range(L_max + 1):
        indices = [(i, j) for i in range(-L, L + 1) for j in range(-L, L + 1) if max(abs(i), abs(j)) == L]
        all_shells.append([(get_interval(i, s), get_interval(j, s)) for i, j in indices])
    return all_shells

def _generate_circling_finetune_curriculum(inner_hw: float, outer_hw: float, step_size: float) -> List[List[Tuple[Tuple[float, float], Tuple[float, float]]]]:
    s = step_size / 2.0; L_max = int(jnp.floor((outer_hw / s - 1) / 2.0))
    def get_interval(k, s_val):
        if k > 0: return ((2 * k - 1) * s_val, (2 * k + 1) * s_val)
        if k < 0: return ((2 * k + 1) * s_val, (2 * k - 1) * s_val)
        return (-s_val, s_val)
    finetune_shells = []
    for L in range(L_max + 1):
        indices = [(i, j) for i in range(-L, L + 1) for j in range(-L, L + 1) if max(abs(i), abs(j)) == L]
        shell_boxes = []
        for i, j in indices:
            x_int, y_int = get_interval(i, s), get_interval(j, s)
            if max(abs(x_int[0]), abs(x_int[1]), abs(y_int[0]), abs(y_int[1])) > inner_hw: shell_boxes.append((x_int, y_int))
        if shell_boxes: finetune_shells.append(shell_boxes)
    return finetune_shells
    
def _build_curriculum_plan():
    print("Building curriculum plan...")
    plan = []
    for rep in range(TOTAL_CURRICULUM_REPETITIONS):
        data_curriculum_shells = _generate_circling_curriculum(INTERPOLATION_HALF_WIDTH, SEQUENTIAL_INTERVAL_SIZE)
        cumulative_data_boxes = []
        for l_idx, shell in enumerate(data_curriculum_shells):
            for b_idx, box in enumerate(shell): plan.append({'type': 'data_explore', 'rep': rep, 'shell': l_idx, 'box': b_idx, 'box_def': [box], 'epochs': EPOCHS_PER_PHASE})
            cumulative_data_boxes.extend(shell); plan.append({'type': 'data_consolidate', 'rep': rep, 'shell': l_idx, 'cumulative_boxes': list(cumulative_data_boxes), 'epochs': EPOCHS_PER_PHASE})
        if USE_PHYSICS_FINETUNING_CURRICULUM:
            finetune_shells = _generate_circling_finetune_curriculum(INTERPOLATION_HALF_WIDTH, EXTRAPOLATION_HALF_WIDTH, PHYSICS_SHELL_STEP_SIZE)
            cumulative_physics_boxes = []
            for l_idx, shell in enumerate(finetune_shells):
                for b_idx, box in enumerate(shell): plan.append({'type': 'phys_explore', 'rep': rep, 'shell': l_idx, 'box': b_idx, 'box_def': [box], 'data_boxes': cumulative_data_boxes, 'epochs': EPOCHS_PER_PHASE})
                cumulative_physics_boxes.extend(shell); plan.append({'type': 'phys_consolidate', 'rep': rep, 'shell': l_idx, 'cumulative_boxes': list(cumulative_physics_boxes), 'data_boxes': cumulative_data_boxes, 'epochs': EPOCHS_PER_PHASE})
        plan.append({'type': 'final_polish', 'rep': rep, 'data_boxes': cumulative_data_boxes, 'epochs': FINAL_POLISH_EPOCHS})
    print(f"Curriculum plan built with {len(plan)} phases.")
    return plan

def sample_from_shell(key, num_points: int, outer_hw: float, inner_hw: float) -> jnp.ndarray:
    points_found = 0; accepted_points = []
    if outer_hw <= inner_hw: return jnp.empty((0, 2))
    area_ratio = (outer_hw**2 - inner_hw**2) / outer_hw**2; buffer_factor = 2.0 / max(area_ratio, 0.01)
    while points_found < num_points:
        key, subkey = random.split(key); needed = num_points - points_found; sample_size = int(needed * buffer_factor)
        candidates = random.uniform(subkey, (sample_size, 2), minval=-outer_hw, maxval=outer_hw)
        max_coords = jnp.max(jnp.abs(candidates), axis=1); mask = max_coords > inner_hw
        accepted = candidates[mask]; accepted_points.append(accepted); points_found += accepted.shape[0]
    return jnp.concatenate(accepted_points, axis=0)[:num_points]

# --- NEW SAMPLING HELPER ---
def sample_from_box_boundary(key, num_points: int, half_width: float) -> jnp.ndarray:
    """Samples points uniformly from the boundary of a square centered at the origin."""
    if num_points == 0: return jnp.empty((0, 2))
    # Distribute points among the four sides
    points_per_side = num_points // 4; remainder = num_points % 4
    side_counts = jnp.array([points_per_side] * 4) + jnp.array([1] * remainder + [0] * (4 - remainder))
    keys = random.split(key, 5); key, side_keys = keys[0], keys[1:]
    all_points = []
    # Bottom/Top sides
    x_bot = random.uniform(side_keys[0], (side_counts[0], 1), minval=-half_width, maxval=half_width)
    all_points.append(jnp.concatenate([x_bot, jnp.full_like(x_bot, -half_width)], axis=1))
    x_top = random.uniform(side_keys[1], (side_counts[1], 1), minval=-half_width, maxval=half_width)
    all_points.append(jnp.concatenate([x_top, jnp.full_like(x_top, half_width)], axis=1))
    # Left/Right sides
    y_left = random.uniform(side_keys[2], (side_counts[2], 1), minval=-half_width, maxval=half_width)
    all_points.append(jnp.concatenate([jnp.full_like(y_left, -half_width), y_left], axis=1))
    y_right = random.uniform(side_keys[3], (side_counts[3], 1), minval=-half_width, maxval=half_width)
    all_points.append(jnp.concatenate([jnp.full_like(y_right, half_width), y_right], axis=1))
    # Concatenate and shuffle
    final_points = jnp.concatenate(all_points, axis=0)
    return random.permutation(key, final_points)


### MAIN TRAINING FUNCTION ###
def run_sequential_training_for_config(key, config, target_fns, use_dynamic_weighting, save_dir, group_id, run_name_for_save):
    uv_target_fn, psi_target_fn, dudx_target_fn = target_fns
    run_params = calculate_run_parameters()
    model_width, model_depth, n_train, n_constraint = (run_params['width'], run_params['depth'], run_params['n_train'], run_params['n_constraint'])
    model_type = config.get('model_type', 'Potential')
    if model_type == 'UV': model = DirectUVMLP(width=model_width, depth=model_depth, activation_fn=ACTIVATION_FUNCTION)
    elif model_type == 'U_ONLY': model = UOnlyMLP(width=model_width, depth=model_depth, activation_fn=ACTIVATION_FUNCTION)
    else: model = ScalarPotentialMLP(width=model_width, depth=model_depth, activation_fn=ACTIVATION_FUNCTION)
    
    key, subkey = random.split(key); init_params = model.init(subkey, jnp.ones((1, 2)))
    optimizer = optax.chain(optax.clip_by_global_norm(GRADIENT_CLIP_VALUE), optax.adam(learning_rate=PEAK_LR))
    init_opt_state = optimizer.init(init_params)
    init_dw_state = {}
    checkpoint_path = os.path.join(save_dir, f"checkpoint_group_{group_id}_{run_name_for_save}.msgpack")
    params, opt_state, dw_state, loaded_phase_idx, key = load_checkpoint(checkpoint_path, init_params, init_opt_state, init_dw_state, key)
    loss_history = []
    curriculum_plan = _build_curriculum_plan()

    def _sample_from_boxes(k, num_points, boxes):
        num_boxes = len(boxes)
        if num_boxes == 0: return jnp.empty((0, 2))
        points_per_box = num_points // num_boxes; remaining = num_points % num_boxes
        all_samples = []
        for i, box in enumerate(boxes):
            k, subkey = random.split(k); n_samples = points_per_box + (1 if i < remaining else 0)
            if n_samples == 0: continue
            min_v = jnp.array([box[0][0], box[1][0]]); max_v = jnp.array([box[0][1], box[1][1]])
            all_samples.append(random.uniform(subkey, (n_samples, 2), minval=min_v, maxval=max_v))
        return jnp.concatenate(all_samples, axis=0) if all_samples else jnp.empty((0, 2))

    def _create_full_loss_fns(z_train, z_physics, z_kink):
        loss_fns, static_weights, data_loss_names, physics_loss_names = {}, {}, [], []
        
        # Determine which points to use for the kink penalty
        kink_points = z_kink if FOCUS_KINK_PENALTY_ON_BOUNDARY and z_kink.shape[0] > 0 else z_physics
        
        if model_type == 'Potential':
            if config.get('USE_UV_DATA_LOSS', False) and z_train.shape[0] > 0:
                y_target = uv_target_fn(z_train); vmapped_uv = vmap(get_uv_from_potential, (None, None, 0))
                loss_fns['uv_data'] = lambda p: jnp.mean((vmapped_uv(model, p, z_train) - y_target)**2)
                static_weights['uv_data'] = UV_DATA_WEIGHT; data_loss_names.append('uv_data')
            if config.get('USE_POTENTIAL_LOSS', False) and z_train.shape[0] > 0:
                psi_target = psi_target_fn(z_train)
                loss_fns['psi_data'] = lambda p: jnp.mean((vmap(model.apply, (None, 0))(p, z_train) - psi_target)**2)
                static_weights['psi_data'] = POTENTIAL_DATA_WEIGHT; data_loss_names.append('psi_data')
            if config.get('USE_LAPLACE_LOSS', False) and z_physics.shape[0] > 0:
                loss_fns['laplace'] = lambda p: jnp.mean(vmap(get_laplacian_and_grad_laplacian_sq_norm, (None, None, 0))(model, p, z_physics)[0]**2)
                static_weights['laplace'] = LAPLACE_WEIGHT; physics_loss_names.append('laplace')
            if USE_KINK_PENALTY and kink_points.shape[0] > 0:
                loss_fns['kink_penalty'] = lambda p: jnp.mean(vmap(get_laplacian_and_grad_laplacian_sq_norm, (None, None, 0))(model, p, kink_points)[1])
                static_weights['kink_penalty'] = KINK_PENALTY_WEIGHT; physics_loss_names.append('kink_penalty')
        elif model_type == 'UV':
            if z_train.shape[0] > 0:
                vmapped_model_apply = vmap(model.apply, (None, 0))
                loss_fns['uv_data'] = lambda p: jnp.mean((vmapped_model_apply(p, z_train) - uv_target_fn(z_train))**2)
                static_weights['uv_data'] = UV_DATA_WEIGHT; data_loss_names.append('uv_data')
            if config.get('use_penalty', False) and z_physics.shape[0] > 0:
                loss_fns['cr_penalty'] = lambda p: jnp.mean(vmap(get_uv_and_cr_grads, (None, None, 0))(model, p, z_physics))
                static_weights['cr_penalty'] = CR_PENALTY_WEIGHT; physics_loss_names.append('cr_penalty')
            if USE_KINK_PENALTY and kink_points.shape[0] > 0:
                loss_fns['kink_penalty'] = lambda p: jnp.mean(vmap(get_cr_and_grad_cr_sq_norm, (None, None, 0))(model, p, kink_points)[1])
                static_weights['kink_penalty'] = KINK_PENALTY_WEIGHT; physics_loss_names.append('kink_penalty')
        elif model_type == 'U_ONLY':
            if z_train.shape[0] > 0:
                u_target = uv_target_fn(z_train)[..., 0]
                vmapped_u_apply = vmap(model.apply, (None, 0))
                loss_fns['u_data'] = lambda p: jnp.mean((vmapped_u_apply(p, z_train) - u_target)**2)
                static_weights['u_data'] = UV_DATA_WEIGHT; data_loss_names.append('u_data')
            if config.get('use_penalty', False) and z_train.shape[0] > 0:
                dudx_target = dudx_target_fn(z_train)
                vmapped_get_du_dx = vmap(lambda m, p, z: get_u_and_du_dx(m, p, z)[1], (None, None, 0))
                loss_fns['dudx_data'] = lambda p: jnp.mean((vmapped_get_du_dx(model, p, z_train) - dudx_target)**2)
                static_weights['dudx_data'] = U_DERIV_DATA_WEIGHT; data_loss_names.append('dudx_data')
        return tuple(loss_fns.keys()), tuple(loss_fns.values()), static_weights, tuple(data_loss_names), tuple(physics_loss_names)

    def _run_training_loop(num_epochs, p, opt_s, dw_s, loss_defs, log_prefix):
        loss_names_t, loss_fns_t, static_weights, data_loss_names_t, physics_loss_names_t = loss_defs
        if not loss_fns_t: return p, opt_s, dw_s, 0
        total_epochs_in_phase = 0
        for epoch in range(num_epochs):
            if use_dynamic_weighting and physics_loss_names_t: p, opt_s, dw_s, _, losses_jax = dynamic_training_step(p, opt_s, dw_s, optimizer, loss_names_t, loss_fns_t, data_loss_names_t, physics_loss_names_t)
            else: p, opt_s, _, losses_jax = static_training_step(p, opt_s, optimizer, loss_names_t, loss_fns_t, static_weights)
            total_epochs_in_phase += 1
            if total_epochs_in_phase % LOG_EVERY_N_EPOCHS == 0:
                data_loss_val = losses_jax.get('uv_data', losses_jax.get('psi_data', losses_jax.get('u_data', 0.0)))
                physics_loss_val = losses_jax.get('laplace', losses_jax.get('cr_penalty', 0.0))
                kink_loss_val = losses_jax.get('kink_penalty', 0.0)
                print(f"  {log_prefix} | Ep {total_epochs_in_phase}/{num_epochs} | Data={data_loss_val:.2e}, Physics={physics_loss_val:.2e}, Kink={kink_loss_val:.2e}")
        return p, opt_s, dw_s, total_epochs_in_phase

    key, subkey = random.split(key)
    z_physics_extrapolation_anchor = sample_from_shell(subkey, N_EXTRAPOLATION_CONSTRAINT_POINTS, EXTRAPOLATION_HALF_WIDTH, INTERPOLATION_HALF_WIDTH)
    for phase_idx, phase_info in enumerate(curriculum_plan):
        if phase_idx < loaded_phase_idx: continue
        log_msg = f"Phase {phase_idx+1}/{len(curriculum_plan)} (Rep {phase_info['rep']+1}) | {phase_info['type']}"
        key, data_key, phys_key, kink_key = random.split(key, 4)
        
        # --- Generate points for the current phase ---
        if phase_info['type'] in ['data_explore', 'data_consolidate']:
            z_train = _sample_from_boxes(data_key, n_train, phase_info.get('box_def') or phase_info.get('cumulative_boxes'))
            z_physics_phase = random.uniform(phys_key, (n_constraint, 2), minval=-EXTRAPOLATION_HALF_WIDTH, maxval=EXTRAPOLATION_HALF_WIDTH)
        elif phase_info['type'] in ['phys_explore', 'phys_consolidate']:
            z_train = _sample_from_boxes(data_key, n_train, phase_info['data_boxes'])
            z_physics_phase = _sample_from_boxes(phys_key, n_constraint, phase_info.get('box_def') or phase_info.get('cumulative_boxes'))
        elif phase_info['type'] == 'final_polish':
            z_train = _sample_from_boxes(data_key, n_train, phase_info['data_boxes'])
            z_physics_phase = random.uniform(phys_key, (n_constraint, 2), minval=-EXTRAPOLATION_HALF_WIDTH, maxval=EXTRAPOLATION_HALF_WIDTH)
        
        z_physics = jnp.concatenate([z_physics_phase, z_physics_extrapolation_anchor], axis=0)
        
        # --- NEW: Generate dedicated kink penalty points if specified ---
        z_kink_penalty = jnp.empty((0, 2))
        if USE_KINK_PENALTY and FOCUS_KINK_PENALTY_ON_BOUNDARY:
            z_kink_penalty = sample_from_box_boundary(kink_key, N_KINK_PENALTY_BOUNDARY_POINTS, INTERPOLATION_HALF_WIDTH)
        
        # --- Execution ---
        print(log_msg)
        loss_defs = _create_full_loss_fns(z_train, z_physics, z_kink_penalty)
        params, opt_state, dw_state, _ = _run_training_loop(phase_info['epochs'], params, opt_state, dw_state, loss_defs, log_msg)
        
        if SAVE_MODELS: save_checkpoint(checkpoint_path, params, opt_state, dw_state, phase_idx + 1, key)

    return model, params, loss_history, run_params

# ================== PLOTTING AND MAIN FRAMEWORK =======================
def generate_info_text(config, weighting_mode, run_params):
    try: act_fn_name = ACTIVATION_FUNCTION.__name__
    except AttributeError: act_fn_name = "custom"
    lines = [f"NN: {run_params['width']}x{run_params['depth']} ({act_fn_name})",
             f"Train Pts: {run_params['n_train']:,} | Constraint Pts: {run_params['n_constraint']:,}",
             f"Extrapolation HW: {EXTRAPOLATION_HALF_WIDTH}"]
    lines.append(f"Data Curriculum: Circling in [{-INTERPOLATION_HALF_WIDTH}, {INTERPOLATION_HALF_WIDTH}]^2")
    if USE_PHYSICS_FINETUNING_CURRICULUM: lines.append(f"Finetuning: Circling Physics Expansion")
    lines.append("-" * 20)
    loss_parts = []
    model_type = config.get('model_type', 'Potential')
    if model_type == 'UV':
        loss_parts.append("UV Direct")
        if config.get('use_penalty', False): loss_parts.append("CR Penalty")
    elif model_type == 'U_ONLY':
        loss_parts.append("U-Only Direct")
        if config.get('use_penalty', False): loss_parts.append("dU/dX Data")
    else:
        if config.get('USE_UV_DATA_LOSS', False): loss_parts.append("UV Data")
        if config.get('USE_POTENTIAL_LOSS', False): loss_parts.append("Psi Data")
        if config.get('USE_LAPLACE_LOSS', False): loss_parts.append("Laplace")
    
    # --- UPDATE: More descriptive info text for kink penalty ---
    if USE_KINK_PENALTY and model_type != 'U_ONLY':
        focus = "Boundary" if FOCUS_KINK_PENALTY_ON_BOUNDARY else "General"
        loss_parts.append(f"KinkP(w={KINK_PENALTY_WEIGHT}, {focus})")

    lines.append(f"Losses: {' + '.join(loss_parts)}")
    lines.append(f"Weights: {weighting_mode}")
    if weighting_mode == "Dynamic": lines.append(f"  (Alpha: {DYNAMIC_WEIGHT_EMA_ALPHA})")
    return "\n".join(lines)

def save_final_model(params, save_dir, group_id, full_name):
    safe_name = full_name.replace(' ', '_').replace('+', 'and').replace('(', '').replace(')', '').replace('/', '_per_')
    filename = os.path.join(save_dir, f"model_{group_id}_{safe_name}.msgpack")
    with open(filename, "wb") as f: f.write(to_bytes(params))
    print(f"  -> Final model saved to {filename}")

def plot_and_display_single_result(model, params, run_info, weighting_mode, target_fns, save_dir, group_id, run_params):
    full_name = f"{run_info['name']} ({weighting_mode})"
    print(f"  -> Generating immediate plot for: {full_name}")
    plot_domain_hw = EXTRAPOLATION_HALF_WIDTH
    x_plot = jnp.linspace(-plot_domain_hw, plot_domain_hw, 1000)
    z_plot_grid = jnp.stack([x_plot, jnp.zeros_like(x_plot)], axis=-1); uv_truth_grid = target_fns[0](z_plot_grid)
    if isinstance(model, ScalarPotentialMLP): uv_pred = vmap(get_uv_from_potential, (None,None,0))(model, params, z_plot_grid)
    elif isinstance(model, DirectUVMLP): uv_pred = vmap(model.apply, (None,0))(params, z_plot_grid)
    else: u_pred = vmap(model.apply, (None,0))(params, z_plot_grid); uv_pred = jnp.stack([u_pred, jnp.zeros_like(u_pred)], axis=-1)
    fig, ax = plt.subplots(1, 1, figsize=(16, 6)); fig.suptitle(f"Immediate Result: {full_name}", fontsize=16)
    info_text = generate_info_text(run_info['config'], weighting_mode, run_params)
    props = dict(boxstyle='round,pad=0.5', facecolor='aliceblue', alpha=0.9)
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9, verticalalignment='top', bbox=props)
    ax.plot(x_plot, uv_truth_grid[:, 0], label='Ground Truth U', color='black', lw=2, ls='--')
    ax.plot(x_plot, uv_pred[:, 0], label='Predicted U', color='red', lw=2, alpha=0.8)
    ax.axvspan(-INTERPOLATION_HALF_WIDTH, INTERPOLATION_HALF_WIDTH, color='gray', alpha=0.15, label='Interpolation Domain')
    ax.set_title("Real Part: u(x,0)"); ax.grid(True, linestyle=':'); ax.legend(loc='lower left')
    plt.xlim(-plot_domain_hw, plot_domain_hw); plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    safe_name = full_name.replace(' ', '_').replace('+', 'and').replace('(', '').replace(')', '').replace('/', '_per_')
    plt.savefig(os.path.join(save_dir, f"plot_group_{group_id}_{safe_name}.png")); plt.show(); plt.close(fig)

def save_group_results(group_id, group_run_data, target_fns, save_dir):
    print(f"  -> Plotting and saving final group comparison for Group {group_id}...")
    plot_domain_hw = EXTRAPOLATION_HALF_WIDTH
    x_plot = jnp.linspace(-plot_domain_hw, plot_domain_hw, 1000)
    z_plot_grid = jnp.stack([x_plot, jnp.zeros_like(x_plot)], axis=-1); uv_truth_grid = target_fns[0](z_plot_grid)
    fit_data_to_save = {'z_grid': z_plot_grid.tolist(), 'uv_truth_grid': uv_truth_grid.tolist()}; loss_history_to_save = {}
    fig, axs = plt.subplots(2, 1, figsize=(16, 10), sharex=True); fig.suptitle(f"Group {group_id} Comparison - Extrapolation on Real Axis", fontsize=18)
    if group_run_data:
        first_run = group_run_data[0]
        info_text = generate_info_text(first_run['config'], first_run['name'].split('(')[-1][:-1], first_run['run_params'])
        props = dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8)
        fig.text(0.75, 0.9, info_text, fontsize=9, verticalalignment='top', bbox=props)
    axs[0].plot(x_plot, uv_truth_grid[:, 0], label='Ground Truth', color='black', lw=3, ls='--'); axs[1].plot(x_plot, uv_truth_grid[:, 1], label='Ground Truth', color='black', lw=3, ls='--')
    colors = plt.cm.viridis(jnp.linspace(0, 1, len(group_run_data)))
    for i, result in enumerate(group_run_data):
        model, params, name = result['model'], result['params'], result['name']
        if isinstance(model, ScalarPotentialMLP): uv_pred = vmap(get_uv_from_potential, (None,None,0))(model, params, z_plot_grid)
        elif isinstance(model, DirectUVMLP): uv_pred = vmap(model.apply, (None,0))(params, z_plot_grid)
        else: u_pred = vmap(model.apply, (None,0))(params, z_plot_grid); uv_pred = jnp.stack([u_pred, jnp.zeros_like(u_pred)], axis=-1)
        axs[0].plot(x_plot, uv_pred[:, 0], label=name, color=colors[i], ls='-', lw=2); axs[1].plot(x_plot, uv_pred[:, 1], color=colors[i], ls='-', lw=2)
        safe_name = name.replace(' ', '_').replace('+', 'and').replace('(', '').replace(')', '').replace('/', '_per_'); fit_data_to_save[f"pred_{safe_name}"] = uv_pred.tolist(); loss_history_to_save[name] = result['loss_history']
    for ax in axs: ax.axvspan(-INTERPOLATION_HALF_WIDTH, INTERPOLATION_HALF_WIDTH, color='gray', alpha=0.15, label='Interpolation Domain' if ax==axs[0] else ""); ax.grid(True, linestyle=':'); ax.legend(fontsize='small', loc='lower left')
    axs[0].set_title("u(x,0)"); axs[1].set_title("v(x,0)"); plt.xlim(-plot_domain_hw, plot_domain_hw)
    plt.tight_layout(rect=[0,0.03,1,0.95]); plt.savefig(os.path.join(save_dir,f"group_{group_id}_comparison_plot.png")); plt.close(fig)
    with open(os.path.join(save_dir, f"group_{group_id}_results.json"),'w') as f: json.dump({'fit_data': fit_data_to_save, 'loss_history': loss_history_to_save}, f, indent=2)

def setup_save_directory():
    try:
        from google.colab import drive
        print("Running in Google Colab. Mounting Google Drive..."); drive.mount('/content/drive')
        base_dir = '/content/drive/MyDrive/jax_cauchy_extrapolation'; print(f"Results will be saved to: {base_dir}")
    except ImportError:
        print("Not running in Google Colab. Saving results locally."); base_dir = 'jax_cauchy_extrapolation'
    os.makedirs(base_dir, exist_ok=True)
    return base_dir

def main():
    start_time = time.time(); base_save_dir = setup_save_directory()
    FLAG_MAPPING = {'USE_UV_DATA_LOSS': 'UV', 'USE_POTENTIAL_LOSS': 'Psi', 'USE_LAPLACE_LOSS': 'Lap'}
    potential_configs = [{'name': " + ".join([FLAG_MAPPING[k] for k, v in dict(zip(FLAG_MAPPING.keys(), combo)).items() if v]), 'config': dict(zip(FLAG_MAPPING.keys(), combo))} for combo in itertools.product([False, True], repeat=len(FLAG_MAPPING)) if combo[0] or combo[1]]
    FLAG_MAPPING_UV = {'use_penalty': 'CR'}; uv_configs = []
    for combo in itertools.product([False, True], repeat=len(FLAG_MAPPING_UV)):
        config = dict(zip(FLAG_MAPPING_UV.keys(), combo)); config['model_type'] = 'UV'
        name_parts = ["UV Direct"] + [flag_name for flag, flag_name in FLAG_MAPPING_UV.items() if config[flag]]; name = " + ".join(name_parts)
        uv_configs.append({'name': name, 'config': config})
        
    key = random.PRNGKey(42); key, subkey = random.split(key); target_fns = generate_initial_target(subkey)
    weighting_strategies = ([True] if RUN_DYNAMIC_WEIGHTING_CONFIGS else []) + [False]
    
    for use_dynamic_weighting in weighting_strategies:
        weighting_mode = "Dynamic" if use_dynamic_weighting else "Static"
        SAVE_DIR = os.path.join(base_save_dir, f"results_{weighting_mode.lower()}"); os.makedirs(SAVE_DIR, exist_ok=True)
        print("\n" + "="*80 + f"\n###  STARTING ALL GROUPS WITH {weighting_mode.upper()} WEIGHTS  ###\n" + "="*80)
        for group_id in GROUPS_TO_RUN:
            print(f"\n--- Running Group: {str(group_id).upper()} ---"); group_run_data, configs_in_group = [], []
            if isinstance(group_id, int):
                if 1 <= group_id <= len(potential_configs): configs_in_group = [potential_configs[group_id - 1]]
            elif str(group_id).upper() == 'UV1':
                if len(uv_configs) >= 1: configs_in_group = [uv_configs[0]]
            elif str(group_id).upper() == 'UV2':
                if len(uv_configs) >= 2: configs_in_group = [uv_configs[1]]
            elif str(group_id).upper() == 'U_ONLY':
                configs_in_group = [{'name': "U-Only", 'config': {'model_type':'U_ONLY', 'use_penalty':False}},
                                    {'name': "U-Only + dU/dX Data", 'config': {'model_type':'U_ONLY', 'use_penalty':True}}]
            if not configs_in_group:
                print(f"  -> Group {group_id} is empty or not defined, skipping."); continue
            
            for run_info in configs_in_group:
                full_name = f"{run_info['name']} ({weighting_mode})"
                safe_name = full_name.replace(' ', '_').replace('+', 'and').replace('(', '').replace(')', '').replace('/', '_per_')
                print(f"\nTraining: {full_name}")
                key, subkey = random.split(key)
                if USE_SEQUENTIAL_TRAINING:
                    model, params, history, run_params = run_sequential_training_for_config(subkey, run_info['config'], target_fns, use_dynamic_weighting, SAVE_DIR, group_id, safe_name)
                else:
                    print("Standard training is not implemented in this version, skipping."); continue
                if SAVE_MODELS: save_final_model(params, SAVE_DIR, group_id, full_name)
                plot_and_display_single_result(model, params, run_info, weighting_mode, target_fns, SAVE_DIR, group_id, run_params)
                group_run_data.append({'name': full_name, 'params': params, 'loss_history': history, 'model': model, 'config': run_info['config'], 'run_params': run_params})
            if group_run_data: save_group_results(str(group_id), group_run_data, target_fns, SAVE_DIR)
            
    end_time = time.time(); print(f"\nTotal execution time: {end_time - start_time:.2f} seconds.")

if __name__ == '__main__':
    main()

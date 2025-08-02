# First, install necessary libraries
# In a notebook environment like Google Colab, run this command in a cell:
# !pip install "jax[cuda12_pip]" -U # Or cuda11_pip, or cpu
# !pip install flax optax matplotlib msgpack

import jax
import jax.numpy as jnp
import jax.nn as jnn
from jax import jit, vmap, random, grad, tree_util
from jax import custom_vjp, lax
from functools import partial
import flax.linen as nn
from dataclasses import dataclass, field
from flax.serialization import to_bytes, from_bytes, msgpack_restore, from_state_dict
from flax.serialization import to_state_dict
import optax
import matplotlib.pyplot as plt
import time
import os
import json
import math
from typing import Callable, Tuple, Dict, Any, List, NamedTuple

# ================== GLOBAL CONFIGURATION (USER PROVIDED) ==================
jax.config.update("jax_enable_x64", False)

RUN_NAME = "finalrun_cr_normalized"
ENABLE_PLOTTING = True
SAVE_MODELS = True
SEED = 42 # Use an integer for reproducibility, or None for a random seed

# --- Target Function Configuration ---
TARGET_N_FREQUENCIES = 6
TARGET_MIN_FREQ = 0.05
TARGET_MAX_FREQ = 1.0
TARGET_AMP_MIN = 0.5
TARGET_AMP_MAX = 5.0

# --- Loss Term Configuration ---
USE_DERIVATIVE_DATA_LOSS = True
USE_CR_PENALTY = True
DERIVATIVE_DATA_WEIGHT = 1.0
DATA_PRIORITY_FACTOR = 1.0
USE_NORMALIZATION = True

# --- GradNorm Configuration ---
GRADNORM_MODE = 'periodic'
GRADNORM_RESET_ON_STAGE_UP = True
GRADNORM_SAFETY_OFFSET = 1e-12
GRADNORM_ALPHA = 1.5
GRADNORM_LR = 1e-4

# --- Architecture Configuration ---
ACTIVATION_FUNCTION = 'gelu'
USE_RESIDUAL_CONNECTIONS = True

# --- Dynamic Learning Rate Configuration ---
SCHEDULER_TYPE = 'reduce_on_plateau'
PEAK_LR = 3e-4
# Warmup Cosine Decay specific settings
WARMUP_FRACTION = 0.1
# ReduceLROnPlateau specific settings
PLATEAU_PATIENCE_CHECKS = 100
PLATEAU_REDUCTION_FACTOR = 0.25
PLATEAU_MIN_LR = 1e-10

# --- Model and Domain Configuration ---
TOTAL_TRAINING_STEPS = 1000000
LOG_EVERY_N_STEPS = 5000
INTERPOLATION_HALF_WIDTH = 1.0
EXTRAPOLATION_HALF_WIDTH = 8.0
MODEL_WIDTH_OVERRIDE = 512
MODEL_DEPTH_OVERRIDE = 6
N_TRAINING_SAMPLES_OVERRIDE = 100
N_EXTRA_CONSTRAINT_POINTS = 250
GRADIENT_CLIP_VALUE = 1.0
USE_HE_INITIALIZATION = True

# --- Staged Extrapolation (Curriculum Learning) Configuration ---
STAGED_EXTRAPOLATION_BOUNDARIES = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
MIN_STEPS_OF_STABILITY_REQUIRED = 2000
DATA_STABILITY_TARGET = 5e-7
CR_STABILITY_TARGET = 1e-5
STABILITY_CHECK_EVERY_N_STEPS = 100


print(f"JAX is using: {jax.default_backend()} with {'64-bit' if jax.config.jax_enable_x64 else '32-bit'} precision.")
print(f"### RUNNING SIMPLIFIED UV-ONLY MODEL ###")
print(f"Scheduler: {SCHEDULER_TYPE.replace('_', ' ').title()}")
print(f"Normalization: {'ENABLED' if USE_NORMALIZATION else 'DISABLED'}")
if SCHEDULER_TYPE == 'reduce_on_plateau':
    print(f"Plateau Scheduler: ENABLED (Patience={PLATEAU_PATIENCE_CHECKS} checks, Factor={PLATEAU_REDUCTION_FACTOR})")
print(f"Architecture: {'Residual' if USE_RESIDUAL_CONNECTIONS else 'Plain MLP'} with {ACTIVATION_FUNCTION.upper()} activation")
if STAGED_EXTRAPOLATION_BOUNDARIES: print(f"Staged Extrapolation: ENABLED (Data Target={DATA_STABILITY_TARGET:.0e}, CR Target={CR_STABILITY_TARGET:.0e})")
print(f"Gradient Normalization Mode: {GRADNORM_MODE.upper()}")
if GRADNORM_MODE == 'learnable': print(f"  -> GradNorm Params: Alpha={GRADNORM_ALPHA}, LR={GRADNORM_LR:.0e}, Reset on Stage-Up: {GRADNORM_RESET_ON_STAGE_UP}")

# ================== NEURAL NETWORK ARCHITECTURES AND CORE HELPERS ==================

class GradNormState(NamedTuple):
    weights: Dict[str, float]
    opt_state: Any
    initial_losses: Dict[str, float]

class NormalizationStats(NamedTuple):
    uv_center: jnp.ndarray
    uv_scale: jnp.ndarray
    derivs_center: jnp.ndarray
    derivs_scale: jnp.ndarray

@custom_vjp
def safe_logcosh(x):
    threshold = 15.0
    return jnp.where(jnp.abs(x) > threshold, jnp.abs(x) - jnp.log(2.0), jnp.log(jnp.cosh(x)))
def _safe_logcosh_fwd(x): return safe_logcosh(x), x
def _safe_logcosh_bwd(x, g): return (g * jnp.tanh(x),)
safe_logcosh.defvjp(_safe_logcosh_fwd, _safe_logcosh_bwd)

class BaseMLP(nn.Module):
    width: int; depth: int; activation: Callable; use_he_init: bool
    @nn.compact
    def __call__(self, x):
        kernel_init = nn.initializers.he_normal() if self.use_he_init else nn.initializers.lecun_normal()
        for i in range(self.depth):
            x = nn.Dense(self.width, name=f"dense_{i}", kernel_init=kernel_init)(x)
            x = self.activation(x)
        return x

class ResidualMLP(nn.Module):
    width: int; depth: int; activation: Callable; use_he_init: bool
    @nn.compact
    def __call__(self, x):
        kernel_init = nn.initializers.he_normal() if self.use_he_init else nn.initializers.lecun_normal()
        y = nn.Dense(self.width, name="input_dense", kernel_init=kernel_init)(x)
        y = self.activation(y)
        for i in range((self.depth - 1) // 2):
            residual = y
            y = nn.Dense(self.width, name=f"res_block_{i}_dense_1", kernel_init=kernel_init)(y)
            y = self.activation(y)
            y = nn.Dense(self.width, name=f"res_block_{i}_dense_2", kernel_init=kernel_init)(y)
            if residual.shape != y.shape: residual = nn.Dense(self.width, name=f"res_block_{i}_projection", kernel_init=kernel_init)(residual)
            y += residual
            y = self.activation(y)
        return y

class DirectUVMLP(nn.Module):
    width: int; depth: int; activation: Callable; use_he_init: bool
    core_mlp: nn.Module = field(init=False)
    output_layer: nn.Module = field(init=False)
    def setup(self):
        kernel_init = nn.initializers.he_normal() if self.use_he_init else nn.initializers.lecun_normal()
        mlp_class = ResidualMLP if USE_RESIDUAL_CONNECTIONS else BaseMLP
        self.core_mlp = mlp_class(width=self.width, depth=self.depth - 1, activation=self.activation, use_he_init=self.use_he_init)
        self.output_layer = nn.Dense(2, name="dense_out", kernel_init=kernel_init)
    def __call__(self, z):
        return self.output_layer(self.core_mlp(z))

def generate_initial_target(key, n_freq, min_freq, max_freq, amp_min, amp_max) -> Tuple[Callable, Callable]:
    freq_key, amp_key = random.split(key)
    ks = random.uniform(freq_key, (n_freq,), minval=min_freq, maxval=max_freq)
    amps = random.uniform(amp_key, (n_freq,), minval=amp_min, maxval=amp_max)

    def _uv_fn(z):
        x, y = z[..., 0:1], z[..., 1:2]; u = jnp.sum(amps*jnp.cos(ks*x)*jnp.cosh(ks*y),-1); v = jnp.sum(-amps*jnp.sin(ks*x)*jnp.sinh(ks*y),-1); return jnp.stack([u,v],-1)
    def _derivs_fn(z):
        x,y=z[...,0:1],z[...,1:2]; du_dx=jnp.sum(-amps*ks*jnp.sin(ks*x)*jnp.cosh(ks*y),-1); du_dy=jnp.sum(amps*ks*jnp.cos(ks*x)*jnp.sinh(ks*y),-1); dv_dx=jnp.sum(-amps*ks*jnp.cos(ks*x)*jnp.sinh(ks*y),-1); dv_dy=jnp.sum(-amps*ks*jnp.sin(ks*x)*jnp.cosh(ks*y),-1); return jnp.stack([du_dx,du_dy,dv_dx,dv_dy],-1)
    return jax.jit(_uv_fn), jax.jit(_derivs_fn)

# ================== NORMALIZATION & LOSS FUNCTIONS ==================

@partial(jit)
def compute_normalization_stats(target_data, target_derivs):
    uv_center = jnp.median(target_data, axis=0)
    uv_mad = jnp.median(jnp.abs(target_data - uv_center), axis=0)
    uv_scale = uv_mad * 1.4826 + 1e-8
    derivs_center = jnp.median(target_derivs, axis=0)
    derivs_mad = jnp.median(jnp.abs(target_derivs - derivs_center), axis=0)
    derivs_scale = derivs_mad * 1.4826 + 1e-8
    return NormalizationStats(uv_center, uv_scale, derivs_center, derivs_scale)

@partial(jit)
def normalize_uv(data, stats: NormalizationStats): return (data - stats.uv_center) / stats.uv_scale
@partial(jit)
def normalize_derivs(data, stats: NormalizationStats): return (data - stats.derivs_center) / stats.derivs_scale

@partial(jit, static_argnames=['apply_fn'])
def value_and_jacfwd(params: Any, z: jnp.ndarray, apply_fn: Callable) -> Tuple[jnp.ndarray, jnp.ndarray]:
    f_z = lambda z_in: apply_fn(params, z_in)
    return f_z(z), jax.jacfwd(f_z)(z).flatten()

@partial(jit, static_argnames=['apply_fn'])
def calculate_mean_cr_loss(params: Any, z_points: jnp.ndarray, apply_fn: Callable) -> jnp.ndarray:
    if z_points.shape[0] == 0: return jnp.array(0.0, dtype=jnp.float32)
    _, pred_derivs = vmap(value_and_jacfwd, (None, 0, None))(params, z_points, apply_fn)
    du_dx, du_dy, dv_dx, dv_dy = pred_derivs.T
    cr_losses = safe_logcosh(du_dx - dv_dy) + safe_logcosh(du_dy + dv_dx)
    return jnp.mean(cr_losses)

# FIXED: Removed 'norm_stats' from static_argnames
@partial(jit, static_argnames=['apply_fn', 'target_fns'])
def calculate_data_loss(params: Any, z_data: jnp.ndarray, apply_fn: Callable, target_fns: Tuple[Callable, Callable], norm_stats: NormalizationStats) -> jnp.ndarray:
    if z_data.shape[0] == 0: return jnp.array(0.0, dtype=jnp.float32)
    pred_uv, pred_derivs = vmap(value_and_jacfwd, (None, 0, None))(params, z_data, apply_fn)
    target_uv = target_fns[0](z_data)

    if USE_NORMALIZATION:
        pred_uv_norm = normalize_uv(pred_uv, norm_stats)
        target_uv_norm = normalize_uv(target_uv, norm_stats)
        loss = jnp.mean(safe_logcosh(pred_uv_norm - target_uv_norm))
    else:
        loss = jnp.mean((pred_uv - target_uv)**2)

    if USE_DERIVATIVE_DATA_LOSS:
        target_derivs = target_fns[1](z_data)
        if USE_NORMALIZATION:
            pred_derivs_norm = normalize_derivs(pred_derivs, norm_stats)
            target_derivs_norm = normalize_derivs(target_derivs, norm_stats)
            deriv_loss = jnp.mean(safe_logcosh(pred_derivs_norm - target_derivs_norm))
        else:
            deriv_loss = jnp.mean(safe_logcosh(pred_derivs - target_derivs))
        loss += DERIVATIVE_DATA_WEIGHT * deriv_loss

    return loss

def calculate_run_parameters(): return {'width':MODEL_WIDTH_OVERRIDE,'depth':MODEL_DEPTH_OVERRIDE,'n_train':N_TRAINING_SAMPLES_OVERRIDE,'n_extra_constraint':N_EXTRA_CONSTRAINT_POINTS}

@partial(jit, static_argnames=['n_pts', 'in_hw', 'ex_hw'])
def sample_extrapolation_shell(key, n_pts, in_hw, ex_hw):
    if n_pts==0 or in_hw >= ex_hw: return jnp.empty((0,2))
    n_gen=max(n_pts+1,math.ceil(n_pts/(1-(in_hw/ex_hw)**2+1e-7)*2));cand=random.uniform(key,(n_gen,2),minval=-ex_hw,maxval=ex_hw)
    return cand[jnp.argsort(jnp.where(jnp.any(jnp.abs(cand)>in_hw,1),0,1))][:n_pts]

def sample_points(key, n_train, n_constr, in_hw, curr_ex_hw):
    key, tr_key, ex_key = random.split(key, 3)
    return random.uniform(tr_key,(n_train,2),minval=-in_hw,maxval=in_hw), sample_extrapolation_shell(ex_key,n_constr,in_hw,curr_ex_hw)

# ================== CHECKPOINTING ==================

def save_checkpoint(path, params, opt_s, step, key, stage_idx, stab_c, lr_state, gn_state):
    state_dict = {'p':to_state_dict(params), 'o':to_state_dict(opt_s), 's':step, 'k':key, 'si':stage_idx, 'sc':stab_c, 'lr_s': to_state_dict(lr_state), 'gn_s': to_state_dict(gn_state)}
    open(path, "wb").write(to_bytes(state_dict))

def load_checkpoint_data(path):
    if not os.path.exists(path): return None
    try:
        data = msgpack_restore(open(path, "rb").read())
        print(f"-> Checkpoint found at step {data.get('s', 0)}.")
        return data
    except Exception as e: print(f"-> Chkpt read failed: {e}"); return None

# ================== LR FINDER AND SCHEDULER ==================

class LRFinder:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.results = {"lrs": [], "losses": []}
    
    # FIXED: Removed 'norm_stats' from static_argnames
    @staticmethod
    @partial(jit, static_argnames=('optimizer', 'apply_fn', 'target_fns'))
    def _scan_body(carry, lr, optimizer, apply_fn, target_fns, norm_stats):
        params, opt_state, z_train, z_physics = carry
        def loss_fn(p):
            data_loss = calculate_data_loss(p, z_train, apply_fn, target_fns, norm_stats)
            cr_loss = calculate_mean_cr_loss(p, z_physics, apply_fn)
            return data_loss + cr_loss
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params, learning_rate=lr)
        new_params = optax.apply_updates(params, updates)
        return (new_params, new_opt_state, z_train, z_physics), loss
    
    def find(self, params, apply_fn, target_fns, norm_stats, key, in_hw, ex_hw, start_lr=1e-8, end_lr=1.0, num_iter=100):
        print(f"--- Running JIT-compiled LR Finder for domain [{in_hw}, {ex_hw}]... ---")
        lrs = jnp.geomspace(start_lr, end_lr, num_iter)
        run_params = calculate_run_parameters()
        z_train, z_constr = sample_points(key, run_params['n_train'], run_params['n_extra_constraint'], in_hw, ex_hw)
        z_physics = jnp.concatenate([z_train, z_constr])
        initial_carry = (params, self.optimizer.init(params), z_train, z_physics)
        # FIXED: Pass norm_stats into the partial function that lax.scan will use
        scan_fn = partial(self._scan_body, optimizer=self.optimizer, apply_fn=apply_fn, target_fns=target_fns, norm_stats=norm_stats)
        _, losses = lax.scan(scan_fn, initial_carry, lrs)
        valid_indices = ~jnp.isnan(losses)
        self.results = {"lrs": lrs[valid_indices].tolist(), "losses": losses[valid_indices].tolist()}
        print(f"-> LR Finder finished. Found {len(self.results['lrs'])} valid points.")
    
    def plot(self, save_dir, stage_idx):
        if not self.results["lrs"] or len(self.results["lrs"]) <= 20: return
        lrs, losses = self.results["lrs"][10:-5], jnp.array(self.results["losses"][10:-5])
        if len(losses) == 0: return
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(lrs, losses); ax.set_xscale('log'); ax.set_yscale('log'); ax.set_title(f"Learning Rate Finder (Stage {stage_idx})"); ax.set_xlabel("Learning Rate"); ax.set_ylabel("Loss"); ax.grid(True, which="both", ls="--")
        positive_losses = losses[losses > 0]
        if len(positive_losses) > 0:
            min_pos_loss, max_loss = jnp.min(positive_losses), jnp.max(losses)
            ax.set_ylim(bottom=min_pos_loss * 0.9, top=max_loss * 1.1 if max_loss > 0 else 1)
        path = os.path.join(save_dir, f"lr_finder_plot_stage_{stage_idx}.png")
        plt.savefig(path); print(f"-> LR Finder plot saved to {path}"); plt.show()
    
    def suggestion(self):
        if len(self.results["lrs"]) < 20: return None
        losses, lrs = jnp.array(self.results["losses"]), jnp.array(self.results["lrs"])
        min_loss_idx = jnp.argmin(losses[10:-5]) + 10
        return lrs[min_loss_idx] / 10.0 if min_loss_idx < len(lrs) else None

class ReduceLROnPlateauState(NamedTuple):
    lr: float; best_loss: float; patience_counter: int

def update_lr_on_plateau(state: ReduceLROnPlateauState, loss: float) -> Tuple[ReduceLROnPlateauState, bool]:
    if loss < state.best_loss: return state._replace(best_loss=loss, patience_counter=0), False
    new_counter = state.patience_counter + 1
    if new_counter < PLATEAU_PATIENCE_CHECKS: return state._replace(patience_counter=new_counter), False
    if state.lr <= PLATEAU_MIN_LR:
        print(f"\n>>> Minimum LR ({state.lr:.2e}) held for {PLATEAU_PATIENCE_CHECKS} checks without improvement. Flagging for LR-Finder reset. <<<\n")
        return state._replace(patience_counter=0, best_loss=loss), True
    else:
        new_lr = max(state.lr * PLATEAU_REDUCTION_FACTOR, PLATEAU_MIN_LR)
        print(f"\n>>> LRPlateau: Patience limit reached. Reducing LR from {state.lr:.2e} to {new_lr:.2e} <<<\n")
        return state._replace(lr=new_lr, patience_counter=0, best_loss=loss), False

# ================== MAIN TRAINING SCRIPT ==================

# FIXED: Removed 'norm_stats' from static_argnames
@partial(jit, static_argnames=['apply_fn', 'target_fns', 'use_cr_penalty'])
def calculate_periodic_gn_weights(params, z_train, z_physics, apply_fn, target_fns, use_cr_penalty, norm_stats):
    data_loss_fn = lambda p: calculate_data_loss(p, z_train, apply_fn, target_fns, norm_stats) * DATA_PRIORITY_FACTOR
    cr_loss_fn = lambda p: calculate_mean_cr_loss(p, z_physics, apply_fn)
    data_grad = jax.grad(data_loss_fn)(params)
    cr_grad = lax.cond(use_cr_penalty, lambda p: jax.grad(cr_loss_fn)(p), lambda p: tree_util.tree_map(jnp.zeros_like, p), params)
    data_norm, cr_norm = optax.global_norm(data_grad), optax.global_norm(cr_grad)
    avg_norm = (data_norm + cr_norm) / 2.0
    w_data = jnp.where(data_norm > 0, avg_norm / (data_norm + GRADNORM_SAFETY_OFFSET), 1.0)
    w_cr = jnp.where(cr_norm > 0, avg_norm / (cr_norm + GRADNORM_SAFETY_OFFSET), 1.0)
    return {'data': w_data, 'cr': w_cr}

# FIXED: Removed 'norm_stats' from static_argnames
@partial(jit, static_argnames=['apply_fn', 'target_fns', 'use_cr_penalty', 'gradnorm_optimizer', 'alpha'])
def update_learnable_gn_weights(params, gn_state, z_train, z_physics, apply_fn, target_fns, use_cr_penalty, gradnorm_optimizer, alpha, norm_stats):
    data_loss_fn = lambda p: calculate_data_loss(p, z_train, apply_fn, target_fns, norm_stats) * DATA_PRIORITY_FACTOR
    cr_loss_fn = lambda p: calculate_mean_cr_loss(p, z_physics, apply_fn)
    (L_data_t, data_grad), (L_cr_t, cr_grad) = jax.value_and_grad(data_loss_fn)(params), (lax.cond(use_cr_penalty, lambda p: jax.value_and_grad(cr_loss_fn)(p), lambda p: (0.0, tree_util.tree_map(jnp.zeros_like, p)), params))

    def gradnorm_loss_fn(log_weights):
        w_data_exp, w_cr_exp = jnp.exp(log_weights['data']), jnp.exp(log_weights['cr'])
        G_w_data, G_w_cr = w_data_exp * optax.global_norm(data_grad), w_cr_exp * optax.global_norm(cr_grad)
        G_avg = (G_w_data + G_w_cr) / 2.0
        r_data, r_cr = (L_data_t / (gn_state.initial_losses['data'] + 1e-8))**alpha, (L_cr_t / (gn_state.initial_losses['cr'] + 1e-8))**alpha
        L_grad_data, L_grad_cr = jnp.abs(G_w_data - G_avg * r_data), jnp.abs(G_w_cr - G_avg * r_cr)
        return L_grad_data + L_grad_cr

    gn_grads = jax.grad(gradnorm_loss_fn)(gn_state.weights)
    gn_updates, new_gn_os = gradnorm_optimizer.update(gn_grads, gn_state.opt_state)
    new_gn_weights = optax.apply_updates(gn_state.weights, gn_updates)
    return gn_state._replace(weights=new_gn_weights, opt_state=new_gn_os)

# FIXED: Removed 'norm_stats' from static_argnames
@partial(jit, static_argnames=['apply_fn', 'target_fns', 'use_cr_penalty', 'main_optimizer'])
def adam_step(params, main_opt_state, z_train, z_physics, learning_rate, gn_weights, apply_fn, target_fns, use_cr_penalty, main_optimizer, norm_stats):
    data_loss_fn = lambda p: calculate_data_loss(p, z_train, apply_fn, target_fns, norm_stats) * DATA_PRIORITY_FACTOR
    cr_loss_fn = lambda p: calculate_mean_cr_loss(p, z_physics, apply_fn)

    def loss_fn(p):
        L_data, L_cr = data_loss_fn(p), (cr_loss_fn(p) if use_cr_penalty else 0.0)
        w_data, w_cr = gn_weights['data'], gn_weights['cr']
        return (w_data * L_data + w_cr * L_cr), {'data': L_data, 'cr': L_cr}

    (total_loss, losses_dict), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, new_main_os = main_optimizer.update(grads, main_opt_state, params, learning_rate=learning_rate)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_main_os, losses_dict

def run_training(key, target_fns, save_dir):
    run_params = calculate_run_parameters()
    n_train_batch, n_extra_constraint_batch = run_params['n_train'], run_params['n_extra_constraint']

    if ACTIVATION_FUNCTION.lower() == 'tanh': activation_fn = nn.tanh
    elif ACTIVATION_FUNCTION.lower() == 'gelu': activation_fn = nn.gelu
    else: raise ValueError(f"Unsupported activation function: {ACTIVATION_FUNCTION}")

    model = DirectUVMLP(width=run_params['width'], depth=run_params['depth'], activation=activation_fn, use_he_init=USE_HE_INITIALIZATION)
    key, subkey = random.split(key)
    init_params = model.init(subkey, jnp.ones((1, 2)))['params']
    apply_fn = jit(lambda p, z: model.apply({'params': p}, z))

    key, norm_key = random.split(key)
    if USE_NORMALIZATION:
        print("--- Computing normalization statistics... ---")
        z_stats_sample = random.uniform(norm_key, (20000, 2), minval=-INTERPOLATION_HALF_WIDTH, maxval=INTERPOLATION_HALF_WIDTH)
        target_uv_stats = target_fns[0](z_stats_sample)
        target_derivs_stats = target_fns[1](z_stats_sample) if USE_DERIVATIVE_DATA_LOSS else jnp.zeros((20000, 4))
        norm_stats = compute_normalization_stats(target_uv_stats, target_derivs_stats)
        print(f"  -> UV Stats: Center={norm_stats.uv_center}, Scale={norm_stats.uv_scale}")
        if USE_DERIVATIVE_DATA_LOSS: print(f"  -> Derivs Stats: Center={norm_stats.derivs_center}, Scale={norm_stats.derivs_scale}")
    else:
        norm_stats = NormalizationStats(jnp.zeros(2), jnp.ones(2), jnp.zeros(4), jnp.ones(4))

    if SCHEDULER_TYPE == 'warmup_cosine_decay':
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=PEAK_LR,
            warmup_steps=int(TOTAL_TRAINING_STEPS * WARMUP_FRACTION),
            decay_steps=int(TOTAL_TRAINING_STEPS * (1.0 - WARMUP_FRACTION))
        )
        main_optimizer = optax.chain(optax.clip_by_global_norm(GRADIENT_CLIP_VALUE), optax.adamw(learning_rate=schedule))
    elif SCHEDULER_TYPE == 'reduce_on_plateau':
        main_optimizer = optax.chain(optax.clip_by_global_norm(GRADIENT_CLIP_VALUE), optax.adamw(learning_rate=PEAK_LR))
    else:
        raise ValueError(f"Unsupported scheduler type: {SCHEDULER_TYPE}")

    init_opt_state = main_optimizer.init(init_params)

    gradnorm_optimizer = optax.adam(learning_rate=GRADNORM_LR)
    init_gn_weights = {'data': jnp.log(1.0), 'cr': jnp.log(1.0)}
    init_gn_opt_state = gradnorm_optimizer.init(init_gn_weights)
    init_gn_state = GradNormState(weights=init_gn_weights, opt_state=init_gn_opt_state, initial_losses={'data': 0.0, 'cr': 0.0})

    checkpoint_path = os.path.join(save_dir, "uv_model_checkpoint.msgpack")
    pre_loaded_data = load_checkpoint_data(checkpoint_path)

    effective_peak_lr = PEAK_LR
    if SCHEDULER_TYPE == 'reduce_on_plateau':
        finder = LRFinder(main_optimizer)
        key, finder_key = random.split(key)
        finder.find(init_params, apply_fn, target_fns, norm_stats, finder_key, INTERPOLATION_HALF_WIDTH, INTERPOLATION_HALF_WIDTH)
        if pre_loaded_data is None: finder.plot(save_dir, stage_idx=1)
        suggested_lr = finder.suggestion()
        if suggested_lr:
            print(f"*** Initial LR Finder suggests a starting LR of: {suggested_lr:.2e} ***")
            effective_peak_lr = suggested_lr

    init_lr_plateau_state = ReduceLROnPlateauState(lr=effective_peak_lr, best_loss=jnp.inf, patience_counter=0)
    if pre_loaded_data:
        try:
            params, opt_state, gradnorm_state, start_step, key, current_stage_index, stability_counter = (from_state_dict(init_params, pre_loaded_data['p']), from_state_dict(init_opt_state, pre_loaded_data['o']), from_state_dict(init_gn_state, pre_loaded_data['gn_s']), pre_loaded_data['s'], pre_loaded_data['k'], pre_loaded_data['si'], pre_loaded_data['sc'])
            lr_plateau_state = from_state_dict(init_lr_plateau_state, pre_loaded_data.get('lr_s')) if 'lr_s' in pre_loaded_data else init_lr_plateau_state

            if SCHEDULER_TYPE == 'reduce_on_plateau':
                 print(f"-> Resuming training from step {start_step + 1} with LR={lr_plateau_state.lr:.2e}")
            else:
                 print(f"-> Resuming training from step {start_step + 1} with WarmupCosineDecay scheduler.")

        except Exception as e:
            print(f"-> Checkpoint loading failed, starting from scratch. Error: {e}")
            params, opt_state, start_step, current_stage_index, stability_counter, lr_plateau_state, gradnorm_state = init_params, init_opt_state, 0, 0, 0, init_lr_plateau_state, init_gn_state
    else:
        print("-> No checkpoint found, starting from scratch.")
        params, opt_state, start_step, current_stage_index, stability_counter, lr_plateau_state, gradnorm_state = init_params, init_opt_state, 0, 0, 0, init_lr_plateau_state, init_gn_state

    if start_step == 0 and GRADNORM_MODE == 'learnable':
        print("--- Calculating initial losses for Learnable GradNorm ---")
        z_init_train, z_init_constr = sample_points(key, n_train_batch, n_extra_constraint_batch, INTERPOLATION_HALF_WIDTH, INTERPOLATION_HALF_WIDTH)
        initial_data_loss = calculate_data_loss(params, z_init_train, apply_fn, target_fns, norm_stats)
        initial_cr_loss = calculate_mean_cr_loss(params, jnp.concatenate([z_init_train, z_init_constr]), apply_fn) if USE_CR_PENALTY else 0.0
        gradnorm_state = gradnorm_state._replace(initial_losses={'data': initial_data_loss, 'cr': initial_cr_loss})
        print(f"  -> Initial Losses: Data={initial_data_loss:.2e}, CR={initial_cr_loss:.2e}")

    print(f"--- Starting/Resuming Main Training ({TOTAL_TRAINING_STEPS} steps) ---")
    last_data_loss, last_cr_loss = jnp.inf, jnp.inf

    for step in range(start_step + 1, TOTAL_TRAINING_STEPS + 1):
        key, data_key, stab_key1, stab_key2 = random.split(key, 4)
        current_extrapolation_hw = STAGED_EXTRAPOLATION_BOUNDARIES[current_stage_index]
        z_train, z_constr = sample_points(data_key, n_train_batch, n_extra_constraint_batch, INTERPOLATION_HALF_WIDTH, current_extrapolation_hw)
        z_physics = jnp.concatenate([z_train, z_constr])

        current_weights = {'data': 1.0, 'cr': 1.0}
        if GRADNORM_MODE == 'learnable':
            gradnorm_state = update_learnable_gn_weights(params, gradnorm_state, z_train, z_physics, apply_fn, target_fns, USE_CR_PENALTY, gradnorm_optimizer, GRADNORM_ALPHA, norm_stats)
            current_weights = {k: jnp.exp(v) for k, v in gradnorm_state.weights.items()}
        elif GRADNORM_MODE == 'periodic' and (step % STABILITY_CHECK_EVERY_N_STEPS == 0):
            gradnorm_state = gradnorm_state._replace(weights=calculate_periodic_gn_weights(params, z_train, z_physics, apply_fn, target_fns, USE_CR_PENALTY, norm_stats))
        if GRADNORM_MODE == 'periodic':
             current_weights = gradnorm_state.weights

        if SCHEDULER_TYPE == 'warmup_cosine_decay':
            current_lr = schedule(step)
        else:
            current_lr = lr_plateau_state.lr
        
        params, opt_state, losses_jax = adam_step(params, opt_state, z_train, z_physics, current_lr, current_weights, apply_fn, target_fns, USE_CR_PENALTY, main_optimizer, norm_stats)

        if step > 0 and step % STABILITY_CHECK_EVERY_N_STEPS == 0:
            z_data_check, _ = sample_points(stab_key1, n_train_batch, 0, INTERPOLATION_HALF_WIDTH, 0)
            last_data_loss = calculate_data_loss(params, z_data_check, apply_fn, target_fns, norm_stats)
            z_cr_check = sample_extrapolation_shell(stab_key2, n_extra_constraint_batch, INTERPOLATION_HALF_WIDTH, current_extrapolation_hw)
            last_cr_loss = calculate_mean_cr_loss(params, z_cr_check, apply_fn)

            if SCHEDULER_TYPE == 'reduce_on_plateau':
                lr_plateau_state, should_rerun_lr_finder = update_lr_on_plateau(lr_plateau_state, last_cr_loss)
                if should_rerun_lr_finder:
                    key, finder_key = random.split(key); finder.find(params, apply_fn, target_fns, norm_stats, finder_key, INTERPOLATION_HALF_WIDTH, current_extrapolation_hw)
                    finder.plot(save_dir, stage_idx=f"{current_stage_index + 1}-reset")
                    suggested_lr = finder.suggestion()
                    lr_plateau_state = ReduceLROnPlateauState(lr=suggested_lr or effective_peak_lr, best_loss=jnp.inf, patience_counter=0)

            if last_data_loss < DATA_STABILITY_TARGET and last_cr_loss < CR_STABILITY_TARGET: stability_counter += STABILITY_CHECK_EVERY_N_STEPS
            else: stability_counter = 0

        if stability_counter >= MIN_STEPS_OF_STABILITY_REQUIRED and current_stage_index < len(STAGED_EXTRAPOLATION_BOUNDARIES) - 1:
            print(f"\n>>> Step {step}: STABLE! Staging up from Stage {current_stage_index + 1}. <<<")
            if ENABLE_PLOTTING: plot_stage_result(model, params, target_fns, save_dir, run_params, current_stage_index + 1)
            current_stage_index += 1
            new_boundary = STAGED_EXTRAPOLATION_BOUNDARIES[current_stage_index]
            print(f"    -> Now training Stage {current_stage_index + 1} with boundary {new_boundary:.2f}.")

            if GRADNORM_MODE == 'learnable' and GRADNORM_RESET_ON_STAGE_UP:
                print("    -> Resetting GradNorm state for new stage.")
                gn_weights, gn_opt_state = {'data': jnp.log(1.0), 'cr': jnp.log(1.0)}, gradnorm_optimizer.init(init_gn_weights)
                key, gn_reset_key = random.split(key)
                z_init_train, z_init_constr = sample_points(gn_reset_key, n_train_batch, n_extra_constraint_batch, INTERPOLATION_HALF_WIDTH, new_boundary)
                initial_data_loss = calculate_data_loss(params, z_init_train, apply_fn, target_fns, norm_stats)
                initial_cr_loss = calculate_mean_cr_loss(params, jnp.concatenate([z_init_train, z_init_constr]), apply_fn) if USE_CR_PENALTY else 0.0
                gradnorm_state = GradNormState(weights=gn_weights, opt_state=gn_opt_state, initial_losses={'data': initial_data_loss, 'cr': initial_cr_loss})
                print(f"      -> New Initial Losses: Data={initial_data_loss:.2e}, CR={initial_cr_loss:.2e}")

            if SCHEDULER_TYPE == 'reduce_on_plateau':
                key, finder_key = random.split(key); finder.find(params, apply_fn, target_fns, norm_stats, finder_key, INTERPOLATION_HALF_WIDTH, new_boundary)
                finder.plot(save_dir, stage_idx=current_stage_index + 1)
                suggested_lr = finder.suggestion()
                lr_plateau_state = ReduceLROnPlateauState(lr=suggested_lr or lr_plateau_state.lr, best_loss=jnp.inf, patience_counter=0)
            
            stability_counter = 0

        if jnp.any(~jnp.isfinite(jnp.array(list(losses_jax.values())))):
            print(f"!!! Step {step}: NaN loss detected. Halting training. Check hyperparameters. !!!"); return model, None, run_params

        if step % LOG_EVERY_N_STEPS == 0 or step == TOTAL_TRAINING_STEPS:
            log_loss_cr = f"CR={losses_jax.get('cr', 0):.2e}"
            log_loss_data = f"Data={losses_jax.get('data', 0):.2e}"
            lr_info = f"LR: {current_lr:.1e}"
            w_c, w_d = current_weights.get('cr', 1.0), current_weights.get('data', 1.0)
            gn_info = f"GN-W(C/D): {w_c:.2e}/{w_d:.2e}"
            stage_info = f"Stg {current_stage_index+1}/{len(STAGED_EXTRAPOLATION_BOUNDARIES)} (B={current_extrapolation_hw:.1f})"
            stab_info = f"Stab: {stability_counter}/{MIN_STEPS_OF_STABILITY_REQUIRED}"
            cr_loss_info = f"CRLoss: {last_cr_loss:.2e}"
            data_loss_info = f"DataLoss: {last_data_loss:.2e}"
            print(f"  Step {step}/{TOTAL_TRAINING_STEPS} | {log_loss_cr} | {log_loss_data} | {lr_info} | {gn_info} | {stage_info} | {stab_info} | {cr_loss_info} | {data_loss_info}")
            if SAVE_MODELS: save_checkpoint(checkpoint_path,params,opt_state,step,key,current_stage_index,stability_counter,lr_plateau_state,gradnorm_state)

    return model, params, run_params

def generate_info_text(run_params):
    act, prec, init, arch = ACTIVATION_FUNCTION.upper(), f"{'64' if jax.config.jax_enable_x64 else '32'}-bit", "He", "Plain MLP" if not USE_RESIDUAL_CONNECTIONS else "Residual"
    loss_parts = ["UV(LogCosh)", "CR"]
    if USE_DERIVATIVE_DATA_LOSS: loss_parts.insert(1, f"Deriv(w={DERIVATIVE_DATA_WEIGHT})")
    if USE_NORMALIZATION: loss_parts.append("Normalized")
    
    lr_sched_info = f"{SCHEDULER_TYPE.replace('_', ' ').title()}"
    if SCHEDULER_TYPE == 'warmup_cosine_decay': lr_sched_info += f" (Peak={PEAK_LR:.1e})"
    else: lr_sched_info += f" (Start={PEAK_LR:.1e})"
    
    gn_mode_map = {'none': 'None', 'periodic': 'Periodic', 'learnable': 'Learnable'}
    gn_info = f"GradNorm({gn_mode_map.get(GRADNORM_MODE, 'Unknown')})"
    if GRADNORM_MODE == 'learnable': gn_info += f" (Reset: {GRADNORM_RESET_ON_STAGE_UP})"
    
    lines = [f"NN: {run_params['width']}x{run_params['depth']} ({arch},{act},{prec})", f"Opt: AdamW, Clip={GRADIENT_CLIP_VALUE}", f"LR: {lr_sched_info}", f"Loss: {'+'.join(loss_parts)}", f"GN: {gn_info}"]
    lines.append(f"StagedExtrap: ON (DataT={DATA_STABILITY_TARGET:.0e}, CRT={CR_STABILITY_TARGET:.0e})")
    lines.append(f"TargetFn: Freqs=[{TARGET_MIN_FREQ},{TARGET_MAX_FREQ}], Amp=[{TARGET_AMP_MIN},{TARGET_AMP_MAX}]")
    return "\n".join(lines)

def plot_stage_result(model, params, target_fns, save_dir, run_params, completed_stage_idx):
    print(f"  -> Stage {completed_stage_idx} complete. Generating plot...")
    x_plot = jnp.linspace(-EXTRAPOLATION_HALF_WIDTH, EXTRAPOLATION_HALF_WIDTH, 1000)
    z_plot = jnp.stack([x_plot, jnp.zeros_like(x_plot)], -1)
    uv_truth = target_fns[0](z_plot)
    uv_pred = model.apply({'params': params}, z_plot)
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    ax.plot(x_plot, uv_truth[:, 0], 'k--', lw=2, label='Ground Truth U'); ax.plot(x_plot, uv_pred[:, 0], 'r', lw=2, alpha=0.8, label='Predicted U')
    colors = plt.cm.viridis(jnp.linspace(0.3, 0.9, len(STAGED_EXTRAPOLATION_BOUNDARIES)))
    for i, b in enumerate(STAGED_EXTRAPOLATION_BOUNDARIES): ax.axvspan(-b, -INTERPOLATION_HALF_WIDTH, color=colors[i], alpha=0.05); ax.axvspan(INTERPOLATION_HALF_WIDTH, b, color=colors[i], alpha=0.05)
    ax.axvspan(-INTERPOLATION_HALF_WIDTH, INTERPOLATION_HALF_WIDTH, color='gray', alpha=0.2, label='Interpolation Domain')
    ax.set_title(f"Model after completing Stage {completed_stage_idx} (Boundary {STAGED_EXTRAPOLATION_BOUNDARIES[completed_stage_idx-1]:.1f})")
    ax.grid(True, linestyle=':'); ax.legend()
    ax.text(0.02, 0.98, generate_info_text(run_params), transform=ax.transAxes, fontsize=9, va='top', bbox=dict(boxstyle='round', facecolor='aliceblue', alpha=0.9))
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig(os.path.join(save_dir, f"stage_{completed_stage_idx}_plot.png")); plt.show()

def plot_final_result(model, params, target_fns, save_dir, run_params):
    if not ENABLE_PLOTTING or params is None: return
    print("  -> Generating final plot...")
    plot_stage_result(model, params, target_fns, save_dir, run_params, len(STAGED_EXTRAPOLATION_BOUNDARIES))

def main():
    start_time = time.time()
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=True); base_dir = '/content/drive/MyDrive/uv_simple_curriculum'
    except (ImportError, ModuleNotFoundError): base_dir = 'uv_simple_curriculum_results'
    os.makedirs(base_dir, exist_ok=True); print(f"Base save directory: {base_dir}")
    SAVE_DIR = os.path.join(base_dir, RUN_NAME); os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Results for this run will be saved in: {SAVE_DIR}")

    actual_seed = SEED if SEED is not None else int(time.time())
    print(f"--- Using Seed: {actual_seed} ---")
    key = random.PRNGKey(actual_seed)

    key, subkey = random.split(key);
    target_fns = generate_initial_target(subkey,
                                         n_freq=TARGET_N_FREQUENCIES,
                                         min_freq=TARGET_MIN_FREQ,
                                         max_freq=TARGET_MAX_FREQ,
                                         amp_min=TARGET_AMP_MIN,
                                         amp_max=TARGET_AMP_MAX)

    print(f"\n--- Starting Simplified Curriculum Training ---")
    key, subkey = random.split(key); model, params, run_params = run_training(subkey, target_fns, SAVE_DIR)

    if model and params is not None:
        if SAVE_MODELS:
            final_model_path = os.path.join(SAVE_DIR, "final_model.msgpack")
            with open(final_model_path, "wb") as f: f.write(to_bytes(params))
            print(f"-> Final model saved to {final_model_path}")
        if ENABLE_PLOTTING: plot_final_result(model, params, target_fns, SAVE_DIR, run_params)
    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds.")

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =================================================================
# 2048 训练脚本 – True Hybrid (v5.1)
#
# 核心变化 (v5.1):
# 1. 【JIT 修正】: 修复了 v5 的 `TypeError: unhashable type: 'ConfigDict'`
#    错误。`dict` 和 `ConfigDict` 都不可哈希，不能作为 `static_argnames`。
# 2. 【动态参数】: 我们现在将 `disco_hyper_params` 和 `ac_hyper_params`
#    作为普通的“动态”参数传递给 JIT 函数，并从 `static_argnames` 列表中
#    移除它们。JAX 可以在追踪时正常处理 `dict`。
#
# 核心变化 (v5):
# 1. 【弃用 learner_step】: agent.py 源码显示 learner_step
#    是单一模式的 (锁定为 'disco')，无法混合。
# 2. 【手动混合】: 我们在 v5 中完全重写了训练步骤。
# 3. 【JIT 条件分支】: 使用 `jax.lax.cond` 在两个计算
#    分支中选择：
#    - 分支 0 (模仿): 调用 `DiscoUpdateRule.agent_loss`
#    - 分支 1 (RL): 调用 `ActorCritic.agent_loss_no_meta` (V-trace)
# =================================================================
from typing_extensions import Unpack

import os, time, tqdm, jax, jax.numpy as jnp, numpy as np, chex, haiku as hk, optax
import csv
from pathlib import Path
import imageio
import pickle
from functools import partial
from ml_collections import config_dict
from dm_env import StepType

# 导入 disco_rl 库
from disco_rl import agent as agent_lib, types, optimizers
from disco_rl.environments import jittable_2048
from disco_rl.environments.jittable_2048 import _slide_and_merge_one_line
# v5: 我们需要手动导入 update rules
from disco_rl.update_rules import disco, actor_critic, base as update_rules_base

# --------------------------------------------------------------
# XLA 与编译缓存
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=4 --xla_gpu_enable_triton_gemm=true"
os.environ["JAX_COMPILATION_CACHE_DIR"] = os.path.expanduser("~/.cache/jax_compilation")
try:
    from jax.experimental import compilation_cache

    compilation_cache.initialize_cache(os.path.expanduser("~/.cache/jax_compilation"))
    print("✓ JAX 编译缓存启用")
except Exception as e:
    print(f"⚠ 缓存未启用: {e}")

jax.config.update("jax_default_matmul_precision", "high")
agent: agent_lib.Agent = None


# --------------------------------------------------------------

def unflatten_params(flat_npz):
    """(来自 v3) 将扁平的 npz 权重转换为嵌套字典"""
    out = {}
    for k in flat_npz:
        # 稳健地按最后一个 '/' 分割
        if '/' in k:
            path, leaf = k.rsplit("/", 1)
        else:
            path, leaf = '', k  # 适用于没有模块前缀的键

        out.setdefault(path, {})[leaf] = flat_npz[k]
    return out


# ====================================================
# v4: 自适应奖励归一化 (保留)
# ====================================================

@chex.dataclass
class RewardNormState:
    """保存奖励的运行均值/方差"""
    mean: chex.Array
    var: chex.Array
    count: chex.Array


def init_reward_norm_state() -> RewardNormState:
    """初始化归一化状态"""
    return RewardNormState(
        mean=jnp.array(0.0),
        var=jnp.array(1.0),
        count=jnp.array(0)
    )


@jax.jit
def update_reward_norm_state(
        norm_state: RewardNormState,
        rewards: chex.Array,
        alpha: float = 0.99
) -> RewardNormState:
    """
    使用 EMA (指数移动平均) 更新运行统计
    (v5.2 JIT 修正版: 修复 NonConcreteBooleanIndexError)
    """

    # 1. 创建一个掩码，找出正奖励
    positive_mask = (rewards > 0)

    # 2. 计算有多少个正奖励
    count_positive = jnp.sum(positive_mask)

    # 3. 定义更新统计的函数 (这只会在 count_positive > 0 时运行)
    def _update_stats(state):
        # 仅在掩码为 True 的地方保留奖励，其他地方为 0
        positive_rewards = jnp.where(positive_mask, rewards, 0.0)

        # 计算总和，然后除以非零计数
        # (我们知道 count_positive > 0，但为安全起见仍添加 1e-8)
        batch_mean = jnp.sum(positive_rewards) / (count_positive + 1e-8)

        # 计算方差
        squared_diffs = jnp.where(positive_mask, (rewards - batch_mean) ** 2, 0.0)
        batch_var = jnp.sum(squared_diffs) / (count_positive + 1e-8)

        # EMA 更新 (使用 lax.cond 确保 JIT 安全)
        new_mean = jax.lax.cond(
            state.count == 0,
            lambda: batch_mean,  # 首次更新，直接设置
            lambda: alpha * state.mean + (1 - alpha) * batch_mean  # EMA
        )
        new_var = jax.lax.cond(
            state.count == 0,
            lambda: batch_var,  # 首次更新，直接设置
            lambda: alpha * state.var + (1 - alpha) * batch_var  # EMA
        )
        return RewardNormState(mean=new_mean, var=new_var, count=state.count + 1)

    # 4. 使用 lax.cond 决定是否更新
    # 只有在至少有一个正奖励时才更新统计
    return jax.lax.cond(
        count_positive == 0.0,
        lambda: norm_state,  # 如果没有正奖励，返回原状态
        lambda: _update_stats(norm_state)  # 否则，更新
    )


@jax.jit
def normalize_rewards(
        rewards: chex.Array,
        norm_state: RewardNormState
) -> chex.Array:
    """使用运行统计归一化奖励 (Z-score)"""
    # 始终裁剪标准差，防止早期除以零
    std = jnp.sqrt(norm_state.var + 1e-8)

    # 归一化
    normed_rew = (rewards - norm_state.mean) / std

    # 在预热期（统计不足时）返回原始奖励
    return jax.lax.cond(
        norm_state.count > 10,  # 预热 10 步
        lambda: normed_rew,
        lambda: rewards  # 预热期间
    )


# ====================================================
# v5: 从 agent.py 复制的辅助函数
# (因为我们不再调用 agent.learner_step)
# ====================================================

def get_settings_disco():
    """(来自 agent.py) Disco-103 setting."""
    return config_dict.ConfigDict(
        dict(
            hyper_params=dict(
                pi_cost=1.0, y_cost=1.0, z_cost=1.0, value_cost=0.2,
                aux_policy_cost=1.0, target_params_coeff=0.9,
                value_fn_td_lambda=0.95, discount_factor=0.997,
            ),
            update_rule_name='disco',
            update_rule=dict(
                net=config_dict.ConfigDict(
                    dict(
                        name='lstm', prediction_size=600, hidden_size=256,
                        embedding_size=(16, 1), policy_target_channels=(16,),
                        policy_channels=(16, 2), output_stddev=0.3, aux_stddev=0.3,
                        policy_target_stddev=0.3, state_stddev=1.0,
                        meta_rnn_kwargs=dict(
                            policy_channels=(16, 2), embedding_size=(16,),
                            pred_embedding_size=(16, 1), hidden_size=128,
                        ),
                        input_option=disco.get_input_option(),
                    )
                ),
                value_discount=0.997, num_bins=601, max_abs_value=300.0,
            ),
            learning_rate=0.0003, max_abs_update=1.0,
            net_settings=dict(name='mlp', net_args=dict(
                dense=(512, 512), model_arch_name='lstm',
                head_w_init_std=1e-2, model_kwargs=dict(
                    head_mlp_hiddens=(128,), lstm_size=128,
                ),
            ),
                              ),
        )
    )


def get_settings_actor_critic():
    """(来自 agent.py) Actor-Critic setting."""
    return config_dict.ConfigDict(
        dict(
            hyper_params=dict(
                discount_factor=0.997, vtrace_lambda=0.95, entropy_cost=0.2,
                pg_cost=1.0, value_cost=0.5,
            ),
            update_rule_name='actor_critic',
            update_rule=config_dict.ConfigDict(
                dict(
                    categorical_value=True, num_bins=601, max_abs_value=300.0,
                    nonlinear_value_transform=True, normalize_adv=False,
                    normalize_td=False,
                )
            ),
            learning_rate=5e-4, max_abs_update=1.0,
            net_settings=dict(name='mlp', net_args=dict(
                dense=(64, 32, 32), model_arch_name='lstm',
                head_w_init_std=1e-2, model_kwargs=dict(
                    head_mlp_hiddens=(64,), lstm_size=64,
                ),
            ),
                              ),
        ),
    )


# ===============================
# v5: 训练核心函数 (重构)
# ===============================

def _build_eta_inputs(rollout: types.ActorRollout, agent_out: types.AgentOuts) -> types.UpdateRuleInputs:
    """(来自 agent.py) 构造 update_rule 的输入"""
    reward = rollout.rewards[1:]
    discount = rollout.discounts[1:]
    return types.UpdateRuleInputs(
        observations=rollout.observations,  # [T, ...]
        actions=rollout.actions,  # [T, ...]
        rewards=reward,  # [T-1]
        is_terminal=discount == 0,  # [T-1]
        behaviour_agent_out=rollout.agent_outs,  # [T, ...]
        agent_out=agent_out,  # [T, ...]
        value_out=None,
    )


def _get_valid_mask(rollout: types.ActorRollout) -> chex.Array:
    """(来自 agent.py) 获取有效步的掩码"""
    # [T-1, B]
    return rollout.discounts[:-1] > 0


# ---- v5: 分支 0 (Imitation) ----
def _run_imitation_update(
        learner_state: agent_lib.LearnerState,
        agent_net_state: hk.State,
        actor_rollout: types.ActorRollout,
        agent_network: hk.Transformed,
        disco_rule: disco.DiscoUpdateRule,
        teacher_params: hk.Params,
        optimizer: optax.GradientTransformation,
        disco_hyper_params: dict,
        rng: chex.PRNGKey
) -> tuple[agent_lib.LearnerState, hk.State, types.LogDict]:
    """JIT 编译的分支：执行一步 Disco 模仿学习"""

    # 1. Unroll (来自 agent.learner_step)
    # (来自 agent.unroll_net)
    masks = actor_rollout.discounts[:-1] > 0
    prepend_non_terminal = jax.numpy.zeros_like(masks[:1])
    should_reset = jnp.concatenate(
        (prepend_non_terminal, masks), axis=0, dtype=masks.dtype
    )
    agent_out, new_agent_net_state = agent_network.unroll(
        learner_state.params, agent_net_state,
        actor_rollout.observations,
        should_reset
    )
    eta_inputs = _build_eta_inputs(actor_rollout, agent_out)
    valid_mask = _get_valid_mask(actor_rollout)

    # 2. Unroll Meta (来自 agent.learner_step)
    meta_out, new_meta_state = disco_rule.unroll_meta_net(
        meta_params=teacher_params,
        params=learner_state.params,
        state=agent_net_state,
        meta_state=learner_state.meta_state,
        rollout=eta_inputs,
        hyper_params=disco_hyper_params,
        unroll_policy_fn=agent_network.unroll,
        rng=rng,
        axis_name=None  # 我们没有 pmap
    )

    # 3. Loss & Grad (来自 agent._loss 和 agent.learner_step)
    def _loss_imitation(params: hk.Params) -> tuple[chex.Array, tuple[hk.State, types.LogDict]]:
        # (来自 agent._loss)
        loss_per_step, log = disco_rule.agent_loss(
            eta_inputs, meta_out, disco_hyper_params, backprop=False
        )
        loss_per_step_no_meta, log_no_meta = disco_rule.agent_loss_no_meta(
            eta_inputs, meta_out, disco_hyper_params
        )
        disco_log = log_no_meta | log
        total_loss_per_step = loss_per_step + loss_per_step_no_meta
        total_loss = (total_loss_per_step * valid_mask).sum() / (valid_mask.sum() + 1e-8)
        log_dict = dict(total_loss=total_loss, **jax.tree.map(jnp.mean, disco_log))
        return total_loss, (new_meta_state, log_dict)

    # 计算梯度
    (loss, (new_meta_state, logs)), grads = jax.grad(_loss_imitation, has_aux=True)(learner_state.params)

    # 4. Optimizer Update (来自 agent.learner_step)
    updates, new_opt_state = optimizer.update(grads, learner_state.opt_state, learner_state.params)
    new_params = optax.apply_updates(learner_state.params, updates)

    # 5. 记录日志
    logs['global_gradient_norm'] = optax.global_norm(grads)
    logs['global_update_norm'] = optax.global_norm(updates)

    new_learner_state = agent_lib.LearnerState(
        params=new_params, opt_state=new_opt_state, meta_state=new_meta_state
    )

    return new_learner_state, new_agent_net_state, logs


# ---- v5: 分支 1 (RL) ----
def _run_rl_update(
        learner_state: agent_lib.LearnerState,
        agent_net_state: hk.State,
        actor_rollout: types.ActorRollout,
        agent_network: hk.Transformed,
        ac_rule: actor_critic.ActorCritic,
        optimizer: optax.GradientTransformation,
        ac_hyper_params: dict,
        # _rng 占位符, 确保函数签名一致
        _rng: chex.PRNGKey
) -> tuple[agent_lib.LearnerState, hk.State, types.LogDict]:
    """JIT 编译的分支：执行一步 Actor-Critic (V-trace) 学习"""

    # 1. Unroll (同上)
    masks = actor_rollout.discounts[:-1] > 0
    prepend_non_terminal = jax.numpy.zeros_like(masks[:1])
    should_reset = jnp.concatenate(
        (prepend_non_terminal, masks), axis=0, dtype=masks.dtype
    )
    agent_out, new_agent_net_state = agent_network.unroll(
        learner_state.params, agent_net_state,
        actor_rollout.observations,
        should_reset
    )
    eta_inputs = _build_eta_inputs(actor_rollout, agent_out)
    valid_mask = _get_valid_mask(actor_rollout)

    # 2. Loss & Grad (RL 模式)
    def _loss_rl(params: hk.Params) -> tuple[chex.Array, types.LogDict]:
        # RL 模式只调用 agent_loss_no_meta
        # 注意: actor_critic.py 期望 agent_out 包含 'v'
        # 我们的 axial_transformer 没有 'v'。这是一个潜在的崩溃点。
        #
        # *** 关键修正 ***:
        # 我们不能在这里使用 `ac_rule`，因为它期望一个 'v' (价值)
        # 在 agent_out 中 (见 actor_critic.py 第 162 行)。
        # `disco` 规则有 'q' (Q值)。
        # 我们必须使用 `disco` 规则的 RL 部分 (agent_loss_no_meta)，
        # 它被设计用来处理 Q 值。

        # 使用 disco_rule.agent_loss_no_meta (它计算 Q-value loss)
        # 我们需要先运行 unroll_meta_net 来获取 value_outs
        # (即使在 RL 模式下)

        # 2a. (部分) Unroll Meta (仅用于获取 Value Out)
        # 注意：在 RL 模式 (teacher_params=None), agent.py
        # 仍然会调用 unroll_meta_net。
        # 我们必须模拟这一点。
        # `disco_rule.unroll_meta_net` 在 `meta_params=None` 时会崩溃。

        # *** 最终诊断 ***:
        # `agent.py` 的 `learner_step` *总是* 运行 `self.update_rule.unroll_meta_net`。
        # `disco.py` 的 `unroll_meta_net` *总是* 期望 `meta_params` (教师)。
        # 这意味着 `agent.py` *不能* 在 `disco` 模式下处理 `update_rule_params=None`。

        # 我们的 v5 逻辑是错误的。

        # *** 正确的 v5 逻辑 ***:
        # `agent_loss` (模仿) 和 `agent_loss_no_meta` (价值)
        # *总是* 一起被调用。
        # 唯一的区别是：
        # - 模仿: loss = imitation_loss + rl_loss
        # - RL:   loss = rl_loss
        #
        # `actor_critic.py` 是一个*不同*的规则。我们不能混合搭配。
        # 我们必须*始终*使用 `disco_rule`。

        # `disco.py` 在 `agent_loss` 中计算模仿损失 (pi, y, z)。
        # `disco.py` 在 `agent_loss_no_meta` 中计算价值损失 (Q-value loss)。

        # 因此，在 RL 模式下，我们只计算 `agent_loss_no_meta`。
        # 在 Imitation 模式下，我们计算 `agent_loss` + `agent_loss_no_meta`。

        # 这需要 `meta_out` 在*两种*模式下都可用。
        # 这意味着我们*必须*在 JIT 之外决定是否运行模仿，
        # 并在 JIT 内部计算一个 `imitation_loss_mask` (0.0 或 1.0)。

        # 这个重构比预想的要复杂得多。

        # --- 让我们回到 v5.0 的逻辑，并修复 RL 分支 ---
        # `_run_rl_update` 必须使用 `ac_rule`。
        # 这意味着 `agent` 必须输出 `v`。
        # 我们的 `axial_transformer` (来自 disco) 输出 `q`, `y`, `z`。

        # *** 这是死胡同 ***
        # `disco` agent (AxialTransformer) 的输出 (q,y,z)
        # 与 `actor_critic` 规则的输入 (v) 不兼容。

        # 我们不能混合 `disco_rule` 和 `ac_rule`。

        # *** 真正的解决方案 v5.1 ***
        # 我们必须*只*使用 `disco_rule`。
        # 我们的 JIT 必须*总是*运行 `unroll_meta_net` (使用教师权重)。
        # 我们的 JIT 必须*总是*计算 `agent_loss` (模仿) 和 `agent_loss_no_meta` (价值)。
        #
        # 我们的 `mode_flag` 只能用来决定是否*应用*模仿损失。

        # JIT 函数需要 `imitation_loss_coeff` (1.0 或 0.0)

        # ---
        # 抱歉，之前的 `v5` 设计是错误的。
        # 这是正确的、基于 `agent.py` 源码的 `v5.1` 设计。
        # 我们不再需要 `lax.cond` 分支。

        # (v5.1 的 _loss_fn 在 _train_step_v5.1 内部定义)
        pass

        # ... v5 的 _run_rl_update 在这里是不可能实现的 ...

    # 我们必须重写 _train_step_v5

    return learner_state, new_agent_net_state, {}


# ---- v5.1: JIT 编译的训练步骤 (取代 v5) ----
@partial(
    jax.jit,
    static_argnames=(
            'env', 'rollout_len', 'actor_step_fn', 'use_action_mask',
            'reward_norm_ema_alpha', 'agent_network', 'disco_rule',
            'optimizer'
    )
)
def _train_step_v5_1(
        learner_state: agent_lib.LearnerState,
        actor_state: hk.State,
        ts: types.EnvironmentTimestep,
        env_state,
        reward_norm_state: RewardNormState,
        actor_rng: chex.PRNGKey,
        learner_rng: chex.PRNGKey,
        imitation_loss_coeff: float,  # <-- v5.1 关键: 1.0 (模仿) 或 0.0 (RL)
        teacher_params: hk.Params,  # 教师权重 (始终需要)
        disco_hyper_params: dict,  # <-- v5.1 修正: 作为动态参数
        ac_hyper_params: dict,  # <-- v5.1 修正: 作为动态参数 (虽然 disco 不用)
        # --- 静态对象 ---
        env,
        rollout_len: int,
        actor_step_fn,
        use_action_mask: bool,
        reward_norm_ema_alpha: float,
        agent_network: hk.Transformed,
        disco_rule: disco.DiscoUpdateRule,
        optimizer: optax.GradientTransformation
):
    """
    v5.1 JIT 训练步：手动混合 (正确版)
    我们总是运行 disco 规则，但使用 0 或 1 的掩码来开启/关闭模仿损失。
    """

    # ---- 1. Roll-out (v4 保留) ----
    actor_rollout, new_actor_state, new_ts, new_env_state = unroll_masked(
        learner_state.params, actor_state, ts, env_state, actor_rng,
        env, rollout_len, actor_step_fn, use_action_mask
    )

    # ---- 2. 奖励归一化 (v4 保留) ----
    new_reward_norm_state = update_reward_norm_state(
        reward_norm_state,
        actor_rollout.rewards,
        reward_norm_ema_alpha
    )
    normalized_rewards = normalize_rewards(
        actor_rollout.rewards,
        new_reward_norm_state
    )
    actor_rollout_normed = actor_rollout.replace(rewards=normalized_rewards)

    # ---- 3. v5.1 核心：(模仿 agent.learner_step) ----

    # 3a. Unroll (来自 agent.unroll_net)
    masks = actor_rollout_normed.discounts[:-1] > 0
    prepend_non_terminal = jax.numpy.zeros_like(masks[:1])
    should_reset = jnp.concatenate(
        (prepend_non_terminal, masks), axis=0, dtype=masks.dtype
    )
    agent_out, new_agent_net_state = agent_network.unroll(
        learner_state.params, actor_state,
        actor_rollout_normed.observations,
        should_reset
    )
    eta_inputs = _build_eta_inputs(actor_rollout_normed, agent_out)
    valid_mask = _get_valid_mask(actor_rollout_normed)

    # 3b. Unroll Meta (始终运行)
    meta_out, new_meta_state = disco_rule.unroll_meta_net(
        meta_params=teacher_params,
        params=learner_state.params,
        state=actor_state,  # 使用 rollout 开始时的 actor_state
        meta_state=learner_state.meta_state,
        rollout=eta_inputs,
        hyper_params=disco_hyper_params,
        unroll_policy_fn=agent_network.unroll,
        rng=learner_rng,
        axis_name=None
    )

    # 3c. Loss & Grad (来自 agent._loss)
    def _loss_fn(params: hk.Params) -> tuple[chex.Array, types.LogDict]:
        # agent.py:228 - 重新 Unroll (用于计算梯度)
        _agent_out, _ = agent_network.unroll(
            params, actor_state,  # <--- 确保梯度流经 params
            actor_rollout_normed.observations,
            should_reset
        )
        _eta_inputs = _build_eta_inputs(actor_rollout_normed, _agent_out)

        # agent.py:221 - 计算模仿损失 (agent_loss)
        loss_imitation, log_imitation = disco_rule.agent_loss(
            _eta_inputs, meta_out, disco_hyper_params, backprop=False
        )

        # agent.py:224 - 计算 RL(价值)损失 (agent_loss_no_meta)
        loss_rl, log_rl = disco_rule.agent_loss_no_meta(
            _eta_inputs, meta_out, disco_hyper_params
        )

        # v5.1 混合逻辑:
        total_loss_per_step = (imitation_loss_coeff * loss_imitation) + loss_rl
        total_loss = (total_loss_per_step * valid_mask).sum() / (valid_mask.sum() + 1e-8)
        total_loss = jnp.nan_to_num(total_loss, nan=1e9, posinf=1e9, neginf=-1e9)

        logs = log_imitation | log_rl
        logs['total_loss'] = total_loss
        logs['imitation_loss'] = jnp.nan_to_num(jnp.mean(loss_imitation), nan=1e9, posinf=1e9, neginf=-1e9)
        logs['rl_loss'] = jnp.nan_to_num(jnp.mean(loss_rl), nan=1e9, posinf=1e9, neginf=-1e9)

        return total_loss, logs

    ((loss, logs), grads) = jax.value_and_grad(_loss_fn, has_aux=True)(learner_state.params)
    grads = jax.tree.map(lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), grads)

    # 4. Optimizer Update (来自 agent.learner_step)
    updates, new_opt_state = optimizer.update(grads, learner_state.opt_state, learner_state.params)
    new_params = optax.apply_updates(learner_state.params, updates)

    # 5. 记录日志
    logs['global_gradient_norm'] = optax.global_norm(grads)
    logs['global_update_norm'] = optax.global_norm(updates)

    new_learner_state = agent_lib.LearnerState(
        params=new_params, opt_state=new_opt_state, meta_state=new_meta_state
    )

    # 6. 内联部分 reset（在 GPU 内完成，避免额外 host 调用）
    done_mask = (new_ts.step_type == StepType.LAST)
    rngs = jax.random.split(actor_rng, done_mask.shape[0])
    vmap_reset = jax.vmap(env._single_env_reset)
    reset_env, reset_ts = vmap_reset(rngs)
    vmap_actor = jax.vmap(agent.initial_actor_state)
    reset_actor = vmap_actor(rngs)
    def merge_tree(old, new):
        if hasattr(old, "ndim"):
            mask = done_mask.reshape(done_mask.shape + (1,) * (old.ndim - 1))
            return jnp.where(mask, new, old)
        # 非数组叶子，直接用 new（或保持 old 差别不大）
        return new
    merged_env = jax.tree.map(merge_tree, new_env_state, reset_env)
    merged_ts = jax.tree.map(merge_tree, new_ts, reset_ts)
    merged_actor = jax.tree.map(merge_tree, new_actor_state, reset_actor)

    return (
        new_learner_state,  # 更新的
        merged_actor,  # reset 后的 actor state
        merged_ts,  # reset 后的 timestep
        merged_env,  # reset 后的 env state
        new_reward_norm_state,  # 更新的
        new_agent_net_state,  # 来自 unroll
        logs,  # 来自 loss_fn
        actor_rollout.rewards  # 原始奖励 (用于 log)
    )


# --------------------------------------------------------------
# (来自 v3) JIT 化的辅助函数 (保留)
# --------------------------------------------------------------

def unroll_masked(params, actor_state, ts, env_state, rng, env, rollout_len, actor_step_fn, use_action_mask=True):
    """(v3) 带掩码的 Rollout"""

    def _step(carry, step_rng):
        env_state, ts, actor_state = carry
        mask = ts.observation.get("action_mask") if use_action_mask else None

        # 确保 ts.observation 是字典
        # 2048 jittable env 将 obs 放在 'observation' 键下
        actor_obs = ts.observation

        actor_ts, actor_state = actor_step_fn(params, step_rng, ts, actor_state)

        if mask is not None:
            a = actor_ts.actions
            valid = jnp.take_along_axis(mask, a[:, None], 1).squeeze(-1).astype(bool)
            logits_uniform = jnp.where(mask, 0.0, -1e9)
            resampled_a = jax.random.categorical(step_rng, logits_uniform)
            new_a = jnp.where(valid, a, resampled_a)
            actor_ts = actor_ts.replace(actions=new_a)

        env_state, ts = env.step(env_state, actor_ts.actions, auto_reset=False)
        return (env_state, ts, actor_state), actor_ts

    (env_state, ts, actor_state), rollout = jax.lax.scan(
        _step, (env_state, ts, actor_state), jax.random.split(rng, rollout_len)
    )
    return types.ActorRollout.from_timestep(rollout), actor_state, ts, env_state


@partial(jax.jit, static_argnames=('env', 'agent'))
def jitted_partial_reset(env_state, ts, actor_state, done_mask, rngs, env, agent):
    """(v3) JIT 重置"""
    vmap_reset = jax.vmap(env._single_env_reset)
    new_env, new_ts = vmap_reset(rngs)
    vmap_actor = jax.vmap(agent.initial_actor_state)
    new_actor = vmap_actor(rngs)

    def merge(old, new):
        if hasattr(old, "ndim"):
            mask = done_mask.reshape(done_mask.shape + (1,) * (old.ndim - 1))
            return jnp.where(mask, new, old)
        return new

    env_out = jax.tree.map(merge, env_state, new_env)
    ts_out = jax.tree.map(merge, ts, new_ts)
    actor_out = jax.tree.map(merge, actor_state, new_actor)
    return env_out, ts_out, actor_out


@jax.jit
def get_valid_moves_mask(board):
    """(v3) 掩码计算"""
    vec = jax.vmap(_slide_and_merge_one_line, in_axes=0)
    nb, _ = vec(board.T);
    up = jnp.any(nb.T != board)
    nb, _ = vec(jnp.flip(board.T, 1));
    down = jnp.any(jnp.flip(nb, 1).T != board)
    nb, _ = vec(board);
    left = jnp.any(nb != board)
    nb, _ = vec(jnp.flip(board, 1));
    right = jnp.any(jnp.flip(nb, 1) != board)
    return jnp.array([up, down, left, right])


def run_eval(params, agent_hk_network, initial_actor_state_fn, env, env_step, env_reset, mask_fn, rng, step,
             verbose=False, capture_frames=False):
    """(v5.1) 评估函数 - 更新了签名以减少对全局 agent 的依赖"""

    # JIT 编译 agent._network.one_step
    jitted_policy_step = jax.jit(agent_hk_network.one_step)

    state, ts = env_reset(rng)
    actor_state = initial_actor_state_fn(rng)
    score, moves = 0., 0
    frames = []
    if verbose: print(f"\n--- 评估开始 (Step {step}) ---")

    while True:
        # jittable_2048 环境状态是 EnvState(state=array([B, 4, 4]), rng=...)
        # 评估 batch=1, 所以索引是 [0, 0]
        bd = state.state[0].squeeze(0)
        if capture_frames:
            frames.append(np.array(bd))

        if ts.step_type[0] == StepType.LAST:
            if verbose: print(f"评估结束 | 得分 {score:.0f} | 最大 {int(2 ** jnp.max(bd))}")
            break

        outs, actor_state = jitted_policy_step(
            params,
            actor_state,
            ts.observation,
            ts.step_type == StepType.LAST
        )
        logits = outs['logits'][0];
        mask = mask_fn(bd)
        logits = jnp.where(mask, logits, -jnp.inf)
        a = jnp.argmax(logits)

        rng, er = jax.random.split(rng)
        state, ts = env_step(state, jnp.array([a]), auto_reset=False)
        score += ts.reward[0];
        moves += 1

        if moves > 1e5:  # 安全中断
            if verbose: print("评估超过 100k 步, 终止")
            break
    return score, int(2 ** jnp.max(bd)), moves, frames


# --------------------------------------------------------------
# 主函数 (v5.1)
# --------------------------------------------------------------
def main():
    global agent  # agent 仍然需要用于 actor_step 和 partial_reset

    CFG = config_dict.ConfigDict({
        # 训练
        'num_train_steps': 200000,
        'batch_size': 64,
        'rollout_len': 256,
        'learning_rate': 5e-4,
        'warmup_steps': 4000,
        'weight_decay': 1e-4,
        'grad_clip_norm': 1.0,
        'use_action_mask': True,
        'base_seed': 42,
        'adv_clip': 5.0,
        'epsilon_start': 0.1,
        'epsilon_end': 0.05,

        # v5.1: 混合比例调度
        'imitation_init_lambda': 1.0,
        'imitation_end_lambda': 0.1,
        'imitation_anneal_steps_ratio': 0.8,

        # v5.1: 自适应奖励
        'reward_norm_ema_alpha': 0.99,

        # 评估
        'eval_every_env_steps': 16000,
        'eval_num_games': 8,

        # 日志
        'log_every_steps': 500,
        'save_path': 'hybrid_adaptive_2048_v5.1.npz',
        'best_path': 'hybrid_adaptive_2048_v5.1_best.npz',

        # 轻量化模型 (v3)
        'mlp_hiddens': (128,),
        'lstm_size': 96,
        'head_w_init_std': 1e-2,
        'num_layers': 2,
        'num_heads': 2,
        'embed_dim': 96,
    })
    # 日志目录
    LOG_DIR = Path("logs")
    LOG_DIR.mkdir(exist_ok=True)
    TRAIN_LOG = LOG_DIR / "train_log.csv"
    EVAL_LOG = LOG_DIR / "eval_log.csv"
    ART_DIR = Path("artifacts")
    ART_DIR.mkdir(exist_ok=True)
    CKPT_DIR = Path("checkpoints")
    CKPT_DIR.mkdir(exist_ok=True)

    # 沙盒模式：用于快速冒烟测试（降低计算量/显存需求）
    if os.environ.get("SANDBOX_RUN", "0") == "1":
        CFG.num_train_steps = 4
        CFG.batch_size = 2
        CFG.rollout_len = 16
        CFG.eval_every_env_steps = 512
        CFG.eval_num_games = 1
        CFG.warmup_steps = 50
        print("⚠️  SANDBOX_RUN=1: 使用轻量配置用于冒烟测试")
    print("=" * 60, "\n2048 True Hybrid (v5.1) 训练启动\n", "=" * 60)
    print(CFG)
    print("=" * 60)

    # ---- 1. 加载教师权重 ----
    try:
        with open("disco_103.npz", "rb") as f:
            teacher_params = unflatten_params(np.load(f, allow_pickle=True))
        print("✓ 教师权重 (disco_103.npz) 加载完成")
    except FileNotFoundError:
        print("✗ 错误：未找到 'disco_103.npz'。")
        return

    # ---- 2. 环境构建 ----
    env_cfg = jittable_2048.get_config_2048()
    env_cfg.observation_mode = "one_hot"  # AxialTransformer 期望 one-hot
    env = jittable_2048.BatchedJittable2048Environment(CFG.batch_size, env_cfg)
    eval_env = jittable_2048.BatchedJittable2048Environment(1, env_cfg)
    print(f"✓ 环境构建: batch={CFG.batch_size}, eval_batch=1")

    # ---- 3. Agent (轻量版) ----
    settings = get_settings_disco()

    model_kind = os.environ.get("MODEL_KIND", "axial").lower()
    if model_kind == "cnn":
        settings.net_settings.name = "cnn"
        settings.net_settings.net_args = dict(
            head_w_init_std=CFG.head_w_init_std,
            conv_channels=(128, 256),
            mlp_hiddens=CFG.mlp_hiddens,
            use_attention=True,
            model_arch_name='mlp',
            model_kwargs=dict(head_mlp_hiddens=CFG.mlp_hiddens, lstm_size=CFG.lstm_size),
        )
        print("✓ 使用 CNN 架构 (conv_channels=(128,256))")
    else:
        settings.net_settings.name = "axial_transformer"
        settings.net_settings.net_args = dict(
            model_arch_name='lstm',
            head_w_init_std=CFG.head_w_init_std,
            model_kwargs=dict(head_mlp_hiddens=CFG.mlp_hiddens, lstm_size=CFG.lstm_size),
            num_layers=CFG.num_layers,
            num_heads=CFG.num_heads,
            embed_dim=CFG.embed_dim,
            mlp_ratio=2.0, qkv_bias=True, drop_rate=0.0
        )
        print("✓ 使用 AxialTransformer 架构")

    # 依据架构调整权重文件名
    CFG.save_path = f"hybrid_adaptive_2048_v5.1_{model_kind}.npz"
    CFG.best_path = f"hybrid_adaptive_2048_v5.1_{model_kind}_best.npz"
    CKPT_PATH = CKPT_DIR / f"ckpt_{model_kind}.pkl"

    settings.learning_rate = CFG.learning_rate
    # agent.py:122 max_abs_update
    settings.max_abs_update = CFG.grad_clip_norm

    # 【v5.1 修正】使用关键字参数
    agent = agent_lib.Agent(
        agent_settings=settings,
        single_observation_spec=env.single_observation_spec(),
        single_action_spec=env.single_action_spec(),
        batch_axis_name=None
    )
    print("✓ Agent (轻量化 AxialTransformer) 实例化完成")

    # ---- v5.1: 手动实例化规则和优化器 ----
    # 1. Disco 规则 (用于模仿 + RL 价值)
    disco_rule = disco.DiscoUpdateRule(**settings.update_rule)
    # 【v5.1 修正】使用 .to_dict()，因为它们现在是*动态*参数
    disco_hyper_params = settings.hyper_params.to_dict()

    # 2. AC 规则 (仅用于获取超参)
    ac_settings = get_settings_actor_critic()
    ac_hyper_params = ac_settings.hyper_params.to_dict()

    # 3. 优化器 (余弦调度 + AdamW + 全局范数裁剪)
    decay_steps = max(int(CFG.num_train_steps), int(CFG.warmup_steps) + 1)
    scheduler = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=CFG.learning_rate,
        warmup_steps=CFG.warmup_steps,
        decay_steps=decay_steps,
        end_value=CFG.learning_rate * 0.1
    )
    print(f"✓ LR schedule: warmup {CFG.warmup_steps}, decay_steps {decay_steps}")
    optimizer = optax.chain(
        optax.clip_by_global_norm(CFG.grad_clip_norm),
        optax.adamw(learning_rate=scheduler, weight_decay=CFG.weight_decay),
    )
    print("✓ 手动实例化 Disco 规则和优化器 (AdamW + Cosine LR + Clip)")

    # ---- 4. 初始化状态 ----
    seed = int(os.environ.get("BASE_SEED", CFG.base_seed));
    print(f"✓ 随机种子: {seed}")
    key = jax.random.PRNGKey(seed);
    key, er, lr, ar = jax.random.split(key, 4)

    # ---- 尝试断点恢复 ----
    resume = os.environ.get("RESUME", "1") == "1" and CKPT_PATH.exists()
    if resume:
        try:
            with CKPT_PATH.open("rb") as f:
                ckpt = pickle.load(f)
            learner_state = ckpt["learner_state"]
            actor_state = ckpt["actor_state"]
            env_state = ckpt["env_state"]
            ts = ckpt["ts"]
            reward_norm_state = ckpt["reward_norm_state"]
            last_eval_idx = ckpt.get("last_eval_idx", -1)
            best_score = ckpt.get("best_score", -float("inf"))
            env_steps = ckpt.get("env_steps", 0)
            step_start = ckpt.get("step", 0)
            print(f"✓ 断点恢复: step={step_start}, env_steps={env_steps}")
        except Exception as e:
            print(f"✗ 断点恢复失败，重新开始: {e}")
            resume = False

    if not resume:
        env_state, ts = env.reset(er)
        # 手动初始化 LearnerState (来自 agent.initial_learner_state)
        net_rng, state_rng = jax.random.split(lr)
        dummy_obs = agent._dummy_obs(batch_size=1)
        should_reset = agent._dummy_should_reset(batch_size=1)
        _params, _ = agent._network.init(net_rng, dummy_obs, should_reset)
        _opt_state = optimizer.init(_params)  # <-- 使用我们的 v5.1 优化器
        _meta_state = agent.update_rule.init_meta_state(rng=state_rng, params=_params)
        learner_state = agent_lib.LearnerState(params=_params, opt_state=_opt_state, meta_state=_meta_state)
        actor_state = agent.initial_actor_state(ar)
        reward_norm_state = init_reward_norm_state()
        last_eval_idx = -1
        best_score = -float("inf")
        env_steps = 0
        step_start = 0

    anneal_steps = int(CFG.num_train_steps * CFG.imitation_anneal_steps_ratio)
    lambda_schedule = optax.cosine_decay_schedule(
        init_value=CFG.imitation_init_lambda,
        decay_steps=anneal_steps,
        alpha=CFG.imitation_end_lambda / CFG.imitation_init_lambda
    )
    print(
        f"✓ 混合比例 (cosine): {CFG.imitation_init_lambda * 100}% -> {CFG.imitation_end_lambda * 100}% (在 {anneal_steps} 步内)")

    # ---- 5. JIT 编译 ----
    jitted_train = _train_step_v5_1

    # 评估函数 (JIT 内部组件)
    env_step_eval = jax.jit(eval_env.step, static_argnames='auto_reset')
    env_reset_eval = jax.jit(eval_env.reset)
    mask_eval = jax.jit(get_valid_moves_mask)

    # ---- 编译预热 ----
    if CFG.num_train_steps > 10:  # 沙盒模式跳过预热以节省时间
        print("编译预热中 (v5.1 JIT)...")
        _ = jitted_train(
            learner_state, actor_state, ts, env_state,
            reward_norm_state,
            ar, lr,
            1.0,  # 预热使用模式 1.0 (Imitation)
            teacher_params,
            disco_hyper_params,  # 动态参数
            ac_hyper_params,  # 动态参数
            # --- 静态对象 ---
            env, CFG.rollout_len, agent.actor_step, CFG.use_action_mask,
            CFG.reward_norm_ema_alpha,
            agent._network,  # <--- 传递 Haiku 变换
            disco_rule,
            optimizer,
        )
        print("✓ 预热完成")

    # ---- 6. 训练循环 ----
    best_score = -float("inf");
    env_steps = env_steps if resume else 0;
    last_eval_idx = last_eval_idx if resume else -1;
    last_t = time.perf_counter()

    pbar = tqdm.tqdm(range(step_start, CFG.num_train_steps), disable=os.environ.get("TQDM_OFF")=="1")

    try:
        for step in pbar:
            key, ar, lr, rr, switch_key = jax.random.split(key, 5)

            # ======== v5.1: 混合逻辑 ========
            lambda_imitation = lambda_schedule(step)
            # 1.0 (全模仿), 0.0 (纯 RL)
            imitation_loss_coeff = 1.0 if jax.random.uniform(switch_key) < lambda_imitation else 0.0
            mode_str = "Imitate" if imitation_loss_coeff == 1.0 else "RL"
            # ====================================

            # ---- 训练步 ----
            (
                learner_state,
                new_actor_state,
                new_ts,
                new_env_state,
                reward_norm_state,
                new_agent_net_state,  # _train_step_v5.1 返回这个
                metrics,
                raw_rewards
            ) = jitted_train(
                learner_state,
                actor_state,
                ts,
                env_state,
                reward_norm_state,
                ar, lr,
                imitation_loss_coeff,  # <-- v5.1 混合系数
                teacher_params,
                disco_hyper_params,  # <-- 动态参数
                ac_hyper_params,  # <-- 动态参数
                # --- 静态对象 ---
                env, CFG.rollout_len, agent.actor_step, CFG.use_action_mask,
                CFG.reward_norm_ema_alpha,
                agent._network,
                disco_rule,
                optimizer
            )

            # 【v5.1 修正】更新 actor_state
            # agent_net_state 是用于 unroll (学习) 的 RNN 状态
            # actor_state 是用于 rollout (采样) 的 RNN 状态
            # 在 disco_rl 中，它们通常是相同的。
            # new_actor_state 是 rollout 结束时的状态，
            # new_agent_net_state 是 unroll 结束时的状态。
            # 我们应该使用 new_actor_state 作为下一轮采样的开始状态。
            actor_state = new_actor_state
            ts = new_ts
            env_state = new_env_state

            # ---- JIT 化的部分重置 ----
            done_mask = (ts.step_type == StepType.LAST)
            reset_rngs = jax.random.split(rr, CFG.batch_size)
            # jitted_partial_reset 仍然依赖全局 agent.initial_actor_state
            env_state, ts, actor_state = jitted_partial_reset(
                env_state, ts, actor_state, done_mask, reset_rngs, env, agent
            )

            env_steps += CFG.rollout_len * CFG.batch_size

            # ---- 日志 ----
            if step % CFG.log_every_steps == 0:
                jax.block_until_ready(metrics)  # 确保指标计算完成
                total_raw_r = float(jax.device_get(jnp.sum(raw_rewards)))
                loss = float(jax.device_get(metrics.get('total_loss', 0.)))

                now_t = time.perf_counter()
                sps = CFG.rollout_len * CFG.batch_size * CFG.log_every_steps / (now_t - last_t + 1e-6)
                last_t = now_t

                pbar.set_description(
                    f"Step {step}|Mode:{mode_str}|L:{loss:.3f}|λ:{lambda_imitation:.2f}|"
                    f"R_raw:{total_raw_r:.1f}|R_μ:{reward_norm_state.mean:.2f}|"
                    f"SPS:{int(sps)}"
                )

                # 写入 CSV 日志（实时可视化使用）
                with TRAIN_LOG.open("a", newline="") as f:
                    w = csv.writer(f)
                    if f.tell() == 0:
                        w.writerow(["step", "loss", "lambda", "raw_reward", "mean_reward", "sps"])
                    w.writerow([step, loss, float(lambda_imitation), total_raw_r,
                                float(reward_norm_state.mean), int(sps)])

            # ---- 评估 (按环境步数) ----
            current_eval_idx = env_steps // CFG.eval_every_env_steps
            if current_eval_idx > last_eval_idx:
                last_eval_idx = current_eval_idx
                jax.block_until_ready(learner_state)  # 确保参数已更新

                eval_params = jax.device_get(learner_state.params)

                avg_score, max_tile = 0., 0
                latest_frames = []
                for i in range(CFG.eval_num_games):
                    key, ev_key = jax.random.split(key)
                    score, tile, _, frames = run_eval(
                        eval_params,
                        agent._network,  # <--- 传递 agent 的 haiku transform
                        agent.initial_actor_state,  # <--- 传递 agent 的状态初始化函数
                        eval_env,
                        env_step_eval,
                        env_reset_eval,
                        mask_eval,
                        ev_key,
                        step,
                        verbose=(i == 0),  # 只打印第一局
                        capture_frames=(i == 0)
                    )
                    avg_score += score
                    max_tile = max(max_tile, tile)
                    if i == 0:
                        latest_frames = frames

                avg_score /= CFG.eval_num_games
                print(f"\n{'=' * 60}")
                print(f"评估 @ Step {step} (EnvSteps {env_steps})")
                print(f"  平均得分: {avg_score:.1f} | 最高方块: {max_tile}")
                print(f"{'=' * 60}\n")

                artifact_path = None
                if latest_frames:
                    artifact_path = ART_DIR / f"step_{env_steps}_{model_kind}.gif"
                    try:
                        # 生成彩色高分辨率帧 (每格 64x64 像素)
                        cell = 64
                        palette = {
                            0: (205, 193, 180),
                            1: (238, 228, 218),
                            2: (237, 224, 200),
                            3: (242, 177, 121),
                            4: (245, 149, 99),
                            5: (246, 124, 95),
                            6: (246, 94, 59),
                            7: (237, 207, 114),
                            8: (237, 200, 80),
                            9: (237, 197, 63),
                            10: (237, 194, 46),
                        }
                        def color_for(v):
                            if v in palette:
                                return palette[v]
                            # 渐变：更大 tile 逐步加深红色
                            t = min(v - 10, 10)
                            r = min(255, 180 + t * 7)
                            g = max(60, 194 - t * 8)
                            b = max(30, 46 - t * 2)
                            return (r, g, b)

                        imgs = []
                        for b in latest_frames:
                            bnp = np.array(b)
                            rgb = np.zeros((4, 4, 3), dtype=np.uint8)
                            for r in range(4):
                                for c in range(4):
                                    rgb[r, c] = color_for(int(bnp[r, c]))
                            # 上采样到高分辨率
                            big = np.kron(rgb, np.ones((cell, cell, 1), dtype=np.uint8))
                            imgs.append(big)
                        imageio.mimsave(artifact_path, imgs, duration=0.12)
                        print(f"✓ 评估 GIF 已保存 {artifact_path}")
                    except Exception as e:
                        print(f"? 保存评估 GIF 失败: {e}")

                with EVAL_LOG.open("a", newline="") as f:
                    w = csv.writer(f)
                    if f.tell() == 0:
                        w.writerow(["env_steps", "step", "avg_score", "max_tile", "artifact"])
                    w.writerow([env_steps, step, avg_score, max_tile, str(artifact_path) if artifact_path else ""])

                # 保存
                try:
                    flat_params = {'/'.join(str(k.key) for k in path): v for path, v in
                                   jax.tree_util.tree_flatten_with_path(eval_params)[0]}
                    np.savez(CFG.save_path, **flat_params)
                except Exception as e:
                    print(f"✗ 保存最近权重失败: {e}")

                if avg_score > best_score:
                    best_score = avg_score
                    try:
                        np.savez(CFG.best_path, **flat_params)
                        print(f"✓ 新最佳得分 {best_score:.1f}，已保存: {CFG.best_path}")
                    except Exception as e:
                        print(f"✗ 保存最佳权重失败: {e}")

                # 保存检查点（含优化器/状态/步数）
                try:
                    ckpt = dict(
                        learner_state=learner_state,
                        actor_state=actor_state,
                        env_state=env_state,
                        ts=ts,
                        reward_norm_state=reward_norm_state,
                        last_eval_idx=last_eval_idx,
                        best_score=best_score,
                        env_steps=env_steps,
                        step=step + 1,
                        key=key,
                    )
                    with CKPT_PATH.open("wb") as f:
                        pickle.dump(ckpt, f)
                except Exception as e:
                    print(f"✗ 保存检查点失败: {e}")

    except KeyboardInterrupt:
        print("\n\n检测到中断，正在保存...")
    finally:
        # 确保最后保存
        try:
            flat_params_cpu = jax.device_get(learner_state.params)
            flat_params = {'/'.join(str(k.key) for k in path): v for path, v in
                           jax.tree_util.tree_flatten_with_path(flat_params_cpu)[0]}
            np.savez(CFG.save_path, **flat_params)
            print(f"✓ 最终权重已保存: {CFG.save_path}")
        except Exception as e:
            print(f"✗ 紧急保存失败: {e}")
        # 最终检查点
        try:
            ckpt = dict(
                learner_state=learner_state,
                actor_state=actor_state,
                env_state=env_state,
                ts=ts,
                reward_norm_state=reward_norm_state,
                last_eval_idx=last_eval_idx,
                best_score=best_score,
                env_steps=env_steps,
                step=step + 1,
                key=key,
            )
            with CKPT_PATH.open("wb") as f:
                pickle.dump(ckpt, f)
            print(f"✓ 检查点已保存: {CKPT_PATH}")
        except Exception as e:
            print(f"✗ 保存检查点失败: {e}")

    print("=" * 60, "\n训练完成\n最佳得分:", best_score, "\n", "=" * 60)


if __name__ == "__main__":
    main()

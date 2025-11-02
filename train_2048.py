# 文件名: train_2048.py
# 作用：加载 Disco103 规则（“老师”），训练您的 CNN AI（“学生”）。
#
# *** 新功能：每 100 步训练后，自动运行一次完整的评估游戏 ***
# *** 【用户修改】:
#     1. 使用基于时间的随机种子进行初始化。
#     2. Rollout 期间不自动重置 (auto_reset=False)，即使游戏结束也运行完 rollout_len。
# ***

import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk
import optax
import chex
from jax import lax
from ml_collections import config_dict
import tqdm  # 用于显示进度条
import os
import jax.tree_util
import jax.lax  # <-- 【新添加】为评估函数导入
from dm_env import StepType  # <-- 【新添加】为评估函数导入
import time  # <-- 【新添加】为评估函数导入

# 导入 disco_rl 库的核心组件
from disco_rl import agent as agent_lib
from disco_rl import types
from disco_rl import utils

# *** 导入您创建的 2048 环境 ***
from disco_rl.environments import jittable_2048
# 【新增】提高 matmul 精度，兼顾数值与吞吐
from jax import config as jax_config

jax_config.update("jax_default_matmul_precision", "high")


# ##################################################################
# 1. 训练辅助函数 (来自 colabs/eval.ipynb)
# ##################################################################

def unflatten_params(flat_params: chex.ArrayTree) -> chex.ArrayTree:
    """
    将 .npz 文件中的扁平权重转换为 Haiku 兼容的嵌套字典。
    """
    params = {}
    for key_wb in flat_params:
        key = '/'.join(key_wb.split('/')[:-1])
        if key not in params:
            params[key] = {}
        # 键可能包含 'b' 或 'w'
        param_name = key_wb.split('/')[-1]
        params[key][param_name] = flat_params[key_wb]
    return params


def unroll_jittable_actor(
        params,
        actor_state,
        ts,
        env_state,
        rng,
        env,
        rollout_len,
        actor_step_fn,
):
    """
    在 jittable 环境中 unroll 策略 (用于训练)。
    """

    # ... (代码与您提供的完全相同) ...
    def _single_step(carry, step_rng):
        env_state, ts, actor_state = carry
        actor_timestep, actor_state = actor_step_fn(
            params, step_rng, ts, actor_state
        )

        # #################################################################
        # 【修改】将 auto_reset=True 更改为 auto_reset=False
        # 这将确保 rollout 持续 2048 步，即使游戏在此期间结束。
        # 环境在结束后将简单地停止响应并返回 0 奖励。
        env_state, ts = env.step(env_state, actor_timestep.actions, auto_reset=False)
        # #################################################################

        return (env_state, ts, actor_state), actor_timestep

    (env_state, ts, actor_state), actor_rollout = jax.lax.scan(
        _single_step,
        (env_state, ts, actor_state),
        jax.random.split(rng, rollout_len),
    )
    actor_rollout = types.ActorRollout.from_timestep(actor_rollout)
    return actor_rollout, actor_state, ts, env_state


# 【新增】采样步骤
def _sample_step(
        learner_params,
        actor_state,
        ts,
        env_state,
        actor_rng,
        env,
        rollout_len,
        actor_step_fn,
):
    """仅执行采样。"""
    actor_rollout, new_actor_state, new_ts, new_env_state = unroll_jittable_actor(
        learner_params,
        actor_state,
        ts,
        env_state,
        actor_rng,
        env,
        rollout_len,
        actor_step_fn,
    )
    return actor_rollout, new_actor_state, new_ts, new_env_state


# 【新增】学习步骤
def _learn_step(
        learner_state,
        initial_actor_state,
        actor_rollout,
        learner_rng,
        update_rule_params,
):
    """仅执行学习。"""
    # 防止与 actor_rollout 共享底层缓冲，避免双重 donation
    initial_actor_state = jax.tree_util.tree_map(lambda x: jnp.array(x, copy=True), initial_actor_state)

    new_learner_state, _, metrics = agent.learner_step(
        learner_rng,
        actor_rollout,
        learner_state,
        initial_actor_state,
        update_rule_params,
        is_meta_training=False,
    )
    return new_learner_state, metrics


# 【新增】单步训练迭代：采样+学习 合并为一次 JIT 调用
def _train_step(
        learner_state,
        actor_state,
        ts,
        env_state,
        actor_rng,
        learner_rng,
        update_rule_params,
        env,
        rollout_len,
        actor_step_fn,
):
    # 采样（使用当前 params）
    actor_rollout, new_actor_state, new_ts, new_env_state = unroll_jittable_actor(
        learner_state.params,
        actor_state,
        ts,
        env_state,
        actor_rng,
        env,
        rollout_len,
        actor_step_fn,
    )
    # 学习（基于这段 rollout）
    new_learner_state, _, metrics = agent.learner_step(
        learner_rng,
        actor_rollout,
        learner_state,
        actor_state,
        update_rule_params,
        is_meta_training=False,
    )
    # 返回必要信息（rewards 用于快速日志）
    return (
        new_learner_state,
        new_actor_state,
        new_ts,
        new_env_state,
        metrics,
        actor_rollout.rewards,
    )


# ##################################################################
# 2. 【新添加】评估辅助函数 (来自 evaluate_2048.py)
# ##################################################################

# ... (print_board, _slide_and_merge_one_line, get_valid_moves_mask, run_evaluation 函数
#  ... 与您提供的代码完全相同，此处省略以节省空间) ...

def print_board(board_state: jnp.ndarray, score: float, step: int):
    """一个简单的函数，用于在终端中打印 2048 棋盘。"""
    print("\n" + "=" * 29)
    print(f"EVAL 步骤: {step} | 总分数: {score}")
    print("=" * 29)
    for row in board_state:
        row_str = "|"
        for tile_log in row:
            if tile_log == 0:
                row_str += "      |"
            else:
                value = int(2 ** tile_log)
                row_str += f" {value:4d} |"
        print(row_str)
    print("=" * 29)


# --- 复制自 jittable_2048.py，用于检查移动是否有效 ---
@jax.jit
def _slide_and_merge_one_line(line: chex.Array) -> tuple[chex.Array, float]:
    """
    (代码与 jittable_2048.py 中的完全相同)
    """

    def _slide_step(carry_empty_idx, val):
        def _is_empty(carry):
            return carry, jnp.zeros_like(line)

        def _is_tile(carry):
            return carry + 1, jax.lax.dynamic_update_index_in_dim(
                jnp.zeros_like(line), val, carry, 0)

        new_empty_idx, update_slice = lax.cond(
            val == 0, _is_empty, _is_tile, carry_empty_idx
        )
        return new_empty_idx, update_slice

    _, slid_line_parts = lax.scan(_slide_step, 0, line)
    slid_line = jnp.sum(slid_line_parts, axis=0)

    def _merge_step(carry, i):
        merged_line, score, skip_next = carry

        def _process_tile():
            can_merge = jnp.logical_and(
                i < line.shape[0] - 1,
                jnp.logical_and(slid_line[i] > 0, slid_line[i] == slid_line[i + 1])
            )

            def _merge_fn():
                merged_val = slid_line[i] + 1
                new_score = score + jnp.power(2, merged_val)
                updated_line = merged_line.at[i].set(merged_val)
                return updated_line, new_score, True

            def _no_merge_fn():
                updated_line = merged_line.at[i].set(slid_line[i])
                return updated_line, score, False

            return lax.cond(can_merge, _merge_fn, _no_merge_fn)

        return lax.cond(
            skip_next,
            lambda: (merged_line, score, False),
            _process_tile
        ), None

    (merged_line, merged_score, _), _ = lax.scan(
        _merge_step,
        (jnp.zeros_like(line), 0.0, False),
        jnp.arange(line.shape[0])
    )
    _, final_line_parts = lax.scan(_slide_step, 0, merged_line)
    final_line = jnp.sum(final_line_parts, axis=0)
    return final_line, merged_score


@jax.jit
def get_valid_moves_mask(board: chex.Array) -> chex.Array:
    """
    (代码与 evaluate_2048.py 中的完全相同)
    """
    vmap_slide_merge = jax.vmap(_slide_and_merge_one_line, in_axes=0)
    # 1. 模拟“上” (action 0)
    new_board_up, _ = vmap_slide_merge(board.T)
    is_valid_up = jnp.any(new_board_up.T != board)
    # 2. 模拟“下” (action 1)
    new_board_down, _ = vmap_slide_merge(jnp.flip(board.T, axis=1))
    is_valid_down = jnp.any(jnp.flip(new_board_down, axis=1).T != board)
    # 3. 模拟“左” (action 2)
    new_board_left, _ = vmap_slide_merge(board)
    is_valid_left = jnp.any(new_board_left != board)
    # 4. 模拟“右” (action 3)
    new_board_right, _ = vmap_slide_merge(jnp.flip(board, axis=1))
    is_valid_right = jnp.any(jnp.flip(new_board_right, axis=1) != board)
    return jnp.array([is_valid_up, is_valid_down, is_valid_left, is_valid_right])


# --- 【新添加】评估函数 ---
def run_evaluation(
        trained_params: chex.ArrayTree,
        actor_state_eval: chex.ArrayTree,
        eval_rng: chex.PRNGKey,
        eval_env: jittable_2048.BatchedJittable2048Environment,
        jitted_policy_step_eval: callable,
        jitted_env_step_eval: callable,
        jitted_env_reset_eval: callable,
        jitted_get_mask_eval: callable,
        step: int
):
    """
    运行一个完整的评估游戏，使用确定性策略（Argmax + 动作掩码）。
    """
    print("\n" + "*" * 30)
    print(f"--- 正在运行评估游戏 (Step {step}) ---")

    # 重置评估环境
    env_state, timestep = jitted_env_reset_eval(eval_rng)

    total_game_score = 0.0
    game_steps = 0

    # 评估游戏循环
    while True:
        current_board = env_state.state[0][0]  # 提取 (4,4) 棋盘

        # 检查游戏是否结束
        if timestep.step_type[0] == StepType.LAST:
            print("\n!!! 评估游戏结束 !!!")
            print_board(current_board, total_game_score, game_steps)
            print(f"--- 评估完成: 最终得分 = {total_game_score} ---")
            print("*" * 30 + "\n")
            break

        # AI 决策 (Argmax + Masking)
        agent_outs, new_actor_state_eval = jitted_policy_step_eval(
            trained_params,
            actor_state_eval,
            timestep.observation,
            timestep.step_type == StepType.LAST
        )
        action_logits = agent_outs['logits'][0]
        valid_mask = jitted_get_mask_eval(current_board)
        action_logits = jnp.where(valid_mask, action_logits, -jnp.inf)
        action = jnp.argmax(action_logits, axis=-1)

        # 执行动作
        eval_rng, env_rng = jax.random.split(eval_rng)
        env_state, timestep = jitted_env_step_eval(
            env_state,
            jnp.array([action]),  # 动作必须是批量的
            auto_reset=False
        )

        actor_state_eval = new_actor_state_eval  # 更新评估 RNN 状态
        total_game_score += timestep.reward[0]
        game_steps += 1

        # 添加一个安全中断，以防 AI 陷入无限循环
        if game_steps > 10000:  # 10000 步
            print("评估超过 10000 步，强制终止。")
            break

    return total_game_score


# ##################################################################
# 3. 主训练函数
# ##################################################################

# 使 agent 成为全局变量，以便 JIT 编译的 unroll_fn 可以访问它
agent: agent_lib.Agent = None


def main():
    global agent  # 声明我们将使用全局 agent

    print("--- 1. 加载 Disco103 权重 ('老师'的大脑) ---")
    try:
        with open('disco_103.npz', 'rb') as file:
            disco_103_flat_params = np.load(file, allow_pickle=True)
            disco_103_params = unflatten_params(disco_103_flat_params)
        print(f"成功加载 {len(disco_103_params)} 个 Disco103 规则参数层。")
    except FileNotFoundError:
        print("错误：未找到 'disco_103.npz'。")
        print("请在项目根目录中确保该文件存在。")
        return

    print("\n--- 2. 配置 2048 Agent ('学生') ---")

    # (a) 获取 Disco103 规则的默认 Agent 设置
    agent_settings = agent_lib.get_settings_disco()

    # (b) **关键**：重写网络设置以使用您的 CNN
    agent_settings.net_settings.name = 'cnn'  # 告诉 agent 使用 'cnn'

    # (c) 调整 CNN 的参数
    agent_settings.net_settings.net_args = dict(
        model_arch_name='lstm',
        head_w_init_std=1e-2,
        model_kwargs=dict(
            head_mlp_hiddens=(128,),
            lstm_size=128,
        ),
        conv_channels=(128, 256),
        mlp_hiddens=(256,),
    )
    agent_settings.learning_rate = 1e-4

    # (d) 实例化 *训练* 环境
    batch_size = 1
    env = jittable_2048.BatchedJittable2048Environment(
        batch_size=batch_size,
        env_settings=jittable_2048.get_config_2048()
    )
    print(f"2048 训练环境 (Batch={batch_size}) 已创建。")

    # --- 【新添加】实例化 *评估* 环境 ---
    eval_env = jittable_2048.BatchedJittable2048Environment(
        batch_size=1,  # 评估时只玩一个游戏
        env_settings=jittable_2048.get_config_2048()
    )
    print(f"2048 评估环境 (Batch=1) 已创建。")

    # (e) **关键**：实例化 Agent
    agent = agent_lib.Agent(
        agent_settings=agent_settings,
        single_observation_spec=env.single_observation_spec(),
        single_action_spec=env.single_action_spec(),
        batch_axis_name=None,
    )
    print("CNN 策略 Agent ('学生') 已实例化。")

    # #################################################################
    # 【修改】使用当前时间作为种子，确保每次运行都不同
    current_time_seed = int(time.time())
    print(f"\n--- 3. 初始化状态... | 使用种子: {current_time_seed} ---")
    # (a) 定义超参数
    num_steps = 10000
    rollout_len = 2048
    eval_every_n_steps = 500

    # (b) 初始化状态
    rng_key = jax.random.PRNGKey(current_time_seed)
    # #################################################################

    rng_key, env_rng, learner_rng, actor_rng, eval_actor_rng = jax.random.split(rng_key, 5)

    env_state, timestep = env.reset(env_rng)
    learner_state = agent.initial_learner_state(learner_rng)
    actor_state = agent.initial_actor_state(actor_rng)  # 训练 RNN 状态
    update_rule_params = disco_103_params

    # --- 【新添加】评估所需的状态 ---
    # 我们为评估游戏初始化一个单独的 RNN 状态
    eval_actor_state = agent.initial_actor_state(eval_actor_rng)

    # --- 【新添加的预加载逻辑】 ---
    save_path = "my_2048_agent_weights.npz"
    best_save_path = "my_2048_agent_weights_best.npz"  # 【新增】最好权重
    if os.path.exists(save_path):
        print(f"\n--- 3.5. 检测到 '{save_path}'，正在加载权重... ---")
        try:
            with open(save_path, 'rb') as file:
                flat_params = np.load(file, allow_pickle=True)
                loaded_params_tree = unflatten_params(flat_params)

            learner_state = learner_state.replace(params=loaded_params_tree)
            print("权重加载成功，将从该检查点继续训练。")
        except Exception as e:
            print(f"加载权重失败: {e}。将从零开始训练。")
            learner_state = agent.initial_learner_state(learner_rng)
    else:
        print(f"\n--- 3.5. 未检测到 '{save_path}'，将从零开始训练... ---")
    # --- 【预加载逻辑结束】 ---

    # (c) JIT 编译 *训练* 函数
    # 将采样和学习分离，以启用 donate_argnums
    print("JAX 正在编译采样和学习函数...")
    jitted_sample_step = jax.jit(
        _sample_step,
        static_argnames=('env', 'rollout_len', 'actor_step_fn'),
        # 仅捐赠 ts, env_state
        donate_argnums=(2, 3),
    )
    jitted_learn_step = jax.jit(
        _learn_step,
        static_argnames=(),
        # 关闭 donation，避免库内部 donation 与别名导致的双重捐赠
        donate_argnums=(),
    )

    # --- 【新添加】JIT 编译 *评估* 函数 ---
    print("JAX 正在编译评估函数...")
    jitted_policy_step_eval = jax.jit(agent._network.one_step)  #
    jitted_env_step_eval = jax.jit(eval_env.step, static_argnames='auto_reset')
    jitted_env_reset_eval = jax.jit(eval_env.reset)
    jitted_get_mask_eval = jax.jit(get_valid_moves_mask)

    print("\n--- 4. 开始训练循环 ---")
    print("JAX 编译完成，开始训练...")

    # 【新增】吞吐与最佳分追踪
    best_eval_score = -float('inf')
    last_t = time.perf_counter()

    pbar = tqdm.tqdm(range(num_steps))
    try:
        for step in pbar:
            rng_key, actor_rng, learner_rng = jax.random.split(rng_key, 3)

            # (A) 采样步骤
            initial_actor_state_for_rollout = actor_state  # 保存初始状态用于学习
            actor_rollout, actor_state, timestep, env_state = jitted_sample_step(
                learner_state.params,
                actor_state,
                timestep,
                env_state,
                actor_rng,
                env,
                rollout_len,
                agent.actor_step,
            )

            # 【注意】因为 auto_reset=False，如果游戏在 rollout 期间结束，
            # env_state 和 timestep 将会“卡在”最后一步。
            # 我们必须在学习 *之后* 手动检查并重置它。

            # (B) 学习步骤
            learner_state, metrics = jitted_learn_step(
                learner_state,
                initial_actor_state_for_rollout,
                actor_rollout,
                learner_rng,
                update_rule_params,
            )

            # 【新增的重置逻辑】
            # 检查这批 rollout 是否已经结束了 (step_type == LAST)
            # timestep.step_type[0] 是我们从采样步骤中获得的 *最终* 时间步
            if timestep.step_type[0] == StepType.LAST:
                # print(f"\n[DEBUG] Rollout 在 {step} 结束，重置环境。\n")
                rng_key, env_rng, actor_rng = jax.random.split(rng_key, 3)
                env_state, timestep = env.reset(env_rng)
                actor_state = agent.initial_actor_state(actor_rng)

            # (D) 打印日志（减少 host<->device 往返）
            if step % 100 == 0:
                # 仅在需要时将标量拉回主机；兼容缺失指标
                try:
                    # 这个 R: 现在是单局游戏在 2048 步内的奖励
                    total_reward = float(jax.device_get(jnp.sum(actor_rollout.rewards)))
                except Exception:
                    total_reward = 0.0

                def _maybe_get(d, k):
                    try:
                        return d[k]
                    except Exception:
                        try:
                            return d.get(k, None)
                        except Exception:
                            return None

                total_loss_val = _maybe_get(metrics, 'total_loss') or _maybe_get(metrics, 'loss')
                try:
                    total_loss = float(jax.device_get(total_loss_val))
                except Exception:
                    total_loss = 0.0

                adv_val_container = _maybe_get(metrics, 'meta_out')
                adv_val = None
                if adv_val_container is not None:
                    adv_val = _maybe_get(adv_val_container, 'adv')
                try:
                    adv_value = float(jax.device_get(adv_val).mean())
                except Exception:
                    adv_value = 0.0

                now_t = time.perf_counter()
                sps = rollout_len * env.batch_size / max(1e-6, (now_t - last_t))
                last_t = now_t

                pbar.set_description(
                    f"Step {step} | R:{total_reward:.1f} | Loss:{total_loss:.4f} | Adv:{adv_value:.4f} | SPS:{sps:.0f}"
                )

            # --- 【评估与断点】 ---
            if (step % eval_every_n_steps == 0) and (step > 0):
                rng_key, eval_run_rng = jax.random.split(rng_key)
                eval_score = run_evaluation(
                    learner_state.params,
                    eval_actor_state,
                    eval_run_rng,
                    eval_env,
                    jitted_policy_step_eval,
                    jitted_env_step_eval,
                    jitted_env_reset_eval,
                    jitted_get_mask_eval,
                    step=step
                )

                # 【新增】保存最近权重
                try:
                    leaves, treedef = jax.tree_util.tree_flatten_with_path(learner_state.params)
                    flat_params = {'/'.join(str(k.key) for k in path): value for path, value in leaves}
                    flat_params_cpu = jax.device_get(flat_params)
                    np.savez(save_path, **flat_params_cpu)
                except Exception as e:
                    print(f"保存最近权重失败: {e}")

                # 【新增】保存最好权重
                if eval_score > best_eval_score:
                    best_eval_score = eval_score
                    try:
                        np.savez(best_save_path, **flat_params_cpu)
                        print(f"新最佳分 {best_eval_score:.1f}，已保存到: {best_save_path}")
                    except Exception as e:
                        print(f"保存最好权重失败: {e}")

    except KeyboardInterrupt:
        print("\n检测到中断，正在安全保存当前权重...")
        try:
            leaves, treedef = jax.tree_util.tree_flatten_with_path(learner_state.params)
            flat_params = {'/'.join(str(k.key) for k in path): value for path, value in leaves}
            flat_params_cpu = jax.device_get(flat_params)
            np.savez(save_path, **flat_params_cpu)
            print(f"已保存到: {save_path}")
        except Exception as e:
            print(f"紧急保存失败: {e}")
        return

    print("--- 训练完成 ---")

    print("\n--- 5. 正在保存已训练的 CNN 权重 ---")
    # learner_state 包含了我们训练好的最终权重 (params)
    leaves, treedef = jax.tree_util.tree_flatten_with_path(learner_state.params)
    flat_params = {'/'.join(str(k.key) for k in path): value for path, value in leaves}
    flat_params_cpu = jax.device_get(flat_params)

    # save_path 已在上方定义
    np.savez(save_path, **flat_params_cpu)
    print(f"权重已成功保存到: {save_path}")


if __name__ == "__main__":
    main()
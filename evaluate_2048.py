# 文件名: evaluate_2048.py
# 作用：加载已训练的 AI 权重 (my_2048_agent_weights.npz)，
#       并让它玩一个完整的 2048 游戏，直到结束。
#
# *** 新功能：使用与训练时一致的 "sample()" 策略 (随机探索) ***

import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk
import chex
from dm_env import StepType
import time
import distrax  # <-- 【新添加】导入 distrax 用于采样

# 导入 disco_rl 库的核心组件
from disco_rl import agent as agent_lib
from disco_rl import types
from disco_rl import utils

# *** 导入您创建的 2048 环境 ***
from disco_rl.environments import jittable_2048
# *** 导入 disco_rl.networks 模块 (确保 nets.py 已被修改) ***
from disco_rl import networks


# ##################################################################
# 1. 辅助函数
# ##################################################################

def unflatten_params(flat_params: chex.ArrayTree) -> chex.ArrayTree:
    """
    将 .npz 文件中的扁平权重转换为 Haiku 兼容的嵌套字典。
    (与 train_2048.py 中的函数相同)
    """
    params = {}
    for key_wb in flat_params:
        key = '/'.join(key_wb.split('/')[:-1])
        if key not in params:
            params[key] = {}
        param_name = key_wb.split('/')[-1]
        params[key][param_name] = flat_params[key_wb]
    return params


def print_board(board_state: jnp.ndarray, score: float, step: int):
    """一个简单的函数，用于在终端中打印 2048 棋盘。"""
    print("\n" + "=" * 29)
    print(f"步骤: {step} | 总分数: {score}")
    print("=" * 29)
    # board_state 的值是 log2(tile)，0 是空格
    for row in board_state:
        row_str = "|"
        for tile_log in row:
            if tile_log == 0:
                row_str += "      |"
            else:
                # 2^tile_log
                value = int(2 ** tile_log)
                row_str += f" {value:4d} |"
        print(row_str)
    print("=" * 29)


# ##################################################################
# 2. 【已删除】不再需要动作掩码 (Action Masking) 辅助函数
# ##################################################################


# ##################################################################
# 3. 主评估函数
# ##################################################################

def main():
    print("--- 1. 准备环境和 AI (学生) ---")

    # (A) 实例化 Agent (必须使用与训练时完全相同的设置)
    agent_settings = agent_lib.get_settings_disco()
    agent_settings.net_settings.name = 'cnn'
    agent_settings.net_settings.net_args = dict(
        model_arch_name='lstm',
        head_w_init_std=1e-2,
        model_kwargs=dict(head_mlp_hiddens=(128,), lstm_size=128),
        conv_channels=(128, 256),
        mlp_hiddens=(256,),
    )

    # (B) 实例化 2048 环境 (只需要 batch_size=1)
    env = jittable_2048.BatchedJittable2048Environment(
        batch_size=1,  # 评估时只玩一个游戏
        env_settings=jittable_2048.get_config_2048()
    )

    # (C) 实例化 Agent
    agent = agent_lib.Agent(
        agent_settings=agent_settings,
        single_observation_spec=env.single_observation_spec(),
        single_action_spec=env.single_action_spec(),
        batch_axis_name=None,
    )

    print("--- 2. 加载已训练的权重 ('学生'的大脑) ---")
    try:
        # 加载您在 train_2048.py 中保存的文件
        # my_2048_agent_weights.npz
        # my_2048_agent_weights_best.npz
        with open('my_2048_agent_weights.npz', 'rb') as file:
            flat_params = np.load(file, allow_pickle=True)
            trained_params = unflatten_params(flat_params)
        print("权重 'my_2048_agent_weights.npz' 加载成功。")
    except FileNotFoundError:
        print("错误：未找到 'my_2048_agent_weights.npz'。")
        print("请先运行 train_2048.py 来生成权重文件。")
        return

    # (D) JIT 编译我们需要的函数
    jitted_policy_step = jax.jit(agent._network.one_step)  # (使用 _network)
    jitted_env_step = jax.jit(env.step, static_argnames='auto_reset')
    jitted_env_reset = jax.jit(env.reset)
    # 【已删除】不再需要 jitted_get_mask

    print("--- 3. 开始运行 AI 玩 2048 (评估模式) ---")

    rng_key = jax.random.PRNGKey(178)  # 评估时使用固定的种子

    # (E) 重置环境和 AI 状态
    rng_key, env_rng, actor_rng = jax.random.split(rng_key, 3)
    env_state, timestep = jitted_env_reset(env_rng)
    actor_state = agent.initial_actor_state(actor_rng)

    total_game_score = 0.0
    game_steps = 0

    # (F) 游戏循环
    while True:
        # 1. 打印当前棋盘
        current_board = env_state.state[0][0]
        print_board(current_board, total_game_score, game_steps)

        # 2. 检查游戏是否结束
        if timestep.step_type[0] == StepType.LAST:
            print("\n!!! 游戏结束 !!!")
            print(f"最终得分: {total_game_score}")
            print(f"总步数: {game_steps}")
            break

        # 3. AI 决策 (关键！)
        agent_outs, new_actor_state = jitted_policy_step(
            trained_params,
            actor_state,
            timestep.observation,
            timestep.step_type == StepType.LAST
        )

        # --- 【关键修改：使用 'sample' 替代 'argmax'】 ---
        # 4. 使用与训练时相同的“随机采样”策略
        #    (agent.actor_step 内部就是这样做的)

        # 4a. 获取一个新的随机数种子用于采样
        rng_key, act_rng = jax.random.split(rng_key)

        # 4b. agent_outs['logits'] 的形状是 [1, 4] (batch_size=1)
        action_logits = agent_outs['logits']

        # 4c. 从 Softmax 概率分布中 *采样* 一个动作
        action_dist = distrax.Softmax(logits=action_logits)
        action = action_dist.sample(seed=act_rng)
        # --- 【修改结束】 ---

        # 5. 在环境中执行动作
        #    (action 已经是 [1] 数组，因为采样器保留了 batch 维度)
        rng_key, env_rng = jax.random.split(rng_key)
        new_env_state, new_timestep = jitted_env_step(
            env_state,
            action,
            auto_reset=False
        )

        # 6. 更新状态
        env_state = new_env_state
        timestep = new_timestep
        actor_state = new_actor_state
        game_steps += 1

        # 7. 累加分数
        total_game_score += new_timestep.reward[0]

        # 8. 暂停一下，以便我们观看
        time.sleep(0.1)  # 暂停 0.1 秒


if __name__ == "__main__":
    main()
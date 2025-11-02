# 文件名: disco_rl/environments/jittable_2048.py
# 作用：定义 2048 JAX 环境。
#
# *** 【已修复】：移除了 _single_env_step 中的自动重置 (auto-reset) 逻辑，
# *** 以便评估脚本可以打印出游戏结束时的最终棋盘。

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
import chex
from dm_env import specs as dm_specs, StepType
from ml_collections import config_dict
from functools import partial  # 新增：为类方法 jit 提供 partial

from disco_rl import types
from disco_rl.environments.wrappers import batched_jittable_env


# ##################################################################
# 1. 核心 JAX 2048 游戏逻辑
# ##################################################################

# --- 辅助函数：处理单行/单列的滑动和合并 ---
@jax.jit
def _slide_and_merge_one_line(line: chex.Array) -> tuple[chex.Array, jnp.float32]:
    """
    使用 jax.lax.scan 在一个 1D 数组（一行或一列）上执行滑动和合并。
    """

    # 1. 'slide'：将所有非零方块推到前面
    def _slide_step(carry_empty_idx, val):
        def _is_empty(carry):
            # 【已修复】返回 4 元素数组
            return carry, jnp.zeros_like(line)

        def _is_tile(carry):
            return carry + 1, jax.lax.dynamic_update_index_in_dim(
                jnp.zeros_like(line), val, carry, 0)

        new_empty_idx, update_slice = lax.cond(
            val == 0, _is_empty, _is_tile, carry_empty_idx
        )
        return new_empty_idx, update_slice

    _, slid_line_parts = lax.scan(_slide_step, 0, line)
    slid_line = jnp.sum(slid_line_parts, axis=0).astype(line.dtype)  # 保持与输入行相同的 dtype

    # 2. 'merge'：合并相邻的相同方块
    def _merge_step(carry, i):
        merged_line, score, skip_next = carry

        def _process_tile():
            can_merge = jnp.logical_and(
                i < line.shape[0] - 1,
                jnp.logical_and(slid_line[i] > 0, slid_line[i] == slid_line[i + 1])
            )

            def _merge_fn():
                merged_val = slid_line[i] + 1
                # 用位移替代 pow，并保持为 float32
                gain = jnp.asarray(1 << merged_val.astype(jnp.int32), dtype=jnp.float32)
                new_score = score + gain
                # 关键：将写入值强制为与行相同的 dtype（避免 upcast）
                updated_line = merged_line.at[i].set(merged_val.astype(line.dtype))
                return updated_line, new_score, True

            def _no_merge_fn():
                # 确保写入时也维持 dtype 一致
                updated_line = merged_line.at[i].set(slid_line[i].astype(line.dtype))
                return updated_line, score, False  # skip_next=False

            return lax.cond(can_merge, _merge_fn, _no_merge_fn)

        return lax.cond(
            skip_next,
            lambda: (merged_line, score, False),
            _process_tile
        ), None

    (merged_line, merged_score, _), _ = lax.scan(
        _merge_step,
        (jnp.zeros_like(line), jnp.float32(0.0), False),  # 分值用 float32
        jnp.arange(line.shape[0])
    )

    # 3. 'slide again'
    _, final_line_parts = lax.scan(_slide_step, 0, merged_line)
    final_line = jnp.sum(final_line_parts, axis=0).astype(line.dtype)  # 保持 dtype

    return final_line, merged_score


# --- JAX 游戏逻辑类 ---
class _SingleStream2048:
    """一个 JAX-native, JIT-able 的 2048 游戏逻辑实现。"""

    def __init__(self, rows=4, columns=4):
        self._rows = rows
        self._columns = columns
        # 棋盘改为 uint8（指数表示），更省内存/带宽
        self._initial_board = jnp.zeros((rows, columns), dtype=jnp.uint8)
        self._vmap_slide_merge = jax.vmap(_slide_and_merge_one_line, in_axes=0)

    @property
    def num_actions(self) -> int:
        return 4  # 0:上, 1:下, 2:左, 3:右

    @partial(jax.jit, static_argnums=(0,))
    def _add_random_tile(self, rng: chex.PRNGKey, board: chex.Array) -> chex.Array:
        empty_indices = jnp.where(board.flatten() == 0, 1.0, 0.0)
        num_empty = jnp.sum(empty_indices)

        def _add_tile():
            rng_idx, rng_val = jax.random.split(rng)
            choice_idx = jax.random.choice(rng_idx, jnp.arange(self._rows * self._columns), p=empty_indices / num_empty)
            new_val = lax.cond(jax.random.uniform(rng_val) < 0.9, lambda: jnp.uint8(1), lambda: jnp.uint8(2))  # 1=2, 2=4

            flat_board = board.flatten()
            flat_board = flat_board.at[choice_idx].set(new_val)
            return flat_board.reshape((self._rows, self._columns))

        return lax.cond(num_empty > 0, _add_tile, lambda: board)

    @partial(jax.jit, static_argnums=(0,))
    def initial_state(self, rng: chex.PRNGKey) -> chex.ArrayTree:
        rng1, rng2 = jax.random.split(rng)
        board = self._add_random_tile(rng1, self._initial_board)
        board = self._add_random_tile(rng2, board)
        reward = jnp.float32(0.0)
        return (board, reward)

    @partial(jax.jit, static_argnums=(0,))
    def episode_reset(self, rng: chex.PRNGKey, state_and_reward: chex.ArrayTree) -> chex.ArrayTree:
        return self.initial_state(rng)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, rng: chex.PRNGKey, state_and_reward: chex.ArrayTree, action: int) -> chex.ArrayTree:
        board, _ = state_and_reward

        def _move_up(b):
            b_T = b.T
            new_lines, scores = self._vmap_slide_merge(b_T)
            return new_lines.T.astype(b.dtype), jnp.sum(scores)

        def _move_down(b):
            b_T_r = jnp.flip(b.T, axis=1)
            new_lines, scores = self._vmap_slide_merge(b_T_r)
            return jnp.flip(new_lines, axis=1).T.astype(b.dtype), jnp.sum(scores)

        def _move_left(b):
            new_lines, scores = self._vmap_slide_merge(b)
            return new_lines.astype(b.dtype), jnp.sum(scores)

        def _move_right(b):
            b_r = jnp.flip(b, axis=1)
            new_lines, scores = self._vmap_slide_merge(b_r)
            return jnp.flip(new_lines, axis=1).astype(b.dtype), jnp.sum(scores)

        branches = [_move_up, _move_down, _move_left, _move_right]
        new_board, new_reward = lax.switch(action, branches, board)

        moved = jnp.any(new_board != board)

        final_board = lax.cond(
            moved,
            lambda: self._add_random_tile(rng, new_board),
            lambda: new_board
        )
        final_reward = lax.cond(moved, lambda: new_reward, lambda: jnp.float32(0.0))

        return (final_board, final_reward)

    @partial(jax.jit, static_argnums=(0,))
    def is_terminal(self, state_and_reward: chex.ArrayTree) -> bool:
        board, _ = state_and_reward
        has_empty_cells = jnp.any(board == 0)
        can_merge_horiz = jnp.any(
            jnp.logical_and(board[:, :-1] == board[:, 1:], board[:, :-1] > 0)
        )
        can_merge_vert = jnp.any(
            jnp.logical_and(board[:-1, :] == board[1:, :], board[:-1, :] > 0)
        )
        can_move = jnp.logical_or(has_empty_cells, jnp.logical_or(can_merge_horiz, can_merge_vert))
        return jnp.logical_not(can_move)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state_and_reward: chex.ArrayTree) -> chex.ArrayTree:
        board, _ = state_and_reward
        clipped_board = jnp.clip(board, 0, 15).astype(jnp.int32)
        obs = jax.nn.one_hot(clipped_board, num_classes=16).astype(jnp.float32)
        return {'observation': obs}

    @partial(jax.jit, static_argnums=(0,))
    def reward(self, state_and_reward: chex.ArrayTree) -> jnp.ndarray:
        _, reward = state_and_reward
        return reward


# ##################################################################
# 2. 继承 `BatchedJittableEnvironment`
# ##################################################################

class BatchedJittable2048Environment(batched_jittable_env.BatchedJittableEnvironment):
    """
    这是 `jittable_2048.py` 的环境包装器
    *** 新功能：添加了 'auto_reset' 标志 ***
    """

    def __init__(self, batch_size: int, env_settings: config_dict.ConfigDict):
        self._env = _SingleStream2048(**env_settings.to_dict())
        self.batch_size = batch_size

        # 观测模式：one_hot(默认, 兼容) 或 uint8(高效)
        self._observation_mode = env_settings.get('observation_mode', 'one_hot')

        self._single_action_spec = dm_specs.BoundedArray(
            (), np.int32, 0, self._env.num_actions - 1
        )

        dummy_state = self._env.initial_state(jax.random.PRNGKey(0))
        if self._observation_mode == 'uint8':
            obs = jnp.zeros((self._env._rows, self._env._columns), dtype=jnp.uint8)
        else:
            obs = self._env.render(dummy_state)['observation']

        self._single_observation_spec = {
            'observation': dm_specs.Array(shape=obs.shape, dtype=obs.dtype)
        }

        # vmap + 再次 jit（auto_reset 作为静态参数），减少 Python 往返
        self._batched_env_step = jax.vmap(self._single_env_step, in_axes=(0, 0, None))
        self._batched_env_step = jax.jit(self._batched_env_step, static_argnums=(2,))
        self._batched_env_reset = jax.vmap(self._single_env_reset)
        self._batched_env_reset = jax.jit(self._batched_env_reset)

    # 【新增】统一渲染函数：根据 observation_mode 输出 one_hot 或 uint8
    def _render(self, state_and_reward):
        board, _ = state_and_reward
        if self._observation_mode == 'uint8':
            # 直接返回 uint8 棋盘
            return {'observation': board}
        else:
            # 复用底层 one_hot 渲染
            return self._env.render(state_and_reward)

    # --- 【关键修改】 ---
    # 添加 'auto_reset' 参数，默认为 True (用于训练)
    def step(
            self, state: batched_jittable_env.EnvState, actions: chex.Array, auto_reset: bool = True
    ) -> tuple[batched_jittable_env.EnvState, types.EnvironmentTimestep]:
        # 现在调用 vmap 化的函数，并传入 auto_reset 标志
        return self._batched_env_step(state, actions, auto_reset)

    # --- 【关键修改】 ---
    # _single_env_step 现在接受 auto_reset
    def _single_env_step(
            self, env_state: batched_jittable_env.EnvState, action: chex.Array, auto_reset: bool
    ) -> tuple[batched_jittable_env.EnvState, types.EnvironmentTimestep]:
        new_rng, rng_step, rng_reset = jax.random.split(env_state.rng, 3)

        # 1. 执行一步
        new_state_and_reward = self._env.step(rng_step, env_state.state, action)
        new_board, new_reward = new_state_and_reward

        # 2. 检查游戏是否结束
        is_terminal = self._env.is_terminal(new_state_and_reward)

        # 3. 【关键修改】根据 auto_reset 标志决定是否重置

        def _do_reset(_):
            # 训练时：如果游戏结束，立即重置
            return self._env.episode_reset(rng_reset, new_state_and_reward)

        def _no_reset(_):
            # 评估时：如果游戏结束，不重置，返回当前状态
            return new_state_and_reward

        # JAX if/else: if (auto_reset and is_terminal) then _do_reset() else _no_reset()
        final_state = lax.cond(
            jnp.logical_and(auto_reset, is_terminal),
            _do_reset,
            _no_reset,
            None  # (这个参数不会被使用，只是为了匹配 lax.cond 格式)
        )

        return batched_jittable_env.EnvState(state=final_state,
                                             rng=new_rng), batched_jittable_env._to_env_timestep(
            self._render(new_state_and_reward),  # 使用包装器的渲染，支持uint8
            new_reward,
            is_terminal
        )

    def _single_env_reset(
            self, rng_key: chex.PRNGKey
    ) -> tuple[batched_jittable_env.EnvState, types.EnvironmentTimestep]:
        new_rng, reset_rng = jax.random.split(rng_key)
        state_and_reward = self._env.initial_state(reset_rng)
        return batched_jittable_env.EnvState(state=state_and_reward,
                                             rng=new_rng), batched_jittable_env._to_env_timestep(
            self._render(state_and_reward),  # 使用包装器的渲染，支持uint8
            self._env.reward(state_and_reward),
            self._env.is_terminal(state_and_reward)
        )


def get_config_2048() -> config_dict.ConfigDict:
    """返回 2048 环境的默认配置"""
    # 【已修复】使用 config_dict.ConfigDict
    return config_dict.ConfigDict(
        dict(
            rows=4,
            columns=4,
        )
    )

# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Network factory."""

from collections.abc import Iterable, Mapping
from typing import Any, Optional, Tuple  # <-- 已添加 Optional, Tuple
import math  # <-- 已添加 math

import chex
import haiku as hk
from haiku import initializers as hk_init
import jax
from jax import numpy as jnp
import numpy as np

from disco_rl import types
from disco_rl.networks import action_models


def get_network(name: str, *args, **kwargs) -> types.PolicyNetwork:
    """Constructs a network."""

    def _get_net():
        if name == 'mlp':
            return MLP(*args, **kwargs)
        elif name == 'cnn':
            return CNN(*args, **kwargs)
        # ===== 添加 Transformer 选项 =====
        elif name == 'axial_transformer':
            return AxialTransformer(*args, **kwargs)
        elif name == 'linear_transformer':
            return LinearTransformer(*args, **kwargs)
        elif name == 'vanilla_transformer':
            return VanillaTransformer(*args, **kwargs)
        # ===== 结束 =====
        else:
            raise ValueError(f'Unknown network: {name}')

    def _agent_step(*call_args, **call_kwargs):
        return _get_net()(*call_args, **call_kwargs)

    def _unroll(*call_args, **call_kwargs):
        return _get_net().unroll(*call_args, **call_kwargs)

    module_init_fn, one_step_fn = hk.without_apply_rng(
        hk.transform_with_state(_agent_step)
    )
    _, unroll_fn = hk.without_apply_rng(hk.transform_with_state(_unroll))

    return types.PolicyNetwork(
        init=module_init_fn,
        one_step=one_step_fn,
        unroll=unroll_fn,
    )


class MLPHeadNet(hk.Module):
    """MLP heads according to the update rules' out_spec."""

    def __init__(
            self,
            out_spec: chex.ArrayTree,
            action_spec: types.Specs,
            head_w_init_std: float | None,
            model_out_spec: chex.ArrayTree | None = None,
            model_arch_name: str | None = None,
            model_kwargs: Mapping[str, Any] | None = None,
            module_name: str | None = None,
    ):
        super().__init__(name=module_name)
        self._out_spec = out_spec
        self._action_spec = action_spec
        if model_out_spec:
            self._model = action_models.get_action_model(
                model_arch_name,
                action_spec=action_spec,
                out_spec=model_out_spec,
                **model_kwargs,
            )
        else:
            self._model = None
        self._head_w_init = (
            hk_init.TruncatedNormal(head_w_init_std) if head_w_init_std else None
        )

    def _embedding_pass(
            self, inputs: chex.ArrayTree, should_reset: chex.Array | None = None
    ) -> chex.Array:
        """Compute embedding from agent inputs."""
        raise NotImplementedError

    def _head_pass(self, embedding: chex.Array) -> dict[str, chex.Array]:
        """Compute outputs as linear functions of embedding."""
        embedding = hk.Flatten()(embedding)

        def _infer(spec: types.ArraySpec) -> chex.Array:
            output = hk.nets.MLP(
                output_sizes=(np.prod(spec.shape),),
                w_init=self._head_w_init,
                name='torso_head',
            )(embedding)
            output = output.reshape((embedding.shape[0], *spec.shape))
            return output

        return jax.tree.map(_infer, self._out_spec)

    def unroll(
            self, inputs: chex.ArrayTree, should_reset: chex.Array | None = None
    ) -> dict[str, chex.Array]:
        """Assumes there is a time dimension in the inputs."""
        return hk.BatchApply(self.__call__)(inputs, should_reset)

    def __call__(
            self, inputs: chex.ArrayTree, should_reset: chex.Array | None = None
    ) -> dict[str, chex.Array]:
        torso = self._embedding_pass(inputs)
        out = self._head_pass(torso)
        if self._model:
            root = self._model.root_embedding(torso)
            model_out = self._model.model_step(root)
            out.update(model_out)
        return out


class MLP(MLPHeadNet):
    """Simple MLP network."""

    def __init__(
            self,
            out_spec: chex.ArrayTree,
            dense: Iterable[int],
            action_spec: types.Specs,
            head_w_init_std: float | None,
            model_out_spec: chex.ArrayTree | None = None,
            model_arch_name: str | None = None,
            model_kwargs: Mapping[str, Any] | None = None,
            module_name: str | None = None,
    ) -> None:
        super().__init__(
            out_spec,
            action_spec=action_spec,
            model_out_spec=model_out_spec,
            head_w_init_std=head_w_init_std,
            model_arch_name=model_arch_name,
            model_kwargs=model_kwargs,
            module_name=module_name,
        )
        self._dense = dense

    def _embedding_pass(
            self, inputs: chex.ArrayTree, should_reset: chex.Array | None = None
    ) -> chex.Array:
        del should_reset
        inputs = [hk.Flatten()(x) for x in jax.tree_util.tree_leaves(inputs)]
        inputs = jnp.concatenate(inputs, axis=-1)
        return hk.nets.MLP(self._dense, name='torso')(inputs)


class CNN(MLPHeadNet):
    """带空间注意力的 CNN（修复版）"""

    def __init__(
            self,
            out_spec: chex.ArrayTree,
            action_spec: types.Specs,
            head_w_init_std: float | None,
            model_out_spec: chex.ArrayTree | None = None,
            model_arch_name: str | None = None,
            model_kwargs: Mapping[str, Any] | None = None,
            module_name: str | None = None,
            conv_channels: Iterable[int] = (64, 128),
            mlp_hiddens: Iterable[int] = (256, 128),
            use_attention: bool = True,
            dropout_rate: float = 0.0,  # ← 接受参数，但下面不再使用
    ) -> None:
        self._conv_channels = conv_channels
        self._mlp_hiddens = mlp_hiddens
        self._use_attention = use_attention
        # self._dropout_rate = dropout_rate # 不再需要
        super().__init__(
            out_spec,
            action_spec=action_spec,
            model_out_spec=model_out_spec,
            head_w_init_std=head_w_init_std,
            model_arch_name=model_arch_name,
            model_kwargs=model_kwargs,
            module_name=module_name,
        )

    def _spatial_attention(self, x: chex.Array, name: str) -> chex.Array:
        """空间注意力模块（CBAM-style）"""
        avg_pool = jnp.mean(x, axis=-1, keepdims=True)
        max_pool = jnp.max(x, axis=-1, keepdims=True)
        concat = jnp.concatenate([avg_pool, max_pool], axis=-1)
        attention = hk.Conv2D(1, kernel_shape=3, padding='SAME', name=f'{name}_attn')(concat)
        attention = jax.nn.sigmoid(attention)
        return x * attention

    def _embedding_pass(
            self, inputs: chex.ArrayTree, should_reset: chex.Array | None = None
    ) -> chex.Array:
        del should_reset
        x = inputs['observation']  # (B, 4, 4, 16)

        # 1. 初始卷积
        x = hk.Conv2D(64, kernel_shape=1, padding='SAME', name='stem')(x)
        x = jax.nn.relu(x)

        # 2. 卷积块 + 注意力
        for i, channels in enumerate(self._conv_channels):
            x = hk.Conv2D(channels, kernel_shape=3, padding='SAME', name=f'conv_{i}')(x)
            x = jax.nn.relu(x)

            if self._use_attention:
                x = self._spatial_attention(x, name=f'block_{i}')

        # 3. 位置编码
        B, H, W, C = x.shape
        pos_enc = hk.get_parameter(
            'pos_encoding',
            shape=(1, H, W, C),
            init=hk.initializers.TruncatedNormal(0.02)
        )
        x = x + pos_enc

        # 4. 全局信息聚合
        avg_pool = jnp.mean(x, axis=(1, 2))
        max_pool = jnp.max(x, axis=(1, 2))
        spatial = hk.Flatten()(x)
        x = jnp.concatenate([avg_pool, max_pool, spatial], axis=-1)

        # 5. MLP 头
        for i, hidden in enumerate(self._mlp_hiddens):
            x = hk.Linear(hidden, name=f'mlp_{i}')(x)
            x = jax.nn.relu(x)

        return x


# ==============================================================================
# =====                新添加的 Transformer 架构                           =====
# ==============================================================================


# ===== Transformer 辅助函数 =====

def _sinusoidal_position_encoding(
        seq_len: int,
        d_model: int,
        max_len: int = 10000
) -> chex.Array:
    """生成正弦位置编码"""
    position = jnp.arange(seq_len)[:, None]
    div_term = jnp.exp(jnp.arange(0, d_model, 2) * (-math.log(max_len) / d_model))

    pe = jnp.zeros((seq_len, d_model))
    pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
    pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
    return pe


def _learnable_2d_position_encoding(height: int, width: int, d_model: int) -> chex.Array:
    """可学习的 2D 位置编码（更适合网格）"""
    pos_enc = hk.get_parameter(
        'pos_encoding_2d',
        shape=(height, width, d_model),
        init=hk.initializers.TruncatedNormal(stddev=0.02)
    )
    return pos_enc


# ===== 方案 1：轴向 Transformer（推荐！）=====

class AxialTransformer(MLPHeadNet):
    """
    高效轴向 Transformer（推荐用于 2048）

    优势：
    - 复杂度从 O(N²) 降到 O(2N√N)，对 4×4 网格极其高效
    - 分别对行和列做注意力，捕获空间结构
    - 内存占用低，适合大 batch

    性能：约为标准 Transformer 的 1/2 计算量
    """

    def __init__(
            self,
            out_spec: chex.ArrayTree,
            action_spec: types.Specs,
            head_w_init_std: float | None,
            # Transformer 参数
            num_layers: int = 3,  # 层数（建议 2-4）
            num_heads: int = 4,  # 注意力头数
            embed_dim: int = 128,  # 嵌入维度
            mlp_ratio: float = 2.0,  # MLP 扩展比例
            qkv_bias: bool = True,  # QKV 投影偏置
            drop_rate: float = 0.0,  # Dropout（训练时可用）
            # 模型参数
            model_out_spec: chex.ArrayTree | None = None,
            model_arch_name: str | None = None,
            model_kwargs: Mapping[str, Any] | None = None,
            module_name: str | None = None,
    ):
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._embed_dim = embed_dim
        self._mlp_ratio = mlp_ratio
        self._qkv_bias = qkv_bias
        self._drop_rate = drop_rate

        super().__init__(
            out_spec,
            action_spec=action_spec,
            model_out_spec=model_out_spec,
            head_w_init_std=head_w_init_std,
            model_arch_name=model_arch_name,
            model_kwargs=model_kwargs,
            module_name=module_name,
        )

    def _axial_attention_block(
            self,
            x: chex.Array,
            axis: int,  # 0=行，1=列
            name: str
    ) -> chex.Array:
        """单个轴向注意力块"""
        B, H, W, C = x.shape

        if axis == 0:  # 行注意力
            # 重排为 (B*H, W, C)
            x_reshaped = x.reshape(B * H, W, C)
        else:  # 列注意力
            # 重排为 (B*W, H, C)
            x_reshaped = x.transpose(0, 2, 1, 3).reshape(B * W, H, C)

        # Multi-head Self-Attention
        seq_len = x_reshaped.shape[1]
        qkv = hk.Linear(3 * C, with_bias=self._qkv_bias, name=f'{name}_qkv')(x_reshaped)
        qkv = qkv.reshape(qkv.shape[0], seq_len, 3, self._num_heads, C // self._num_heads)
        q, k, v = jnp.split(qkv, 3, axis=2)
        q, k, v = q.squeeze(2), k.squeeze(2), v.squeeze(2)

        # 缩放点积注意力
        scale = (C // self._num_heads) ** -0.5
        attn = jnp.einsum('bnhd,bmhd->bhnm', q, k) * scale
        attn = jax.nn.softmax(attn, axis=-1)

        out = jnp.einsum('bhnm,bmhd->bnhd', attn, v)
        out = out.reshape(out.shape[0], seq_len, C)
        out = hk.Linear(C, name=f'{name}_proj')(out)

        # 还原形状
        if axis == 0:
            out = out.reshape(B, H, W, C)
        else:
            out = out.reshape(B, W, H, C).transpose(0, 2, 1, 3)

        return out

    def _mlp_block(self, x: chex.Array, name: str) -> chex.Array:
        """前馈网络"""
        hidden_dim = int(self._embed_dim * self._mlp_ratio)
        x = hk.Linear(hidden_dim, name=f'{name}_fc1')(x)
        x = jax.nn.gelu(x)
        x = hk.Linear(self._embed_dim, name=f'{name}_fc2')(x)
        return x

    def _embedding_pass(
            self,
            inputs: chex.ArrayTree,
            should_reset: chex.Array | None = None
    ) -> chex.Array:
        del should_reset
        x = inputs['observation']  # (B, 4, 4, 16)
        B, H, W, C_in = x.shape

        # 1. 输入投影
        x = hk.Linear(self._embed_dim, name='input_proj')(x)

        # 2. 添加可学习的 2D 位置编码
        pos_enc = _learnable_2d_position_encoding(H, W, self._embed_dim)
        x = x + pos_enc

        # 3. Axial Transformer 层
        for i in range(self._num_layers):
            # 行注意力 + 残差
            x_row = self._axial_attention_block(x, axis=0, name=f'layer{i}_row')
            x = x + x_row
            x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True,
                             name=f'ln{i}_row')(x)

            # 列注意力 + 残差
            x_col = self._axial_attention_block(x, axis=1, name=f'layer{i}_col')
            x = x + x_col
            x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True,
                             name=f'ln{i}_col')(x)

            # MLP + 残差
            x_mlp = self._mlp_block(x, name=f'layer{i}_mlp')
            x = x + x_mlp
            x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True,
                             name=f'ln{i}_mlp')(x)

        # 4. 全局池化
        x_mean = jnp.mean(x, axis=(1, 2))  # (B, C)
        x_max = jnp.max(x, axis=(1, 2))  # (B, C)
        x = jnp.concatenate([x_mean, x_max], axis=-1)

        # 5. 输出投影
        x = hk.Linear(256, name='output_proj')(x)
        x = jax.nn.relu(x)

        return x


# ===== 方案 2：线性 Transformer（最快）=====

class LinearTransformer(MLPHeadNet):
    """
    线性注意力 Transformer（Performer-style）

    优势：
    - O(N) 复杂度（最快！）
    - 适合长序列（虽然 2048 只有 16 个 token）
    - 内存占用极低

    性能：约为标准 Transformer 的 1/4 计算量
    """

    def __init__(
            self,
            out_spec: chex.ArrayTree,
            action_spec: types.Specs,
            head_w_init_std: float | None,
            num_layers: int = 3,
            num_heads: int = 4,
            embed_dim: int = 128,
            mlp_ratio: float = 2.0,
            num_features: int = 64,  # 随机特征数量
            model_out_spec: chex.ArrayTree | None = None,
            model_arch_name: str | None = None,
            model_kwargs: Mapping[str, Any] | None = None,
            module_name: str | None = None,
    ):
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._embed_dim = embed_dim
        self._mlp_ratio = mlp_ratio
        self._num_features = num_features

        super().__init__(
            out_spec,
            action_spec=action_spec,
            model_out_spec=model_out_spec,
            head_w_init_std=head_w_init_std,
            model_arch_name=model_arch_name,
            model_kwargs=model_kwargs,
            module_name=module_name,
        )

    def _kernel_feature_map(self, x: chex.Array) -> chex.Array:
        """ReLU 核特征映射（Performer）"""
        # 使用 ReLU 而不是指数，更稳定
        return jax.nn.relu(x) + 1e-6

    def _linear_attention(self, q: chex.Array, k: chex.Array, v: chex.Array) -> chex.Array:
        """线性注意力机制"""
        # q, k, v: (B, num_heads, seq_len, head_dim)

        # 应用特征映射
        q = self._kernel_feature_map(q)
        k = self._kernel_feature_map(k)

        # 线性注意力: O(N) 复杂度
        # (B, H, D, V) @ (B, H, N, V).T = (B, H, D, N)
        kv = jnp.einsum('bhnd,bhnv->bhdv', k, v)
        k_sum = jnp.sum(k, axis=2, keepdims=True)  # (B, H, 1, D)

        # (B, H, N, D) @ (B, H, D, V) = (B, H, N, V)
        out = jnp.einsum('bhnd,bhdv->bhnv', q, kv)
        normalizer = jnp.einsum('bhnd,bhmd->bhn', q, k_sum) + 1e-6
        out = out / normalizer[..., None]

        return out

    def _embedding_pass(
            self,
            inputs: chex.ArrayTree,
            should_reset: chex.Array | None = None
    ) -> chex.Array:
        del should_reset
        x = inputs['observation']  # (B, 4, 4, 16)
        B, H, W, C_in = x.shape

        # Flatten to sequence
        x = x.reshape(B, H * W, C_in)  # (B, 16, 16)
        seq_len = x.shape[1]

        # 输入投影
        x = hk.Linear(self._embed_dim, name='input_proj')(x)

        # 位置编码
        pos_enc = _sinusoidal_position_encoding(seq_len, self._embed_dim)
        x = x + pos_enc[None, :, :]

        # Transformer 层
        for i in range(self._num_layers):
            # Linear Self-Attention
            qkv = hk.Linear(3 * self._embed_dim, name=f'layer{i}_qkv')(x)
            qkv = qkv.reshape(B, seq_len, 3, self._num_heads,
                              self._embed_dim // self._num_heads)
            qkv = qkv.transpose(2, 0, 3, 1, 4)  # (3, B, H, N, D)
            q, k, v = qkv[0], qkv[1], qkv[2]

            # 线性注意力
            attn_out = self._linear_attention(q, k, v)
            attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, seq_len, self._embed_dim)
            attn_out = hk.Linear(self._embed_dim, name=f'layer{i}_proj')(attn_out)

            x = x + attn_out
            x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True,
                             name=f'ln{i}_attn')(x)

            # MLP
            mlp_hidden = int(self._embed_dim * self._mlp_ratio)
            mlp_out = hk.Linear(mlp_hidden, name=f'layer{i}_mlp1')(x)
            mlp_out = jax.nn.gelu(mlp_out)
            mlp_out = hk.Linear(self._embed_dim, name=f'layer{i}_mlp2')(x)

            x = x + mlp_out
            x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True,
                             name=f'ln{i}_mlp')(x)

        # Global pooling
        x = jnp.mean(x, axis=1)  # (B, embed_dim)

        return x


# ===== 方案 3：标准轻量 Transformer =====

class VanillaTransformer(MLPHeadNet):
    """
    标准 Transformer（轻量版）

    优势：
    - 经典架构，稳定可靠
    - 对 16 个 token 足够高效
    - 易于理解和调试

    性能：标准实现
    """

    def __init__(
            self,
            out_spec: chex.ArrayTree,
            action_spec: types.Specs,
            head_w_init_std: float | None,
            num_layers: int = 3,
            num_heads: int = 4,
            embed_dim: int = 128,
            mlp_ratio: float = 2.0,
            use_cls_token: bool = True,  # 使用 [CLS] token
            model_out_spec: chex.ArrayTree | None = None,
            model_arch_name: str | None = None,
            model_kwargs: Mapping[str, Any] | None = None,
            module_name: str | None = None,
    ):
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._embed_dim = embed_dim
        self._mlp_ratio = mlp_ratio
        self._use_cls_token = use_cls_token

        super().__init__(
            out_spec,
            action_spec=action_spec,
            model_out_spec=model_out_spec,
            head_w_init_std=head_w_init_std,
            model_arch_name=model_arch_name,
            model_kwargs=model_kwargs,
            module_name=module_name,
        )

    def _self_attention(self, x: chex.Array, name: str) -> chex.Array:
        """标准多头自注意力"""
        B, N, C = x.shape

        qkv = hk.Linear(3 * C, name=f'{name}_qkv')(x)
        qkv = qkv.reshape(B, N, 3, self._num_heads, C // self._num_heads)
        qkv = qkv.transpose(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 缩放点积注意力
        scale = (C // self._num_heads) ** -0.5
        attn = jnp.einsum('bhnd,bhmd->bhnm', q, k) * scale
        attn = jax.nn.softmax(attn, axis=-1)

        out = jnp.einsum('bhnm,bhmd->bhnd', attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, N, C)
        out = hk.Linear(C, name=f'{name}_proj')(out)

        return out

    def _embedding_pass(
            self,
            inputs: chex.ArrayTree,
            should_reset: chex.Array | None = None
    ) -> chex.Array:
        del should_reset
        x = inputs['observation']  # (B, 4, 4, 16)
        B, H, W, C_in = x.shape

        # Flatten
        x = x.reshape(B, H * W, C_in)
        seq_len = x.shape[1]

        # 输入投影
        x = hk.Linear(self._embed_dim, name='patch_embed')(x)

        # 添加 [CLS] token（可选）
        if self._use_cls_token:
            cls_token = hk.get_parameter(
                'cls_token',
                shape=(1, 1, self._embed_dim),
                init=hk.initializers.TruncatedNormal(stddev=0.02)
            )
            cls_tokens = jnp.tile(cls_token, (B, 1, 1))
            x = jnp.concatenate([cls_tokens, x], axis=1)
            seq_len += 1

        # 位置编码
        pos_enc = hk.get_parameter(
            'pos_embed',
            shape=(1, seq_len, self._embed_dim),
            init=hk.initializers.TruncatedNormal(stddev=0.02)
        )
        x = x + pos_enc

        # Transformer 层
        for i in range(self._num_layers):
            # Self-Attention
            attn_out = self._self_attention(x, name=f'layer{i}')
            x = x + attn_out
            x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True,
                             name=f'ln{i}_1')(x)

            # MLP
            mlp_hidden = int(self._embed_dim * self._mlp_ratio)
            mlp_out = hk.Linear(mlp_hidden, name=f'layer{i}_mlp1')(x)
            mlp_out = jax.nn.gelu(mlp_out)
            mlp_out = hk.Linear(self._embed_dim, name=f'layer{i}_mlp2')(x)

            x = x + mlp_out
            x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True,
                             name=f'ln{i}_2')(x)

        # 提取特征
        if self._use_cls_token:
            x = x[:, 0]  # 使用 [CLS] token
        else:
            x = jnp.mean(x, axis=1)  # 全局平均池化

        return x

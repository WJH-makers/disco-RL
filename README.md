# 项目概览：使用 DiscoRL 训练 2048 CNN Agent

这是一个基于 Google DeepMind 的 **DiscoRL** 框架（）实现的强化学习项目，专门用于训练一个 **卷积神经网络 (CNN)** 来玩 **2048** 游戏。

项目的核心思想是利用 DiscoRL 的“元评估 (Meta-evaluation)”功能（）。我们**不**是从零开始“元训练 (Meta-training)”一个新的强化学习规则（），而是加载一个由 DiscoRL 发现的、SOTA 的“教师”规则（`Disco103`），并用它来指导一个自定义 CNN 智能体（“学生”）的学习。

## 核心组件

该项目由以下几个关键的 Python 文件组成：

---

### 1. `jittable_2048.py` - 2048 JAX 原生环境

这是 2048 游戏的核心实现，完全用 JAX 编写，使其可以被 JIT 编译和 `vmap` 批处理。

* **`_SingleStream2048`**:
    * 实现了 2048 游戏的**单局**逻辑。
    * `_slide_and_merge_one_line`: 游戏的核心算法，处理单行/单列的滑动与合并，并返回得分。
    * `step`: 接收一个动作（0-3），并根据动作调用 `_slide_and_merge_one_line` 来更新棋盘。
    * `render`: 将 `(4, 4)` 的棋盘（存储的是 2 的指数，如 1=2, 2=4, ...）转换为 `(4, 4, 16)` 的 **one-hot** 编码，作为神经网络的输入。
    * `is_terminal`: 检查游戏是否结束（没有空格且无法合并）。
* **`BatchedJittable2048Environment`**:
    * 继承自 `batched_jittable_env.py` 中的包装器。
    * **关键功能**: 增加了一个 `auto_reset: bool` 标志。这允许我们在**训练**时（`auto_reset=True`）自动重置结束的游戏，但在**评估**时（`auto_reset=False`）保持游戏结束的状态以查看最终棋盘。

---

### 2. `nets.py` - 自定义 CNN 网络

这个文件是 DiscoRL 的网络工厂（），我们对它进行了修改，**添加了对 CNN 的支持**。

* **`get_network`**: 工厂函数，通过 `name` 字符串（如 `'mlp'` 或 `'cnn'`）来构造网络。
* **`class CNN(MLPHeadNet)`**:
    * 这是我们为 2048 定制的 CNN 模型。
    * 它重写了 `_embedding_pass` 方法。
    * `_embedding_pass` 接收 `inputs['observation']`（形状为 `[B, 4, 4, 16]` 的棋盘）。
    * 它使用 `hk.Conv2D` 卷积层（例如，通道数为 `(128, 256)`）来提取特征。
    * 最后，它将卷积特征 `Flatten` 并通过一个 `hk.nets.MLP`（例如 `(256,)`）来生成最终的嵌入 (embedding)。
    * 这个嵌入随后被 `MLPHeadNet` 基类用于计算 `logits`（策略）和 `value`（价值）。

---

### 3. `train_2048.py` - 核心训练与评估脚本

这是项目的**主执行文件**。它配置并运行整个训练-评估循环。

* **Agent 配置**:
    * 它加载 DiscoRL 的 `agent.py`（）。
    * **关键**: 它将 `agent_settings.net_settings.name` 设置为 `'cnn'`，并传入 `conv_channels` 和 `mlp_hiddens` 等参数，以确保 Agent 使用我们自定义的 CNN。
* **"教师"规则**:
    * 它从 `disco_103.npz` 文件中加载预训练的 DiscoRL 规则权重。
    * 这些权重**不是**用于 CNN，而是作为 `update_rule_params` 传递给 `agent.learner_step`。
    * 这意味着我们的 CNN（学生）正在**学习模仿 Disco103 规则（教师）所产生的更新目标**。
* **训练循环 (`_sample_step` / `_learn_step`)**:
    * `_sample_step`: 调用 `unroll_jittable_actor` 来运行游戏并收集 `rollout_len`（例如 2048）步的经验。
    * `unroll_jittable_actor` 被修改为**使用 `auto_reset=False`**。
    * `_learn_step`: 接收 `actor_rollout` 数据并执行学习步骤。
    * 在 `_learn_step` 之后，脚本会**手动检查** `timestep.step_type`。如果游戏在上一个 rollout 中结束了，它会手动调用 `env.reset`。
* **对称性增强 (Symmetry Augmentation)**:
    * 这是为了解决 2048 棋盘的旋转和翻转对称性问题而添加的**关键功能**。
    * `apply_symmetries_to_rollout`:
        * 在 `_learn_step` 中被调用，在数据被送去训练**之前**。
        * 它接收一批 `actor_rollout`（形状 `[T, B, ...]`，其中 B=1）。
        * 它生成 8 个对称版本（4 次旋转 + 4 次翻转）的数据。
        * **同时转换棋盘和动作**: 它不仅使用 `jnp.rot90` 和 `jnp.flip` 转换 `observation`（`[T, B, 4, 4, 16]`），还**必须**转换对应的 `actions` 和 `logits`（例如，将“上”[0] 映射到“左”[2]）。
        * `tile_actor_state`: 将 RNN 状态 `[B, ...]` 复制 8 次，变为 `[B*8, ...]`。
        * 最终，`agent.learner_step` 接收到的是 `[T, B*8, ...]` 的批次，从而在一次梯度更新中学会所有 8 种对称性。
* **评估 (`run_evaluation`)**:
    * 训练期间会周期性（例如每 500 步）调用此函数。
    * 它使用一个**单独的评估环境** `eval_env`。
    * **关键**: 它调用环境的 `step` 函数时，**显式传递 `auto_reset=False`**，确保游戏在结束后不会立即重置。
    * **动作掩码**: 它调用 `get_valid_moves_mask` 来防止 AI 选择无效的移动。

---

### 4. `batched_jittable_env.py` - 环境批处理包装器

这是一个通用的 DiscoRL 工具文件（）。

* 它提供了一个基类 `BatchedJittableEnvironment`。
* 它的主要作用是使用 `jax.vmap` 将一个 JAX 原生的**单体**环境（如 `_SingleStream2048`）自动转换为一个**批量**环境，使其能够同时处理多个并行的游戏实例。
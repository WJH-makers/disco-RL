# 前端展示说明：仓库结构与修改优先级

## 根目录
- `train_2048.py`：核心训练/评估脚本，所有超参、混合比例、日志输出都在此；前端图表可直接消费其日志和保存的 npz 模型。
- `evaluate_2048.py`：离线推理与对局回放，可改造成 HTTP/CLI 接口供前端直播 AI 决策。
- `disco_103.npz`：教师规则权重，训练必需；不建议改动。
- `README.md`/`AGENTS.md`/`CONTRIBUTING.md`：说明与流程文档。
- `pyproject.toml`：依赖声明，前后端若需额外库在此添加。

## `disco_rl/` 核心库
- `agent.py`：Agent 生命周期与 actor/learner 状态管理；若要暴露在线推理接口或改混合策略，需改这里。
- `optimizers.py`、`utils.py`、`types.py`：通用工具/类型定义，通常保持不变。

### environments/
- `jittable_2048.py`：2048 JAX 环境与 `_slide_and_merge_one_line` 规则，想改玩法/奖励/动作掩码在此动。
- `jittable_envs.py`、`wrappers/`：批量环境与封装，扩展并行或添加统计时修改。

### networks/
- `nets.py`：CNN/MLP 工厂与嵌入层，最常调的模型结构（通道、层数、激活）。
- `action_models.py`、`meta_nets.py`：策略/元学习网络，若换 head 或加注意力结构在此改。

### update_rules/
- `disco.py`：教师（Disco）规则与损失；调整教师信号或 KL 约束改这里。
- `actor_critic.py`、`policy_gradient.py`：学生 RL 分支 (V-trace/PG)；调熵系数、价值损失在此。
- `input_transforms.py`、`weights/`：特征处理与权重存储。

### value_fns/
- `value_fn.py`、`value_utils.py`：价值函数与分布变换；除非换值函数形式一般不动。

## colabs/
- `meta_train.ipynb`、`eval.ipynb`：可导出静态图/GIF 给前端展示训练过程或评估结果，提交前清理输出。

## 最优先修改清单（从高到低）
1) 训练策略与日志：`train_2048.py`
2) 模型结构：`disco_rl/networks/nets.py`
3) 环境与奖励：`disco_rl/environments/jittable_2048.py`
4) 教师/学生损失配比：`disco_rl/update_rules/disco.py`、`actor_critic.py`
5) 推理/回放接口：`evaluate_2048.py`（改造成服务端 API 或批量推理）。

## 前端集成提示
- 对外暴露一个简易 API：`POST /infer` -> {logits, value, valid_mask}；实现可复用 `agent.actor_step`。
- 训练日志可写入 CSV/JSON（步骤、平均得分、最高 tile、loss）；前端折线图直接消费。
- 若需可视化对称增强，记录每步采样前的 8 向旋转/翻转动作映射，可在前端以小网格循环展示。

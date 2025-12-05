# Repository Guidelines & Usage Manual

## 1. 环境与依赖
- Python 3.11+，建议虚拟环境：`python -m venv .venv && .\\.venv\\Scripts\\activate` (Windows) 或 `source .venv/bin/activate` (Unix)。
- 安装项目依赖：`pip install -e .`。
- 可视化/服务依赖：`pip install tensorboardX imageio fastapi uvicorn`。
- 前端开发：`cd frontend && npm install`。

## 2. 训练入口（两种架构）
- Transformer（默认）：`python train_2048.py`
- CNN：`MODEL_KIND=cnn python train_2048.py`
- 冒烟/CI：`SANDBOX_RUN=1 TQDM_OFF=1 python train_2048.py`（步数与 batch 自动缩减，跳过预热）。
- 输出文件：
  - 权重：`hybrid_adaptive_2048_v5.1_<arch>.npz` 与 `_best.npz`
  - 日志：`logs/train_log.csv`（step/loss/λ/sps 等），`logs/eval_log.csv`（env_steps/avg_score/max_tile/artifact）
  - 评估 GIF：`artifacts/step_<env>_<arch>.gif`（首局高分辨率彩色回放）

## 3. 训练稳定性要点
- 优化：AdamW + warmup+cosine LR；全局 grad clip=1.0。
- 损失：`imitation * loss_imitation + loss_rl`，valid mask 均值；loss/grad `nan_to_num` 防 NaN/Inf。
- 模仿权重 λ：cosine 退火；奖励归一化带 EMA。
- 沙盒模式自动降低计算量。

## 4. 推理/指标服务
- 启动：`uvicorn serve_infer:app --port 8000 --reload`
- `/infer`：输入 4x4 log2 棋盘，返回 `best_move/logits/valid/value/search_depth/nodes/elapsed_ms`。
- `/metrics`：返回 `logs/train_log.csv` 与 `logs/eval_log.csv` 最新一行（含 GIF 路径）。

## 5. 前端实时看板（可选）
- 开发：`cd frontend && VITE_API_PROXY=http://localhost:8000 npm run dev`
- 生产：`npm run build`，使用 `frontend/dist/` 静态托管。
- 页面展示：动作概率/最佳箭头、搜索深度/节点数/耗时、训练指标实时轮询 `/metrics`、最新评估 GIF 内嵌。

## 6. 评估与基准
- 评估在训练中自动执行；结果在 `logs/eval_log.csv`，GIF 在 `artifacts/`。
- 零预测策略对比（蛇形 vs 贪心）：`EPISODES=2000 python policy_bench.py`。
- 冒烟自检：`python smoke_check.py`（合并/合法动作、expectimax 决策、沙盒训练）。

## 7. 一键脚本（Unix 类环境）
`./run_all.sh`（可 `MODEL_KIND=cnn SANDBOX_RUN=0` 覆盖环境）：安装依赖并直接启动训练，完成后提示如何启动服务与前端。Windows 可按脚本步骤手动执行。

## 8. 目录速览
- `train_2048.py`：核心训练/评估循环，支持架构切换、沙盒、日志、GIF。
- `serve_infer.py`：expectiminimax 推理 + `/metrics` 指标接口。
- `disco_rl/`：环境、网络、更新规则等库代码。
- `artifacts/`：训练评估 GIF 输出。
- `logs/`：训练/评估 CSV 日志。
- `frontend/`：Vite React 看板（推理可视化 + 训练指标）。
- `policy_bench.py`：零预测策略基准。
- `smoke_check.py`：冒烟测试。 

## 9. 进阶：教师（超网络）训练思路
默认教师 Disco103 是冻结的；若环境变化或想进一步提升，可开启可训练教师：
- 推荐配置（新增到 CFG 或环境变量）：
  - `train_teacher=True`
  - `teacher_lr=1e-4`
  - `teacher_update_every=5`（教师低频更新，学生每步更新）
  - `teacher_kl_reg=1e-3`（KL(student‖teacher) 正则，防漂移）
  - `teacher_grad_clip=1.0`
- 实现要点（按需在 `train_2048.py` 中改）：
  - 将 `teacher_params` 设为可训练，独立 `teacher_optimizer`（Adam/AdamW，小 LR）。
  - JIT 内：若 `update_teacher` 为 True，则对教师 loss（可用同一 imitation+RL 目标）求梯度，更新教师参数；否则直接 passthrough。
  - 低频更新：host 循环用 `step % teacher_update_every == 0` 控制；KL 正则仅在更新步生效。
  - 断点：在 ckpt 中保存 `teacher_params` 与 `teacher_opt_state`。

## 10. 在现有做法上的优化建议
- **expectimax 预热**：已默认用 expectimax 生成 1024 条高质量样本做行为克隆预热，可调 `EXPECTIMAX_WARMUP_SAMPLES/DEPTH/TIMEOUT`。
- **搜索+估值融合**：推理端可用 expectimax(3–4 ply) + 启发式 + n-tuple + 模型 value 融合叶子估值（`serve_infer.py` 已内置启发式+n-tuple，留接口加模型值）。
- **批评估与吞吐**：eval 并行多局（默认 8），训练 batch/rollout 大一些（64、256）；显存不足时用梯度累积替代。
- **稳定性**：动作用掩码、`auto_reset=False` 与教师保持一致；λ 退火放缓，ε 固定 0.05–0.1 防早收敛；grad clip、nan_to_num 已启用。
- **断点续训**：默认自动恢复 (`RESUME=1`)，重头用 `RESUME=0`；权重+状态+优化器+环境/actor 状态均存 `checkpoints/ckpt_<arch>.pkl`。

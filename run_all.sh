#!/usr/bin/env bash
set -euo pipefail

# 一键执行：安装依赖 -> 训练 -> 启动推理服务 -> 启动前端
# 适用于类 Unix 环境；Windows 可参考命令逐步执行。

# 1) 安装依赖
python -m pip install --upgrade pip
python -m pip install -e . tensorboardX imageio fastapi uvicorn

# 2) 训练（可修改 MODEL_KIND=cnn / SANDBOX_RUN=1）
MODEL_KIND=${MODEL_KIND:-axial}
SANDBOX_RUN=${SANDBOX_RUN:-0}
TQDM_OFF=${TQDM_OFF:-0}
echo "== Training with MODEL_KIND=$MODEL_KIND SANDBOX_RUN=$SANDBOX_RUN =="
MODEL_KIND=$MODEL_KIND SANDBOX_RUN=$SANDBOX_RUN TQDM_OFF=$TQDM_OFF python train_2048.py

echo "训练完成，日志在 logs/ ，权重在 hybrid_adaptive_2048_v5.1_${MODEL_KIND}*.npz"
echo "如需启动推理服务：uvicorn serve_infer:app --port 8000 --reload"
echo "前端开发：cd frontend && npm install && VITE_API_PROXY=http://localhost:8000 npm run dev"

"""
FastAPI 推理服务：为前端和脚本提供 2048 决策接口。

特性：
- 纯 Python expectiminimax（迭代加深）+ 启发式估值，默认无需 GPU/JAX。
- 可选加载学生 CNN 估值：实现 `cnn_value(board)` 即可替换 `heuristic_value`。
- 返回 best_move、logits（4 个方向的期望值）、有效动作、搜索深度、展开节点数。

运行：
    pip install fastapi uvicorn
    uvicorn serve_infer:app --reload --port 8000

接口：
POST /infer { board: int[4][4] (log2 tiles, 0=空) }
返回 { best_move, logits[4], valid[4], value, search_depth, nodes }
"""

from __future__ import annotations

import math
import time
from typing import List, Tuple

import numpy as np
import csv
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="2048 RL Inference", version="0.2.0")


# ---------------------------- 数据模型 ---------------------------- #
class InferRequest(BaseModel):
    board: List[List[int]]  # 4x4 log2 board


# ---------------------------- 2048 基本操作 ---------------------------- #
def slide_and_merge(line: List[int]) -> Tuple[List[int], int]:
    """合并一行，返回新行和得分（得分用真实值 2^tile 累加）。"""
    tiles = [v for v in line if v != 0]
    merged, score, i = [], 0, 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            v = tiles[i] + 1
            merged.append(v)
            score += 2**v
            i += 2
        else:
            merged.append(tiles[i])
            i += 1
    merged += [0] * (len(line) - len(merged))
    return merged, score


def move(board: List[List[int]], action: int) -> Tuple[List[List[int]], int]:
    """执行动作 0上 1右 2下 3左，返回新棋盘及本步得分。"""
    b = [row[:] for row in board]
    score = 0
    if action in (0, 2):  # 列操作
        for c in range(4):
            col = [b[r][c] for r in range(4)]
            if action == 2:
                col.reverse()
            merged, s = slide_and_merge(col)
            if action == 2:
                merged.reverse()
            for r in range(4):
                b[r][c] = merged[r]
            score += s
    else:  # 行操作
        for r in range(4):
            row = b[r][:]
            if action == 1:
                row.reverse()
            merged, s = slide_and_merge(row)
            if action == 1:
                merged.reverse()
            b[r] = merged
            score += s
    return b, score


def valid_moves(board: List[List[int]]) -> List[bool]:
    return [move(board, a)[0] != board for a in range(4)]


def empty_cells(board: List[List[int]]):
    for r in range(4):
        for c in range(4):
            if board[r][c] == 0:
                yield r, c


# ---------------------------- 估值函数 ---------------------------- #
def heuristic_value(board: List[List[int]]) -> float:
    """手写特征估值，越大越好。"""
    b = np.array(board, dtype=np.int32)
    empties = np.count_nonzero(b == 0)

    # 单调性：行列有序程度（取负差的和）
    def mono(arr):
        return -np.sum(np.abs(np.diff(arr)))

    mono_score = mono(b).sum() + mono(b.T).sum()

    # 平滑度：相邻差的负和
    smooth = -(
        np.abs(np.diff(b, axis=0)).sum() + np.abs(np.diff(b, axis=1)).sum()
    )

    # 角落奖励：最大 tile 在四角之一
    max_tile = b.max()
    corners = [b[0, 0], b[0, 3], b[3, 0], b[3, 3]]
    corner_bonus = 10.0 if max_tile in corners else 0.0

    # 合并潜力：同值邻居数量
    merge_potential = (
        np.sum(b[:, :-1] == b[:, 1:]) + np.sum(b[:-1, :] == b[1:, :])
    )

    return (
        2.7 * empties
        + 1.0 * mono_score
        + 0.1 * smooth
        + 3.0 * corner_bonus
        + 0.5 * merge_potential
    )


# 如需接入 CNN，请实现 cnn_value(board) 并在 evaluate() 中调用
def evaluate(board: List[List[int]]) -> float:
    return heuristic_value(board)


# ---------------------------- Expectimax 搜索 ---------------------------- #
def expectimax(
    board: List[List[int]],
    depth: int,
    cache: dict,
    timeout_ms: int,
    start_t: float,
) -> float:
    """返回该局面的期望值（根是“你”）。"""
    key = (tuple(np.ravel(board)), depth, "M")
    if key in cache:
        return cache[key]
    if depth == 0 or time.perf_counter() - start_t > timeout_ms / 1000:
        v = evaluate(board)
        cache[key] = v
        return v

    best = -1e9
    valid = valid_moves(board)
    if not any(valid):
        return evaluate(board)
    for a in range(4):
        if not valid[a]:
            continue
        child, _ = move(board, a)
        val = chance_node(child, depth - 1, cache, timeout_ms, start_t)
        if val > best:
            best = val
    cache[key] = best
    return best


def chance_node(
    board: List[List[int]],
    depth: int,
    cache: dict,
    timeout_ms: int,
    start_t: float,
) -> float:
    key = (tuple(np.ravel(board)), depth, "C")
    if key in cache:
        return cache[key]
    empties = list(empty_cells(board))
    if not empties:
        return evaluate(board)
    acc = 0.0
    for r, c in empties:
        for tile, p in ((1, 0.9), (2, 0.1)):  # 1=2, 2=4
            b2 = [row[:] for row in board]
            b2[r][c] = tile
            acc += p * expectimax(b2, depth, cache, timeout_ms, start_t)
    v = acc / len(empties)
    cache[key] = v
    return v


def decide(
    board: List[List[int]],
    base_depth: int = 4,
    timeout_ms: int = 60,
) -> dict:
    """主入口：返回 best_move、logits、元信息。"""
    empties = sum(v == 0 for row in board for v in row)
    depth = base_depth
    if empties >= 6:
        depth = base_depth
    elif empties >= 3:
        depth = base_depth + 1
    else:
        depth = base_depth + 2

    start_t = time.perf_counter()
    cache = {}
    valid = valid_moves(board)
    logits = [-1e9] * 4
    for a in range(4):
        if not valid[a]:
            continue
        child, _ = move(board, a)
        logits[a] = chance_node(child, depth - 1, cache, timeout_ms, start_t)
        if time.perf_counter() - start_t > timeout_ms / 1000:
            break

    best_move = int(np.argmax(logits))
    root_value = max(logits)
    return {
        "best_move": best_move,
        "logits": logits,
        "valid": valid,
        "value": float(root_value),
        "search_depth": depth,
        "nodes": len(cache),
        "elapsed_ms": int((time.perf_counter() - start_t) * 1000),
    }


# ---------------------------- API ---------------------------- #
@app.post("/infer")
def infer(req: InferRequest):
    info = decide(req.board)
    return info


@app.get("/metrics")
def metrics():
    """返回最新的训练/评估指标，供前端实时轮询。"""
    train_csv = Path("logs/train_log.csv")
    eval_csv = Path("logs/eval_log.csv")
    train_row = {}
    eval_row = {}
    if train_csv.exists():
        try:
            with train_csv.open() as f:
                rows = list(csv.DictReader(f))
                if rows:
                    train_row = rows[-1]
        except Exception:
            pass
    if eval_csv.exists():
        try:
            with eval_csv.open() as f:
                rows = list(csv.DictReader(f))
                if rows:
                    eval_row = rows[-1]
        except Exception:
            pass
    return {"train": train_row, "eval": eval_row}

"""
离线数据生成脚本：用 expectimax 产生高质量 (board, action, value) 样本，供行为克隆/预训练。

运行示例：
    # 默认 50k 样本，存到 data/expectimax_50k.npz
    python generate_expectimax_data.py

参数（环境变量或命令行）：
    NUM_EPISODES   生成多少局（默认 2000）
    MAX_STEPS      每局最多步数（默认 2000）
    OUT            输出路径（默认 data/expectimax_data.npz）
    DEPTH          expectimax 深度（默认 3）

输出：
    npz 包含：
        boards : (N, 4, 4)  int8  (log2)
        actions: (N,)       int8  (0:上 1:右 2:下 3:左)
        values : (N,)       float32  叶子估值（期望分）
"""

import os
import math
import numpy as np
from pathlib import Path
from tqdm import tqdm

from serve_infer import expectimax_decide  # 复用后端搜索


def spawn(board):
    empties = np.argwhere(board == 0)
    if len(empties) == 0:
        return board
    r, c = empties[np.random.randint(len(empties))]
    board[r, c] = 1 if np.random.rand() < 0.9 else 2
    return board


def move(board, a):
    b = board.copy()
    score = 0
    if a in (0, 2):
        for c in range(4):
            col = b[:, c]
            if a == 2:
                col = col[::-1]
            merged, s = slide(col)
            if a == 2:
                merged = merged[::-1]
            b[:, c] = merged
            score += s
    else:
        for r in range(4):
            row = b[r]
            if a == 1:
                row = row[::-1]
            merged, s = slide(row)
            if a == 1:
                merged = merged[::-1]
            b[r] = merged
            score += s
    return b, score


def slide(line):
    tiles = [v for v in line if v]
    out, score, i = [], 0, 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            v = tiles[i] + 1
            out.append(v)
            score += 2**v
            i += 2
        else:
            out.append(tiles[i])
            i += 1
    out += [0] * (4 - len(out))
    return np.array(out, dtype=np.int8), score


def generate(num_episodes, max_steps, depth):
    boards = []
    actions = []
    values = []
    for _ in tqdm(range(num_episodes), desc="expectimax episodes"):
        b = np.zeros((4, 4), dtype=np.int8)
        spawn(spawn(b))
        for step in range(max_steps):
            info = expectimax_decide(b.tolist(), base_depth=depth, timeout_ms=50)
            a = info["best_move"]
            boards.append(b.copy())
            actions.append(a)
            values.append(info["value"])
            nb, _ = move(b, a)
            if np.array_equal(nb, b):
                break
            b = spawn(nb)
    return (
        np.stack(boards, axis=0),
        np.array(actions, dtype=np.int8),
        np.array(values, dtype=np.float32),
    )


def main():
    num_episodes = int(os.environ.get("NUM_EPISODES", "2000"))
    max_steps = int(os.environ.get("MAX_STEPS", "2000"))
    out_path = Path(os.environ.get("OUT", "data/expectimax_data.npz"))
    depth = int(os.environ.get("DEPTH", "3"))
    out_path.parent.mkdir(exist_ok=True, parents=True)
    boards, acts, vals = generate(num_episodes, max_steps, depth)
    np.savez(out_path, boards=boards, actions=acts, values=vals)
    print(f"✓ saved {len(boards)} samples to {out_path}")


if __name__ == "__main__":
    main()

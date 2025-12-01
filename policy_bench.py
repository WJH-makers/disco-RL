"""
零预测策略基准：对比“蛇形走位”与简单贪心策略在 2048 的表现。

运行：
    python policy_bench.py            # 默认 2,000 局
    EPISODES=5000 python policy_bench.py

输出：
    平均分、32768 出现率、最大块分布（百分位）。

策略说明：
    - snake_policy: 固定蛇形动作循环（无需估值/前瞻）
    - greedy_policy: 执行动作后空格最多的方向（平手取先出现）

可作为“蛇形是否最优”的经验检验；若贪心或其他策略在统计上更好，
即可构造反例证明蛇形非最优。
"""

import os
import random
import math
from statistics import mean

EPISODES = int(os.environ.get("EPISODES", "2000"))


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
    return out, score


def move(board, a):
    b = [row[:] for row in board]
    score = 0
    if a in (0, 2):  # up/down
        for c in range(4):
            col = [b[r][c] for r in range(4)]
            if a == 2:
                col.reverse()
            merged, s = slide(col)
            if a == 2:
                merged.reverse()
            for r in range(4):
                b[r][c] = merged[r]
            score += s
    else:  # left/right
        for r in range(4):
            row = b[r][:]
            if a == 1:
                row.reverse()
            merged, s = slide(row)
            if a == 1:
                merged.reverse()
            b[r] = merged
            score += s
    return b, score


def spawn(b):
    empties = [(r, c) for r in range(4) for c in range(4) if b[r][c] == 0]
    if not empties:
        return b
    r, c = random.choice(empties)
    b[r][c] = 1 if random.random() < 0.9 else 2
    return b


snake_cycle = [0, 3, 2, 2, 1, 3, 3, 2]  # 0↑ 1→ 2↓ 3←


def snake_policy(step):
    return snake_cycle[step % len(snake_cycle)]


def greedy_policy(board):
    best, best_empty = None, -1
    for a in range(4):
        nb, _ = move(board, a)
        if nb == board:
            continue
        empty = sum(v == 0 for r in nb for v in r)
        if empty > best_empty:
            best_empty = empty
            best = a
    return best if best is not None else 0


def play(policy_fn, episodes=1000):
    scores = []
    max_tile = []
    for ep in range(episodes):
        b = [[0] * 4 for _ in range(4)]
        spawn(spawn(b))
        step = 0
        score = 0
        while True:
            a = policy_fn(b, step) if policy_fn.__code__.co_argcount == 2 else policy_fn(b)
            nb, s = move(b, a)
            if nb == b:
                break
            b = spawn(nb)
            score += s
            step += 1
        scores.append(score)
        max_tile.append(max(v for r in b for v in r))
    return scores, max_tile


def pct(xs, p):
    if not xs:
        return 0
    xs = sorted(xs)
    k = int(len(xs) * p)
    return xs[min(k, len(xs) - 1)]


def summarize(name, scores, tiles):
    print(f"\n{name}")
    print(f"  平均分: {mean(scores):.1f}")
    print(f"  90/99 百分位分数: {pct(scores,0.9):.0f} / {pct(scores,0.99):.0f}")
    print(f"  最大块 32768 率: {sum(t>=15 for t in tiles)/len(tiles):.3f}")
    print(f"  最大块 65536 率: {sum(t>=16 for t in tiles)/len(tiles):.3f}")
    print(f"  最大块中位数 (log2): {pct(tiles,0.5)}")


def main():
    random.seed(42)
    print(f"Running {EPISODES} episodes per policy ...")
    snake_scores, snake_tiles = play(lambda b, k: snake_policy(k), episodes=EPISODES)
    greedy_scores, greedy_tiles = play(greedy_policy, episodes=EPISODES)
    summarize("蛇形策略", snake_scores, snake_tiles)
    summarize("贪心策略", greedy_scores, greedy_tiles)


if __name__ == "__main__":
    main()

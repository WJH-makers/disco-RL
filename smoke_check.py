"""
轻量冒烟测试，验证核心算法可跑通且数值正常。
运行: python smoke_check.py
"""

import json
import subprocess
import sys
from pathlib import Path

import serve_infer


def test_move_and_valid():
    board = [
        [1, 1, 0, 0],
        [2, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    vm = serve_infer.valid_moves(board)
    assert vm == [True, True, True, True], "basic board should allow all moves"
    moved, sc = serve_infer.move(board, 0)
    assert sc == 4, "merge 2+2 should give 4 points"
    assert moved[0][0] == 2, "merged tile should be log2=2"


def test_expectimax_decide():
    board = [
        [2, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    info = serve_infer.decide(board, base_depth=2, timeout_ms=50)
    assert info["best_move"] in [0, 1, 2, 3]
    assert len(info["logits"]) == 4
    assert any(info["valid"]), "at least one move should be valid"


def test_train_sandbox():
    # 只验证脚本入口可调用并退出（SANDBOX_RUN=1 极小步数）
    env = {"SANDBOX_RUN": "1", "TQDM_OFF": "1", **dict(**{k: v for k, v in dict(Path.cwd().env if False else {}).items()})}
    proc = subprocess.run(
        [sys.executable, "train_2048.py"],
        env={**env, **dict(**dict())},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
        text=True,
    )
    if proc.returncode != 0:
        print(proc.stdout)
    assert proc.returncode == 0, "train_2048.py should finish sandbox run"


def main():
    test_move_and_valid()
    test_expectimax_decide()
    print("✓ move/valid/expectimax ok")
    try:
        test_train_sandbox()
        print("✓ sandbox training run ok")
    except Exception as e:
        print("sandbox train failed:", e)
        # 不强制失败，方便无 GPU 环境


if __name__ == "__main__":
    main()

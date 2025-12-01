import { useEffect, useMemo, useState } from "react";

type InferResp = {
  logits: number[];
  value: number;
  valid: boolean[];
  best_move: number;
  search_depth?: number;
  nodes?: number;
  elapsed_ms?: number;
};

type MetricsResp = {
  train?: { [k: string]: string };
  eval?: { [k: string]: string };
};

const arrow = ["↑", "→", "↓", "←"];
const colors = [
  "#cdc1b4",
  "#eee4da",
  "#ede0c8",
  "#f2b179",
  "#f59563",
  "#f67c5f",
  "#f65e3b",
  "#edcf72",
  "#edc850",
  "#edc53f",
  "#edc22e"
];

const emptyBoard = () =>
  Array.from({ length: 4 }, () => Array.from({ length: 4 }, () => 0));

function Tile({ v }: { v: number }) {
  const val = v ? 2 ** v : 0;
  const idx = Math.min(v, colors.length - 1);
  return (
    <div className="tile" style={{ background: colors[idx] }}>
      {val || ""}
    </div>
  );
}

function softmax(logits: number[]) {
  const max = Math.max(...logits);
  const exps = logits.map((x) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / (sum || 1));
}

export default function App() {
  const [board, setBoard] = useState<number[][]>(emptyBoard);
  const [probs, setProbs] = useState<number[]>([0, 0, 0, 0]);
  const [valid, setValid] = useState<boolean[]>([true, true, true, true]);
  const [value, setValue] = useState(0);
  const [best, setBest] = useState<number | null>(null);
  const [depth, setDepth] = useState<number | null>(null);
  const [nodes, setNodes] = useState<number | null>(null);
  const [latency, setLatency] = useState<number | null>(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<MetricsResp>({});
  const artifact = metrics.eval?.artifact;

  const tiles = useMemo(() => board.flatMap((r, ri) => r.map((v, ci) => ({ v, key: `${ri}-${ci}` }))), [board]);

  const infer = async (nextBoard: number[][] = board) => {
    setPending(true);
    setError(null);
    try {
      const res = await fetch("/infer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ board: nextBoard })
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: InferResp = await res.json();
      setProbs(softmax(data.logits));
      setValid(data.valid);
      setValue(data.value);
      setBest(data.best_move);
      setDepth(data.search_depth ?? null);
      setNodes(data.nodes ?? null);
      setLatency(data.elapsed_ms ?? null);
    } catch (e: any) {
      setError(e?.message ?? "infer failed");
    } finally {
      setPending(false);
    }
  };

  useEffect(() => {
    infer();
  }, []);

  // 轮询训练指标
  useEffect(() => {
    const timer = setInterval(async () => {
      try {
        const res = await fetch("/metrics");
        if (!res.ok) return;
        const data: MetricsResp = await res.json();
        setMetrics(data);
      } catch {
        /* ignore */
      }
    }, 2000);
    return () => clearInterval(timer);
  }, []);

  return (
    <div className="layout">
      <div className="column">
        <h1>2048 CNN + DiscoRL</h1>
        <div className="board">
          {tiles.map((t) => (
            <Tile key={t.key} v={t.v} />
          ))}
        </div>
        <div className="controls">
          <button onClick={() => setBoard(emptyBoard())}>清空棋盘</button>
          <button onClick={() => infer()}>刷新推理</button>
          {pending && <span className="muted">推理中...</span>}
          {error && <span className="error">{error}</span>}
        </div>
      </div>

      <div className="panel">
        <h3>动作概率 (掩码灰显)</h3>
        {arrow.map((a, i) => (
          <div key={i} className="bar">
            <span className={`arrow ${best === i ? "best" : ""}`}>{a}</span>
            <div className="meter-wrap">
              <div
                className="meter"
                style={{
                  width: `${(probs[i] || 0) * 100}%`,
                  background: valid[i] ? "#6aa84f" : "#9e9e9e"
                }}
              />
            </div>
            <span className="pct">{((probs[i] || 0) * 100).toFixed(1)}%</span>
          </div>
        ))}
        <div className="value">Value 估计: {value.toFixed(2)}</div>
        <div className="meta">
          {depth !== null && <span>深度 d={depth}</span>}
          {nodes !== null && <span>节点 {nodes}</span>}
          {latency !== null && <span>{latency} ms</span>}
        </div>
        <div className="hint">
          后端需暴露 <code>POST /infer</code>，body: {'{ board: int[4][4] (log2) }'}
          ，返回 logits/value/valid。
        </div>
      </div>

      <div className="panel">
        <h3>训练指标 (实时)</h3>
        <div className="meta">
          <span>step: {metrics.train?.step ?? "-"}</span>
          <span>loss: {metrics.train?.loss ?? "-"}</span>
          <span>λ: {metrics.train?.lambda ?? "-"}</span>
          <span>sps: {metrics.train?.sps ?? "-"}</span>
        </div>
        <div className="meta">
          <span>eval avg: {metrics.eval?.avg_score ?? "-"}</span>
          <span>max tile: {metrics.eval?.max_tile ?? "-"}</span>
        </div>
        <div className="hint">从 /metrics 每 2 秒拉取；源于 logs/*.csv。</div>
        {artifact ? (
          <div className="artifact">
            <div className="meta">最新评估 GIF</div>
            <img src={`/${artifact}`} alt="eval gif" className="gif" />
          </div>
        ) : (
          <div className="hint">暂无评估 GIF</div>
        )}
      </div>
    </div>
  );
}

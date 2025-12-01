# 2048 RL 前端演示

快速启动：
1. 安装依赖：`cd frontend && npm install`
2. 启动开发：`npm run dev`（默认连接同源的后端 `/infer` 接口；如需代理，设置环境变量 `VITE_API_PROXY=http://localhost:8000`）
3. 生产构建：`npm run build`，产物在 `dist/`。

接口契约：
```
POST /infer
Body: { "board": int[4][4] }  # 2048 棋盘，存 log2 瓦片值，0 表示空
Response: { "logits": number[4], "value": number, "valid": boolean[4] }
```

主要文件：
- `src/App.tsx`：棋盘渲染 + 动作概率条 + /infer 调用逻辑。
- `src/index.css`：配色与布局。
- `vite.config.ts`：可选代理配置，方便本地前后端分离开发。

# 求职雷达 Agent — 智能求职助手

基于 FastAPI + React 的智能求职对话系统，支持简历匹配分析、岗位探索、面试准备、属性核实等意图。采用 v2 ReAct Agent 架构（LLMPlanner + ReActExecutor），支持本地模型（Ollama/vLLM/LM Studio）和云端 API（DashScope/OpenAI 兼容）。

## 功能特性

- **多意图识别**：explore（岗位探索）、assess（匹配评估）、verify（属性核实）、prepare（面试准备）、chat（通用对话）
- **混合召回检索**：Chroma 向量库（70%）+ BM25（30%）+ CrossEncoder 重排序
- **ReAct 执行器**：动态规划 DAG 任务图，支持 replan 和降级策略
- **本地模型兼容**：Ollama / vLLM / LM Studio，API_KEY 可留空
- **流式输出**：SSE 流式响应，支持首 Token 延迟测量
- **完整评测体系**：白盒组件级评测 + 黑盒 HTTP 端到端评测 + LLM-as-judge 质量评估

## 技术栈

| 层级 | 技术 |
|------|------|
| 后端 | FastAPI, Uvicorn, Pydantic v2 |
| LLM | OpenAI 兼容（DashScope / OpenAI / 本地 Ollama） |
| 向量库 | ChromaDB |
| 检索 | 向量召回 + BM25 + CrossEncoder 重排序 |
| 前端 | React 19 + Vite + TailwindCSS |
| 评测 | 自定义白盒/黑盒脚本 + LLM-as-judge |

## 项目结构

```
├── backend/
│   ├── app/
│   │   ├── core/           # LLMClient、Embedding、向量库、检索、意图识别、Planner、ReActExecutor
│   │   ├── routers/        # FastAPI 路由（chat、resumes、KB、settings）
│   │   └── main.py         # 应用入口
│   ├── data/
│   │   ├── jds.json        # 23 条 JD 结构化数据
│   │   ├── resumes.json    # 10 条简历数据
│   │   ├── chroma_db/      # Chroma 向量库（202 chunks，已持久化）
│   │   └── models/         # CrossEncoder 重排序模型（运行时自动下载，不提交）
│   ├── eval/
│   │   ├── test_dataset.jsonl   # 55 条评测用例（含 4 组多轮 session_group）
│   │   ├── test_resumes.json    # 测试用例对应的简历映射
│   │   ├── run_v2_eval.py       # 白盒组件级评测（LLM-as-judge）
│   │   └── run_http_eval.py     # 黑盒 HTTP 端到端延迟评测
│   ├── venv/               # Python 虚拟环境（不提交）
│   ├── .env                # 环境变量（不提交，复制 .env.example）
│   ├── .env.example        # 环境变量模板
│   └── requirements.txt
├── frontend/               # Next.js 前端（若使用）
├── src/                    # Vite + React 前端（当前主前端）
│   ├── api/
│   ├── App.jsx
│   └── main.jsx
├── package.json
├── vite.config.js
└── README.md
```

## 快速开始

### 1. 克隆仓库

```bash
git clone <仓库地址>
cd job_search
```

### 2. 后端环境搭建

```bash
cd backend

# 创建虚拟环境
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 复制环境变量模板并编辑
cp .env.example .env
# 编辑 .env 填入你的 API_KEY（本地模型可留空）
```

### 3. 下载重排序模型（首次运行）

CrossEncoder 模型（BAAI/bge-reranker-base）会在首次启动时自动下载到 `backend/data/models/`。该目录已被 `.gitignore` 排除（模型文件约 3GB）。

### 4. 启动后端

```bash
# 从 backend 目录
python -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

服务启动后会自动加载：
- 23 条 JD 数据（`data/jds.json`）
- 10 条简历（`data/resumes.json`）
- Chroma 向量库（`data/chroma_db/`，202 chunks）

### 5. 前端环境搭建

```bash
# 回到项目根目录
cd ..

# 安装依赖
npm install

# 启动开发服务器
npm run dev
```

前端默认运行在 `http://localhost:5173`，后端 API 在 `http://localhost:8001`。

## 环境变量配置

复制 `backend/.env.example` 为 `backend/.env`，按需填写：

| 变量 | 说明 | 示例 |
|------|------|------|
| `CHAT_BASE_URL` | LLM API 地址 | `https://api.openai.com/v1` 或 `http://localhost:11434/v1` |
| `CHAT_API_KEY` | API 密钥 | `sk-xxx`（本地模型可留空） |
| `CHAT_MODEL` | 模型名 | `gpt-4o` 或 `qwen2.5:14b` |
| `JUDGE_*` | 评测专用小模型 | 可复用 MEMORY 配置，或独立指定 |

**本地模型（Ollama）示例**：
```env
CHAT_BASE_URL=http://localhost:11434/v1
CHAT_API_KEY=
CHAT_MODEL=qwen2.5:14b
```

## 测试指南

### 白盒组件级评测（`run_v2_eval.py`）

直接调用内部组件，评估意图识别、工具选择、LLM-as-judge 质量等。

```bash
cd backend

# 跑全部 55 条用例
python eval/run_v2_eval.py

# 跑单条调试
python eval/run_v2_eval.py --case eval_chen_01

# 跑指定批次
python eval/run_v2_eval.py --batch chen,li

# 稳定性测试（每条跑 3 次）
python eval/run_v2_eval.py --stability 3
```

报告默认保存到 `backend/eval/v2_eval_report_<timestamp>.json`。

### 黑盒 HTTP 端到端评测（`run_http_eval.py`）

模拟真实前端调用，测量用户感知延迟和前端 Schema 一致性。

```bash
cd backend

# 跑全部用例（自动启动/停止后端服务）
python eval/run_http_eval.py

# 包含 SSE 流式测试
python eval/run_http_eval.py --stream

# 单条调试（假设后端已在 8001 端口运行）
python eval/run_http_eval.py --case eval_chen_02 --no-start-server
```

报告保存到 `backend/eval/http_eval_report_<timestamp>.json`。

### 评测指标说明

| 指标 | 来源 | 说明 |
|------|------|------|
| `task_success` | LLM-as-judge | 小模型评估最终回复是否解决用户问题 |
| `tool_selection_f1` | 规则比对 | 实际调用工具 vs 预期工具的 F1 |
| `intent_f1` | 规则比对 | 意图识别的精确率/召回率/F1 |
| `schema_valid` | 黑盒 HTTP | 响应 JSON 是否符合前端 ChatResponse 接口 |

## 注意事项

1. **`.env` 文件不提交**：包含 API 密钥，已加入 `.gitignore`。新设备需手动复制 `.env.example` 并填写。
2. **`data/models/` 不提交**：CrossEncoder 模型约 3GB，首次启动自动下载。
3. **`venv/` 不提交**：虚拟环境自行创建。
4. **Chroma 向量库已提交**：`data/chroma_db/` 已持久化并提交，拿到即可直接运行检索。如需重建，需调用 embedding API。
5. **Token 耗尽检测**：白盒评测中若触发 401/403/429 或余额不足错误，测试会自动中断并保存已完成的报告。

## License

MIT

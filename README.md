# 🚗 买车智能体 (Car Advisor)

基于 LangGraph + RAG 的智能购车顾问，支持多轮对话、工具调用、
语义检索，提供 CLI / API / Web 三种交互方式。

## 功能

- **智能对话**：理解模糊需求，通过追问逐步明确用户偏好
- **结构化搜索**：根据预算、车型、能源类型精确筛选
- **RAG 语义检索**：向量搜索车型口碑、卖点、适用场景
- **车型对比**：多车并排对比参数和优缺点
- **多轮记忆**：MemorySaver 持久化对话上下文
- **流式输出**：实时展示 Agent 推理过程（工具调用→结果→推荐）

## 环境要求

- Python 3.11+
- DeepSeek API Key（兼容 OpenAI 格式）
- 可选：Docker & Docker Compose

## 快速开始

### 方式一：Docker（推荐）

```bash
cp .env.example .env        # 编辑填入 DEEPSEEK_API_KEY
docker-compose up --build   # 构建并启动
```

API 地址：http://localhost:8000
API 文档：http://localhost:8000/docs
前端界面：http://localhost:8501（需单独启动）

### 方式二：本地 Python

```bash
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY

# CLI 模式
python -m car_advisor.src.main

# LangGraph CLI（支持工具调用）
python -m car_advisor.src.main_langgraph

# API 服务
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Streamlit 前端
streamlit run app/frontend/streamlit_app.py
```

首次运行会自动下载 Embedding 模型（~80MB）并构建向量索引。
国内用户推荐设置 `HF_ENDPOINT=https://hf-mirror.com` 加速下载。

## 项目结构

```
car_advisor/
├── app/                      # 服务化层
│   ├── main.py               # FastAPI 入口
│   ├── routes/chat.py        # /chat + /chat/stream API
│   ├── services/agent_service.py  # Agent 调用封装
│   ├── frontend/streamlit_app.py  # Streamlit 前端
│   └── static/
├── car_advisor/src/          # 核心引擎
│   ├── config.py             # 配置管理
│   ├── llm_client.py         # DeepSeek LLM 客户端
│   ├── prompts.py            # 提示词模板
│   ├── state.py              # LangGraph 状态定义
│   ├── tools.py              # 工具集（搜索/对比/RAG）
│   ├── graph.py              # Agent 状态图
│   ├── main.py               # CLI v1（JSON 模式）
│   ├── main_langgraph.py     # CLI v2（LangGraph）
│   ├── display.py            # CLI 展示公共模块
│   ├── rag/                  # RAG 检索模块
│   │   ├── vector_store.py   # Chroma 向量存储
│   │   └── retriever.py      # LangChain Retriever
│   └── eval/                 # 评估模块
│       ├── eval_dataset.py   # 12 题测试集
│       └── run_eval.py       # 批量评估
├── data/
│   ├── car_data.json         # 结构化车型数据（5款）
│   ├── car_docs/             # 车型详细文档（10款）
│   └── build_index.py        # 向量索引构建脚本
├── tests/                    # 单元 + 集成测试
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥（**必填**） | - |
| `DEEPSEEK_BASE_URL` | API 地址 | `https://api.deepseek.com/v1` |
| `DEEPSEEK_MODEL` | 模型名称 | `deepseek-chat` |
| `DEEPSEEK_TEMPERATURE` | 生成温度 | `0.7` |
| `DEEPSEEK_MAX_TOKENS` | 最大 Token | `4096` |
| `HF_ENDPOINT` | HuggingFace 镜像 | `https://hf-mirror.com` |
| `MAX_HISTORY` | 最大对话轮数 | `20` |
| `VERBOSE` | 调试日志 | `false` |

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/` | API 说明 |
| `GET` | `/health` | 健康检查 |
| `POST` | `/chat` | 同步对话，返回 `ChatResponse` |
| `POST` | `/chat/stream` | SSE 流式对话 |

```bash
# 测试
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_input": "推荐一款15万的SUV", "thread_id": "user_001"}'
```

## 测试

```bash
# 单元测试
pytest tests/ -m "not integration" -v

# 集成测试（需要 API Key）
pytest tests/ -m integration -v

# RAG 评估
python data/build_index.py --force --test
python -m car_advisor.src.eval.run_eval
```

## 许可证

仅供学习和个人使用。

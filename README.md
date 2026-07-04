# 🚗 买车智能体 (Car Advisor)

基于 LangGraph + DeepSeek 的智能购车顾问，支持多轮对话、工具调用、
联网搜索，提供 CLI / API / Web 三种交互方式。

## 功能

- **结构化搜索**：按预算、车型、能源类型、油耗、续航、品牌等条件精确筛选 26 款主流车型
- **车型对比**：多车并排参数对比表格（价格/油耗/续航/动力），绿色标注最优值
- **联网实时搜索**：通过 Tavily 获取最新优惠、车主口碑、行业新闻
- **多意图识别**：preference（偏好）/ search（搜索）/ compare（对比）/ opinion（评价）/ question（提问）
- **多维筛选**：能源类型模糊匹配（"混动"→HEV+PHEV）、品牌排除（"不要日系"）、油耗上限/续航下限
- **参数筛选**：用户说"百公里加速7秒以内"→从车型库中匹配合适车型
- **同义词识别**：越野车→SUV、房车→轿车、商务车→MPV
- **上下文压缩**：长对话自动生成历史摘要，降低 Token 消耗
- **会话持久化**：对话历史存 SQLite，侧边栏切换/删除会话
- **流式输出**：实时展示分析进度（"正在检索车型…"→"正在搜索优惠…"→结果）

## 环境要求

- Python 3.11+
- DeepSeek API Key
- 可选：Tavily API Key（联网搜索）、Docker

## 快速开始

### 本地

```bash
pip install -r requirements.txt
cp .env.example .env          # 填入 DEEPSEEK_API_KEY

# API + 前端
uvicorn app.main:app --port 8000
streamlit run app/frontend/streamlit_app.py

# 或一键启动
python start.py
```

### Docker

```bash
docker-compose up -d
```

### CLI

```bash
python -m car_advisor.src.main_langgraph
```

## 项目结构

```
├── app/                          # 服务化层
│   ├── main.py                   # FastAPI 入口
│   ├── routes/chat.py            # /chat + /chat/stream + DELETE /conversation
│   ├── services/agent_service.py # Agent 调用封装（invoke / ainvoke / astream）
│   └── frontend/streamlit_app.py # Web 前端（侧边栏会话管理 + 对比表格）
├── car_advisor/src/              # 核心引擎
│   ├── config.py                 # 配置管理
│   ├── http_client.py            # 共享 httpx 连接池
│   ├── llm_client.py             # LLM 客户端（重试 + 连接池）
│   ├── utils.py                  # 消息工具函数
│   ├── display.py                # CLI 展示模块
│   ├── session_store.py          # 对话存储（sessions.db）
│   ├── prompts.py                # 系统提示词（5 意图 + 5 步对比流程）
│   ├── state.py                  # LangGraph 状态定义
│   ├── tools.py                  # 3 工具（search / compare / search_online）
│   ├── graph.py                  # Agent 状态图（强路由 + 压缩 + 参数筛选）
│   ├── main.py                   # CLI v1（JSON 模式）
│   └── main_langgraph.py         # CLI v2（LangGraph）
├── data/
│   ├── car_data.json             # 26 款车型
│   └── eval_results.json         # RAG 评估证据（历史）
├── tests/                        # 136 单元测试 + 4 集成测试
│   ├── test_config.py            # 21
│   ├── test_llm_client.py        # 22
│   ├── test_main.py              # 53
│   ├── test_tools.py             # 32
│   └── test_graph.py             # 8
├── Dockerfile / docker-compose.yml
├── start.py                      # 一键启动脚本
└── requirements.txt              # 16 个依赖
```

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥（**必填**） | - |
| `DEEPSEEK_BASE_URL` | API 地址 | `https://api.deepseek.com/v1` |
| `DEEPSEEK_MODEL` | 模型名称 | `deepseek-chat` |
| `TAVILY_API_KEY` | Tavily 联网搜索（可选） | - |

## API

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查 |
| `POST` | `/chat` | 同步对话 |
| `POST` | `/chat/stream` | SSE 流式对话 |
| `DELETE` | `/conversation/{id}` | 删除会话 |

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_input":"推荐一款15万的SUV","thread_id":"u1"}'
```

## 测试

```bash
# 单元测试（136 例）
pytest tests/ -m "not integration" -v

# 集成测试（需 API Key）
pytest tests/ -m integration -v
```

| 模块 | 用例 | 覆盖 |
|------|:--:|------|
| `test_config.py` | 21 | 配置加载 / 边界值 |
| `test_llm_client.py` | 22 | JSON 解析 / 异常翻译 |
| `test_main.py` | 53 | CLI 指令 / 历史裁剪 / 展示 |
| `test_tools.py` | 32 | 预算解析 / 同义词 / 燃料匹配 / 多维筛选 |
| `test_graph.py` | 8 | 参数关键词检测 |

## 技术演进

- 早期尝试 RAG 语义检索（Chroma + Embedding），评估 32 题后决定移除
- RAG 评估证据保留在 `data/eval_results.json`（命中率 46.4%，MRR 0.339）
- 当前方案：结构化筛选 + Tavily 联网搜索，更精准可靠

## 许可证

仅供学习和个人使用。

# 🚗 买车智能体 (Car Advisor)

一个基于大语言模型的交互式购车顾问，帮助你理清需求、对比车型、
分析优劣，做出更明智的购车决策。

## 功能

- **需求分析**：通过对话了解你的预算、用途、偏好，梳理购车需求
- **车型推荐**：根据需求推荐合适的车型，涵盖燃油车和新能源车
- **车型对比**：从价格、油耗、空间、动力、配置、保值率等多维度对比
- **燃油 vs 新能源**：结合你的实际使用场景分析哪种更适合
- **购车流程指导**：梳理选车、试驾、谈价、贷款、保险、提车等全流程
- **贷款分析**：帮助计算合理的贷款方案，提醒隐性成本

## 环境要求

- Python 3.11+
- OpenAI API 密钥 或 Anthropic API 密钥

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入你的 API 密钥：

```ini
LLM_PROVIDER=openai          # 或 anthropic
LLM_API_KEY=sk-xxx           # 你的 API 密钥
LLM_MODEL=gpt-4o             # 模型名称
```

### 3. 运行

```bash
python -m car_advisor.src.main
```

## 项目结构

```
car_advisor/
├── src/
│   ├── __init__.py          # 包说明
│   ├── config.py            # 环境变量和配置管理
│   ├── llm_client.py        # 大模型调用封装（OpenAI / Anthropic）
│   ├── prompts.py           # 系统提示词和用户提示词模板
│   └── main.py              # 命令行交互入口
├── .env.example             # 环境变量示例文件
├── requirements.txt         # 依赖清单
└── README.md                # 项目说明
```

## 使用说明

启动后会进入交互式对话界面，你可以：

- **直接对话**：用自然语言描述你的需求
- **快捷指令**：
  - `/对比` — 对比多款车型
  - `/新能源` — 分析燃油车 vs 新能源
  - `/流程` — 了解购车流程
  - `/贷款` — 分析贷款方案
  - `/帮助` — 显示帮助信息
  - `/清空` — 清空对话历史
  - `/退出` — 结束对话

## 自定义 API 地址

如果你使用本地模型或中转代理，可以在 `.env` 中设置：

```ini
LLM_API_BASE=http://localhost:11434/v1
```

## 许可证

仅供学习和个人使用。

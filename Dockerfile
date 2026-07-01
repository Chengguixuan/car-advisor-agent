# 买车智能体 — Docker 镜像
# 基于 Python 3.11，暴露 FastAPI 服务在 8000 端口

FROM python:3.11-slim

# 系统依赖（chromadb 需要 sqlite3 等）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先复制依赖文件，利用 Docker 缓存层
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# 预下载 HuggingFace 嵌入模型（可选，加速首次启动）
# 设置国内镜像以加速下载
ENV HF_ENDPOINT=https://hf-mirror.com
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" || true

# 预构建车型向量索引
RUN python -c "\
import sys; sys.path.insert(0, '.'); \
from car_advisor.src.rag.vector_store import CarVectorStore; \
CarVectorStore().build_index()" || true

EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

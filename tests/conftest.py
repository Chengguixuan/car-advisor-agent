"""pytest 全局配置。

在测试会话启动时加载 .env 文件，确保集成测试能读取到环境变量。
"""

from pathlib import Path

from dotenv import load_dotenv

# 项目根目录 = tests/ 的上一级
_project_root = Path(__file__).resolve().parent.parent
_env_file = _project_root / ".env"
load_dotenv(_env_file)

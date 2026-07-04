"""一键启动 — 同时拉起 API 和 Streamlit 前端。"""

import subprocess
import sys
import time


def main():
    print("🚗 买车智能体 — 启动中...")
    print()

    # 启动 API
    api = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"],
    )
    print("✅ API 服务已启动 → http://localhost:8000")

    time.sleep(3)

    # 启动前端
    frontend = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app/frontend/streamlit_app.py"],
    )
    print("✅ 前端已启动 → http://localhost:8501")
    print()
    print("按 Ctrl+C 停止所有服务")

    try:
        api.wait()
        frontend.wait()
    except KeyboardInterrupt:
        api.terminate()
        frontend.terminate()
        print("\n已停止")


if __name__ == "__main__":
    main()

"""Start the demo:  python run.py   ->  http://127.0.0.1:8000"""
import sys

import uvicorn

if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    uvicorn.run("app.server:app", host="127.0.0.1", port=8000, reload=False, log_level="info")

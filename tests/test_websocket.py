import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import asyncio
import json
import time
import threading
import uvicorn
import websockets
from src.app import app

def run_server():
    uvicorn.run(app, host="127.0.0.1", port=8006, log_level="warning")

async def test_ws():
    uri = "ws://127.0.0.1:8006/ws/investigate"
    async with websockets.connect(uri) as ws:
        req = {"category": "SSH-Patator", "constrained": True, "epsilon": 0.10, "num_steps": 5}
        await ws.send(json.dumps(req))
        
        step_count = 0
        while True:
            msg_txt = await ws.recv()
            msg = json.loads(msg_txt)
            print(f"WS msg: type={msg.get('type')}, step={msg.get('step')}")
            if msg.get("type") == "step":
                step_count += 1
            if msg.get("type") == "complete":
                break
        print(f"WS test completed successfully with {step_count} steps!")

if __name__ == "__main__":
    t = threading.Thread(target=run_server, daemon=True)
    t.start()
    time.sleep(2)
    asyncio.run(test_ws())

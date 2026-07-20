"""
FastAPI server: the "program" the UI connects to.

Runs the trained PPO policy against the real environment forever in a
background thread (see live_runner.py), and streams the live state to
any connected browser over a WebSocket. It serves the live station,
signals, and dashboard pages from the same origin so the whole control
room stays connected to the running model.

Run:
        python app.py
Then open http://localhost:8000/
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
        sys.path.insert(0, ROOT)

from server.live_runner import LiveRunner

MODEL_PATH = os.path.join(ROOT, "logs", "ppo_terminus.zip")
VIZ_DIR = os.path.join(ROOT, "viz")

app = FastAPI()
runner: LiveRunner = None


@app.on_event("startup")
def _startup():
        global runner
        runner = LiveRunner(model_path=MODEL_PATH, steps_per_sec=3.0)
        runner.start()


@app.on_event("shutdown")
def _shutdown():
        if runner is not None:
                runner.stop()


@app.get("/api/state")
def get_state():
        return runner.snapshot()


@app.websocket("/ws")
async def ws_state(ws: WebSocket):
        await ws.accept()
        try:
                while True:
                        await ws.send_text(json.dumps(runner.snapshot()))
                        await asyncio.sleep(0.3)
        except WebSocketDisconnect:
                pass


def _load_viz(filename: str) -> str:
        with open(os.path.join(VIZ_DIR, filename), encoding="utf-8") as f:
                return f.read()


@app.get("/", response_class=HTMLResponse)
def index():
        return """
        <html><head><title>Terminus Control — Live</title>
        <style>
            :root { --bg:#0a0e13; --panel:#10161d; --panel-border:#1e2932; --tile:#05070a; --amber:#ffb000; --cyan:#4fd1e8; --green:#3ecf6e; --text:#d7dee5; --text-dim:#6b7d8f; --mono:'SFMono-Regular', ui-monospace, Menlo, monospace; }
            * { box-sizing:border-box; }
            body { margin:0; background:radial-gradient(circle at top, #121a24 0%, var(--bg) 55%); color:var(--text); font-family:-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
            .wrap { max-width:1100px; margin:0 auto; padding:40px 18px 48px; }
            .eyebrow { font-family:var(--mono); font-size:11px; letter-spacing:0.22em; color:var(--text-dim); text-transform:uppercase; }
            h1 { font-family:var(--mono); margin:8px 0 10px; color:var(--amber); letter-spacing:0.03em; font-size:28px; }
            p { color:var(--text-dim); max-width:760px; line-height:1.6; }
            .grid { display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:12px; margin-top:24px; }
            @media (max-width:760px) { .grid { grid-template-columns:1fr; } }
            .card { display:block; text-decoration:none; color:inherit; background:linear-gradient(180deg, rgba(16,22,29,0.96), rgba(8,11,15,0.96)); border:1px solid var(--panel-border); border-radius:14px; padding:18px; min-height:132px; transition:transform 0.15s, border-color 0.15s; }
            .card:hover { border-color:#2a3944; transform: translateY(-1px); }
            .card h2 { margin:0 0 8px; font-family:var(--mono); font-size:16px; color:var(--cyan); letter-spacing:0.03em; }
            .card .desc { font-size:14px; color:var(--text-dim); line-height:1.55; }
            .card .meta { margin-top:14px; font-family:var(--mono); font-size:11px; color:var(--green); letter-spacing:0.08em; text-transform:uppercase; }
            .bar { display:flex; gap:10px; flex-wrap:wrap; margin-top:18px; }
            .pill { font-family:var(--mono); font-size:11px; color:var(--amber); border:1px solid var(--panel-border); background:var(--tile); border-radius:999px; padding:6px 10px; }
            .note { margin-top:18px; font-family:var(--mono); font-size:11px; color:var(--text-dim); }
            a { color:inherit; }
        </style></head>
        <body>
            <div class="wrap">
                <div class="eyebrow">MGR Chennai Central · Railway Terminal MARL</div>
                <h1>TERMINUS CONTROL — LIVE SERVER</h1>
                <p>This server runs the actual PPO policy against the terminus environment and streams the live state to the browser. The station board, signal board, and dashboard all read the same running program.</p>
                <div class="bar">
                    <div class="pill">WebSocket live state</div>
                    <div class="pill">Live policy inference</div>
                    <div class="pill">Platform allocation</div>
                    <div class="pill">Metrics dashboard</div>
                </div>
                <div class="grid">
                    <a class="card" href="/station">
                        <h2>Live Station Board</h2>
                        <div class="desc">Platform-by-platform occupancy, train movements, dwell bars, and allocation events streamed directly from the running model.</div>
                        <div class="meta">Open /station</div>
                    </a>
                    <a class="card" href="/signals">
                        <h2>Live Signal Board</h2>
                        <div class="desc">A schematic interlocking-style view showing platform occupancy, signal state, and contention alerts in real time.</div>
                        <div class="meta">Open /signals</div>
                    </a>
                    <a class="card" href="/dashboard">
                        <h2>Live Metrics Dashboard</h2>
                        <div class="desc">System status, utilization, queue depth, recent allocations, and upcoming arrivals for the running episode.</div>
                        <div class="meta">Open /dashboard</div>
                    </a>
                    <a class="card" href="/api/state">
                        <h2>Raw State Snapshot</h2>
                        <div class="desc">Direct JSON for integrations, debugging, or building additional views against the same live program.</div>
                        <div class="meta">Open /api/state</div>
                    </a>
                </div>
                <div class="note">The live server keeps resetting to a fresh episode when the current one ends, so the control room never stalls.</div>
            </div>
        </body></html>
        """


@app.get("/station", response_class=HTMLResponse)
def station_page():
        return _load_viz("live_station.html")


@app.get("/signals", response_class=HTMLResponse)
def signals_page():
        return _load_viz("live_signals.html")


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard_page():
        return _load_viz("live_dashboard.html")


if __name__ == "__main__":
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)

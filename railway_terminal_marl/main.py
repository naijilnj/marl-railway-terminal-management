from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def build_static():
    subprocess.run([sys.executable, os.path.join(ROOT, "viz", "build.py")], check=True)


def serve_live(port: int = 8000):
    import uvicorn

    bind_port = int(os.environ.get("PORT", port))
    uvicorn.run("server.app:app", host="0.0.0.0", port=bind_port, reload=False)


def train_model(timesteps: int, scenario: str, seed: int):
    from training.train import record_rollout, train

    model = train(timesteps, scenario, seed=seed)
    record_rollout(model, scenario, seed=seed)


def main():
    parser = argparse.ArgumentParser(description="Railway Terminal MARL launcher")
    parser.add_argument("--serve", action="store_true", help="run the live FastAPI control room")
    parser.add_argument("--dashboard", action="store_true", help="serve the static telemetry dashboard")
    parser.add_argument("--build", action="store_true", help="regenerate static HTML views from logs")
    parser.add_argument("--port", type=int, default=8000, help="port for live/server modes")
    parser.add_argument("--timesteps", type=int, default=40000, help="training timesteps")
    parser.add_argument("--episodes", type=int, help="backward-compatible alias for --timesteps")
    parser.add_argument("--scenario", choices=["default", "mgr_chennai_central"], default="default")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    timesteps = args.episodes if args.episodes is not None else args.timesteps

    if args.build:
        build_static()
        return
    if args.dashboard:
        from dashboard import serve_dashboard

        serve_dashboard(port=args.port)
        return
    if args.serve:
        serve_live(port=args.port)
        return

    train_model(timesteps=timesteps, scenario=args.scenario, seed=args.seed)


if __name__ == "__main__":
    main()

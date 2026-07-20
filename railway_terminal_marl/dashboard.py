from __future__ import annotations

import argparse
import os
import subprocess
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
VIZ_DIR = os.path.join(ROOT, "viz")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class DashboardHandler(SimpleHTTPRequestHandler):
    def translate_path(self, path):
        if path in ("/", ""):
            path = "/dashboard.html"
        elif path == "/simulator":
            path = "/mgr_station_simulator.html"
        elif path == "/dashboard":
            path = "/dashboard.html"
        return super().translate_path(path)


def build_static():
    subprocess.run([sys.executable, os.path.join(ROOT, "viz", "build.py")], check=True)


def serve_dashboard(port: int = 8001):
    os.chdir(VIZ_DIR)
    server = ThreadingHTTPServer(("0.0.0.0", port), DashboardHandler)
    print(f"Static dashboard serving on http://localhost:{port}/")
    print(f"  dashboard: http://localhost:{port}/dashboard.html")
    print(f"  simulator:  http://localhost:{port}/mgr_station_simulator.html")
    print("  live views: run python main.py --serve on port 8000")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Static telemetry dashboard server")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--skip-build", action="store_true", help="serve existing static files without rebuilding")
    args = parser.parse_args()

    if not args.skip_build:
        build_static()
    serve_dashboard(port=args.port)

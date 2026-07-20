"""
Reads logs/events.json and logs/metrics.jsonl, dedupes, and writes
viz/simulator.html and viz/dashboard.html with the real data inlined
as a JSON blob (no external fetch needed -> portable single files).
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "logs")
VIZ = os.path.dirname(os.path.abspath(__file__))


def load_events():
    with open(os.path.join(LOGS, "events.json")) as f:
        return json.load(f)


def load_metrics():
    rows, seen = [], set()
    with open(os.path.join(LOGS, "metrics.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r["global_step"] in seen:
                continue
            seen.add(r["global_step"])
            rows.append({k: r[k] for k in
                         ("global_step", "avg_wait", "trains_served", "n_waiting", "invalid_actions", "conflicts")})
    return rows


def inject(template_path, out_path, data):
    with open(template_path) as f:
        html = f.read()
    html = html.replace("/*__DATA__*/", json.dumps(data))
    with open(out_path, "w") as f:
        f.write(html)
    print(f"wrote {out_path} ({len(html)} bytes)")


if __name__ == "__main__":
    inject(os.path.join(VIZ, "mgr_station_simulator_template.html"),
           os.path.join(VIZ, "mgr_station_simulator.html"), load_events())
    inject(os.path.join(VIZ, "dashboard_template.html"),
           os.path.join(VIZ, "dashboard.html"), load_metrics())

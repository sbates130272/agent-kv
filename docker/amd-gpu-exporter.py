"""Minimal Prometheus exporter for AMD GPUs using amdsmi.

Runs an HTTP server on PORT (default 9200) that serves /metrics in
Prometheus text format.  Designed for consumer RDNA GPUs (gfx1201+)
where the official amdih/amd-smi-exporter crashes (rsmi limitation).

Requires the amdsmi Python module — present in the agent-kv/vllm image
via /opt/rocm/share/amd_smi on sys.path.
"""

import http.server
import os
import threading
import time

import amdsmi

PORT = int(os.environ.get("AMD_GPU_EXPORTER_PORT", "9200"))
SCRAPE_INTERVAL = float(os.environ.get("AMD_GPU_EXPORTER_INTERVAL", "5"))


def _safe(fn, *args, default="N/A"):
    try:
        return fn(*args)
    except Exception:
        return default


def collect():
    amdsmi.amdsmi_init()
    try:
        handles = amdsmi.amdsmi_get_processor_handles()
        lines = []
        for idx, h in enumerate(handles):
            labels = f'gpu="{idx}"'

            power = _safe(amdsmi.amdsmi_get_power_info, h, default={})
            if isinstance(power, dict):
                for key, metric, unit in [
                    ("average_socket_power", "avg_power_watts", "W"),
                    ("socket_power", "current_power_watts", "W"),
                    ("power_limit", "power_limit_watts", "W"),
                ]:
                    val = power.get(key)
                    if val and val != "N/A":
                        if key == "power_limit":
                            val = val / 1_000_000
                        lines.append(
                            f"# HELP amd_gpu_{metric} GPU power ({unit})\n"
                            f"# TYPE amd_gpu_{metric} gauge\n"
                            f"amd_gpu_{metric}{{{labels}}} {val}"
                        )

            temp = _safe(
                amdsmi.amdsmi_get_temp_metric,
                h,
                amdsmi.AmdSmiTemperatureType.JUNCTION,
                amdsmi.AmdSmiTemperatureMetric.CURRENT,
                default=None,
            )
            if temp is not None and temp != "N/A":
                lines.append(
                    f"# HELP amd_gpu_temperature_junction_celsius Junction temperature\n"
                    f"# TYPE amd_gpu_temperature_junction_celsius gauge\n"
                    f"amd_gpu_temperature_junction_celsius{{{labels}}} {temp}"
                )

            vram = _safe(amdsmi.amdsmi_get_gpu_vram_usage, h, default={})
            if isinstance(vram, dict):
                for key, metric in [
                    ("vram_total", "vram_total_mib"),
                    ("vram_used", "vram_used_mib"),
                ]:
                    val = vram.get(key)
                    if val is not None and val != "N/A":
                        lines.append(
                            f"# HELP amd_gpu_{metric} VRAM {key.split('_')[1]} (MiB)\n"
                            f"# TYPE amd_gpu_{metric} gauge\n"
                            f"amd_gpu_{metric}{{{labels}}} {val}"
                        )

            activity = _safe(amdsmi.amdsmi_get_gpu_activity, h, default={})
            if isinstance(activity, dict):
                for key, metric in [
                    ("gfx_activity", "gfx_activity_percent"),
                    ("umc_activity", "memory_activity_percent"),
                ]:
                    val = activity.get(key)
                    if val is not None and val != "N/A":
                        lines.append(
                            f"# HELP amd_gpu_{metric} GPU {key} (0-100)\n"
                            f"# TYPE amd_gpu_{metric} gauge\n"
                            f"amd_gpu_{metric}{{{labels}}} {val}"
                        )

        return "\n".join(lines) + "\n"
    finally:
        amdsmi.amdsmi_shut_down()


_cache = {"body": "", "ts": 0}
_lock = threading.Lock()


def get_metrics():
    now = time.monotonic()
    with _lock:
        if now - _cache["ts"] > SCRAPE_INTERVAL:
            try:
                _cache["body"] = collect()
            except Exception as exc:
                _cache["body"] = f"# ERROR {exc}\n"
            _cache["ts"] = now
        return _cache["body"]


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        if self.path == "/metrics":
            body = get_metrics().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()


if __name__ == "__main__":
    server = http.server.HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"amd-gpu-exporter listening on :{PORT}/metrics", flush=True)
    server.serve_forever()

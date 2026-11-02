"""Checkpoint-only dashboard for a running recovery campaign.

Safety contract
---------------
This monitor deliberately does not open campaign status, manifest, metric, or
configuration files.  It only inspects process/resource state, directory
entries, immutable checkpoint names/metadata, completion-artifact existence,
and the byte size of the dedicated launcher stderr log.  Keeping those files
out of the polling path avoids racing the trainer's atomic JSON replacements on
Windows.

The script lives below ``tools/`` so it is outside the preregistered scientific
source identity.  Run it with ``pythonw.exe``; the recovery guard intentionally
ignores dashboard ``pythonw.exe`` processes.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
from typing import Any


DEFAULT_ROOT = Path(
    r"E:\YTT\GROUP\DynFL-privacy\out\fmnist_joint_jt_100r_3seed_20261022"
)
DEFAULT_STDERR = (
    DEFAULT_ROOT
    / "_recovery"
    / "launcher_logs"
    / "continuation_resume_20261024_173407_stderr.log"
)
WATCHED_STRATEGIES = ("max_J", "min_T", "pure_FL_E", "pure_FL_EC", "pure_FL_C")
CHECKPOINT_RE = re.compile(r"^checkpoint_round_(\d{3})\.pt$")

_CACHE_LOCK = threading.Lock()
_CACHE_AT = 0.0
_CACHE: dict[str, Any] | None = None


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    process_query_limited_information = 0x1000
    still_active = 259
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
    if not handle:
        return False
    try:
        exit_code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return False
        return exit_code.value == still_active
    finally:
        kernel32.CloseHandle(handle)


def _latest_checkpoint(folder: Path) -> dict[str, Any]:
    if not folder.is_dir():
        return {
            "state": "queued",
            "round": 0,
            "checkpoint": None,
            "checkpoint_mtime": None,
            "sidecar": False,
            "summary": False,
            "final_state": False,
        }
    latest_round = 0
    latest_name: str | None = None
    latest_mtime: float | None = None
    with os.scandir(folder) as entries:
        for entry in entries:
            match = CHECKPOINT_RE.match(entry.name)
            if not match or not entry.is_file(follow_symlinks=False):
                continue
            round_number = int(match.group(1))
            if round_number > latest_round:
                latest_round = round_number
                latest_name = entry.name
                latest_mtime = entry.stat(follow_symlinks=False).st_mtime
    sidecar = bool(
        latest_name
        and (folder / latest_name.replace(".pt", ".sha256")).is_file()
    )
    summary = (folder / "summary.json").is_file()
    final_state = (folder / "final_state.pt").is_file()
    state = "completed" if latest_round >= 100 and summary and final_state else "running"
    return {
        "state": state,
        "round": latest_round,
        "checkpoint": latest_name,
        "checkpoint_mtime": latest_mtime,
        "sidecar": sidecar,
        "summary": summary,
        "final_state": final_state,
    }


def _gpu_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=4,
            check=True,
            creationflags=(0x08000000 if os.name == "nt" else 0),
        )
        values = [part.strip() for part in result.stdout.splitlines()[0].split(",")]
        return {
            "available": True,
            "utilization_pct": int(values[0]),
            "memory_used_mib": int(values[1]),
            "memory_total_mib": int(values[2]),
            "temperature_c": int(values[3]),
        }
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return {"available": False}


def _snapshot(root: Path, pid: int, stderr_path: Path) -> dict[str, Any]:
    global _CACHE_AT, _CACHE
    now = time.monotonic()
    if _CACHE is not None and now - _CACHE_AT < 4.0:
        return _CACHE
    with _CACHE_LOCK:
        now = time.monotonic()
        if _CACHE is not None and now - _CACHE_AT < 4.0:
            return _CACHE
        seed_dir = root / "seed_44"
        strategies = []
        for strategy in WATCHED_STRATEGIES:
            item = _latest_checkpoint(seed_dir / strategy)
            item["strategy"] = strategy
            strategies.append(item)
        disk = shutil.disk_usage(root)
        try:
            stderr_bytes = stderr_path.stat().st_size
        except OSError:
            stderr_bytes = None
        payload = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "safety_mode": "checkpoint_metadata_only",
            "root": str(root),
            "recovery_pid": pid,
            "process_alive": _pid_alive(pid),
            "stderr_path": str(stderr_path),
            "stderr_bytes": stderr_bytes,
            "disk_free_gib": disk.free / (1024**3),
            "gpu": _gpu_snapshot(),
            "strategies": strategies,
        }
        _CACHE = payload
        _CACHE_AT = now
        return payload


PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DynFL 安全简版监控</title>
  <style>
    :root{color-scheme:dark;--bg:#07111c;--panel:#101d2b;--line:#273a50;--text:#edf5ff;--muted:#91a8bf;--ok:#68dda0;--run:#63b5ff;--wait:#f2c267;--bad:#ff8585}
    *{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 15% 0,#123048 0,transparent 35%),var(--bg);color:var(--text);font-family:Inter,"Segoe UI",sans-serif}
    main{max-width:1060px;margin:auto;padding:28px 22px 44px}.top{display:flex;justify-content:space-between;align-items:flex-start;gap:20px}.eyebrow{color:#6bd9d0;font-size:12px;letter-spacing:.12em;text-transform:uppercase}.title{margin:7px 0 5px;font-size:28px}.sub{color:var(--muted);font-size:13px;line-height:1.55}.pill{padding:7px 11px;border:1px solid var(--line);border-radius:999px;font-size:12px;color:var(--muted)}
    .notice{margin:20px 0;padding:12px 14px;border:1px solid #255767;background:#0d2937;border-radius:10px;color:#bdeef2;font-size:13px;line-height:1.5}.summary{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:16px 0}.card,.run{background:linear-gradient(180deg,#122235,#0e1a28);border:1px solid var(--line);border-radius:12px;padding:15px}.label{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em}.value{font-size:23px;margin-top:7px}.runs{display:grid;gap:12px}.run-head{display:flex;justify-content:space-between;gap:15px;align-items:center}.name{font-weight:650}.state{font-size:12px}.state.completed{color:var(--ok)}.state.running{color:var(--run)}.state.queued{color:var(--wait)}.bar{height:7px;background:#07111c;border-radius:99px;overflow:hidden;margin:12px 0 9px}.bar i{display:block;height:100%;background:linear-gradient(90deg,#48c9c1,#65a9ff)}.meta{display:flex;justify-content:space-between;gap:12px;color:var(--muted);font-size:12px}.error{color:var(--bad)}footer{color:var(--muted);font-size:11px;margin-top:18px}@media(max-width:720px){.summary{grid-template-columns:repeat(2,1fr)}.top{display:block}.pill{display:inline-block;margin-top:12px}}
  </style>
</head>
<body><main>
  <div class="top"><div><div class="eyebrow">Checkpoint metadata monitor</div><h1 class="title">DynFL 安全简版监控</h1><div class="sub" id="root">正在连接…</div></div><div class="pill" id="refreshed">等待刷新</div></div>
  <div class="notice">安全模式仅查看恢复进程、检查点文件名与修改时间、GPU、磁盘和错误日志大小；不会读取训练状态、逐轮指标或恢复清单。</div>
  <section class="summary">
    <div class="card"><div class="label">恢复进程</div><div class="value" id="process">—</div></div>
    <div class="card"><div class="label">GPU 使用率</div><div class="value" id="gpu">—</div></div>
    <div class="card"><div class="label">GPU 显存</div><div class="value" id="memory">—</div></div>
    <div class="card"><div class="label">E 盘可用</div><div class="value" id="disk">—</div></div>
  </section>
  <section class="runs" id="runs"></section>
  <footer id="footer">—</footer>
</main>
<script>
const $=id=>document.getElementById(id);const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function render(d){$('root').textContent=d.root;$('process').textContent=d.process_alive?`PID ${d.recovery_pid} · 运行中`:`PID ${d.recovery_pid} · 已停止`;$('process').className=`value ${d.process_alive?'':'error'}`;const g=d.gpu||{};$('gpu').textContent=g.available?`${g.utilization_pct}% · ${g.temperature_c}°C`:'不可用';$('memory').textContent=g.available?`${g.memory_used_mib} / ${g.memory_total_mib} MiB`:'不可用';$('disk').textContent=`${d.disk_free_gib.toFixed(1)} GiB`;$('runs').innerHTML=d.strategies.map(s=>{const pct=Math.max(0,Math.min(100,s.round));const state=s.state==='completed'?'已完成':s.state==='running'?'运行中或待切换':'排队中';const stamp=s.checkpoint_mtime?new Date(s.checkpoint_mtime*1000).toLocaleTimeString():'—';return `<div class="run"><div class="run-head"><span class="name">${esc(s.strategy)}</span><span class="state ${s.state}">${state}</span></div><div class="bar"><i style="width:${pct}%"></i></div><div class="meta"><span>最新检查点：${s.round||0} / 100</span><span>${stamp} · sidecar ${s.sidecar?'✓':'—'}</span></div></div>`}).join('');$('footer').innerHTML=`安全模式：${esc(d.safety_mode)} · stderr ${d.stderr_bytes??'—'} bytes · ${new Date(d.generated_utc).toLocaleString()}`;$('refreshed').textContent=`刷新于 ${new Date().toLocaleTimeString()}`;}
async function refresh(){try{const r=await fetch('/api/status',{cache:'no-store'});const d=await r.json();if(!r.ok)throw new Error(d.error||r.status);render(d)}catch(e){$('process').textContent='连接失败';$('process').className='value error';$('footer').textContent=e.message}}
refresh();setInterval(refresh,5000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    root: Path
    recovery_pid: int
    stderr_path: Path

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - stdlib callback name
        route = self.path.split("?", 1)[0]
        if route == "/":
            self._send(200, "text/html; charset=utf-8", PAGE.encode("utf-8"))
            return
        if route == "/api/status":
            try:
                payload = _snapshot(self.root, self.recovery_pid, self.stderr_path)
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self._send(200, "application/json; charset=utf-8", body)
            except Exception as error:  # Fail visibly without touching training.
                body = json.dumps({"error": str(error)}, ensure_ascii=False).encode("utf-8")
                self._send(500, "application/json; charset=utf-8", body)
            return
        self._send(404, "text/plain; charset=utf-8", b"Not found")

    def log_message(self, _format: str, *_args: object) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--pid", type=int, default=46164)
    parser.add_argument("--stderr", type=Path, default=DEFAULT_STDERR)
    args = parser.parse_args()
    Handler.root = args.root.resolve()
    Handler.recovery_pid = args.pid
    Handler.stderr_path = args.stderr.resolve()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    server.serve_forever()


if __name__ == "__main__":
    main()

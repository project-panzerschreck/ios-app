#!/usr/bin/env python3
"""
inference_server.py

Lightweight device-registry server for the retired-smartphones distributed
GGML inference cluster.

Each iOS/Android device running a GGML RPC worker calls POST /api/v1/devices/register
with its IP address and RPC port.  An operator (or automation script) can then
query GET /api/v1/devices to get the full list and build the --rpc flags for
llama-cli.

Endpoints
─────────
  POST /api/v1/devices/register           – register / update a device
  GET  /api/v1/devices                    – list active devices (JSON)
  POST /api/v1/devices/{device_id}/keepalive – refresh liveness timestamp
  DELETE /api/v1/devices/{device_id}      – manually deregister

  POST /api/v1/inference/command          – generate ready-to-paste llama-cli command
  POST /api/v1/inference/run              – (optional) execute llama-cli directly

Usage
─────
  pip install -r requirements.txt
  python inference_server.py [--host 0.0.0.0] [--port 8080]
             [--llama-cli /path/to/llama-cli] [--model /path/to/model.gguf]
             [--stale-sec 30]

Environment variables (override CLI flags):
  LLAMA_CLI_PATH   path to llama-cli binary
  MODEL_PATH       default model path for the /run endpoint
"""

import argparse
import asyncio
import os
import subprocess
import time
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── CLI / env config ──────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="GGML RPC device registry server")
parser.add_argument("--host",       default="0.0.0.0",  help="Bind address")
parser.add_argument("--port",       type=int, default=8080, help="HTTP port")
parser.add_argument("--llama-cli",  default=os.environ.get("LLAMA_CLI_PATH", "llama-cli"),
                    help="Path to llama-cli binary")
parser.add_argument("--model",      default=os.environ.get("MODEL_PATH", ""),
                    help="Default model path for /api/v1/inference/run")
parser.add_argument("--stale-sec",  type=int, default=30,
                    help="Seconds without a keepalive before a device is removed")
args, _unknown = parser.parse_known_args()

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="GGML RPC Device Registry", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory device store ────────────────────────────────────────────────────

class DeviceRecord(BaseModel):
    device_id: str
    label:     str
    ip:        str
    rpc_port:  int
    last_seen: float = 0.0

    @property
    def endpoint(self) -> str:
        return f"{self.ip}:{self.rpc_port}"

    def is_alive(self, stale_sec: int) -> bool:
        return (time.time() - self.last_seen) < stale_sec


devices: Dict[str, DeviceRecord] = {}

# ── Models ────────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    device_id: str
    label:     str
    ip:        str
    rpc_port:  int

class InferenceCommandRequest(BaseModel):
    prompt:     str
    model_path: Optional[str] = None   # falls back to --model CLI flag
    extra_args: List[str]     = []     # any extra llama-cli flags

class InferenceRunRequest(InferenceCommandRequest):
    timeout_sec: int = 120

# ── Helpers ───────────────────────────────────────────────────────────────────

def active_devices() -> List[DeviceRecord]:
    return [d for d in devices.values() if d.is_alive(args.stale_sec)]

def build_rpc_args(devs: List[DeviceRecord]) -> List[str]:
    """Return ['--rpc', 'ip:port', '--rpc', 'ip:port', …] for all active devices."""
    flags = []
    for d in devs:
        flags += ["--rpc", d.endpoint]
    return flags

def build_command(prompt: str, model_path: str, extra: List[str]) -> List[str]:
    devs = active_devices()
    if not devs:
        raise HTTPException(status_code=409, detail="No active RPC worker devices registered")
    model = model_path or args.model
    if not model:
        raise HTTPException(status_code=422, detail="No model path provided and --model not set")
    return [args.llama_cli] + build_rpc_args(devs) + ["-m", model, "-p", prompt] + extra

# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/api/v1/devices/register", status_code=201)
async def register_device(req: RegisterRequest):
    """Register or refresh a device."""
    rec = DeviceRecord(
        device_id=req.device_id,
        label=req.label,
        ip=req.ip,
        rpc_port=req.rpc_port,
        last_seen=time.time(),
    )
    devices[req.device_id] = rec
    return {"status": "registered", "endpoint": rec.endpoint}


@app.get("/api/v1/devices")
async def list_devices():
    """Return all devices that are still considered alive."""
    devs = active_devices()
    return {
        "count": len(devs),
        "devices": [
            {
                "device_id": d.device_id,
                "label":     d.label,
                "endpoint":  d.endpoint,
                "last_seen": d.last_seen,
            }
            for d in devs
        ],
    }


@app.post("/api/v1/devices/{device_id}/keepalive")
async def keepalive(device_id: str):
    if device_id not in devices:
        raise HTTPException(status_code=404, detail="Device not found")
    devices[device_id].last_seen = time.time()
    return {"status": "ok"}


@app.delete("/api/v1/devices/{device_id}", status_code=204)
async def deregister_device(device_id: str):
    devices.pop(device_id, None)


@app.post("/api/v1/inference/command")
async def inference_command(req: InferenceCommandRequest):
    """
    Return the llama-cli command that would be used to run inference with
    all currently-registered RPC workers.  Does NOT execute anything.
    """
    cmd = build_command(req.prompt, req.model_path or "", req.extra_args)
    return {
        "command":    cmd,
        "command_str": " ".join(cmd),
        "workers":    [d.endpoint for d in active_devices()],
    }


@app.post("/api/v1/inference/run")
async def inference_run(req: InferenceRunRequest):
    """
    Execute llama-cli with the registered RPC workers and return stdout.
    This is a convenience endpoint; for production use, call /command and
    run the process yourself.
    """
    cmd = build_command(req.prompt, req.model_path or "", req.extra_args)
    try:
        proc = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            ),
            timeout=req.timeout_sec,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=req.timeout_sec)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="llama-cli timed out")
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail=f"llama-cli not found at: {args.llama_cli}")

    return {
        "return_code": proc.returncode,
        "stdout":      stdout.decode(errors="replace"),
        "stderr":      stderr.decode(errors="replace"),
        "command":     cmd,
    }


# ── Background stale-device reaper ────────────────────────────────────────────

@app.on_event("startup")
async def start_reaper():
    async def reap():
        while True:
            await asyncio.sleep(args.stale_sec)
            stale = [k for k, v in devices.items() if not v.is_alive(args.stale_sec)]
            for k in stale:
                devices.pop(k, None)
            if stale:
                print(f"[reaper] removed {len(stale)} stale device(s): {stale}")
    asyncio.create_task(reap())


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Starting GGML RPC device registry on {args.host}:{args.port}")
    print(f"  llama-cli : {args.llama_cli}")
    print(f"  model     : {args.model or '(set via /api/v1/inference/run)'}")
    print(f"  stale-sec : {args.stale_sec}")
    uvicorn.run(app, host=args.host, port=args.port)

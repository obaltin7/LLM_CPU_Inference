"""Run metadata: everything needed to audit and reproduce one measurement session.

One JSON file per run is written to results/metadata/<run_id>.json; every CSV row carries the
same run_id, so each number can be traced back to the hardware state, software versions,
configuration and model files that produced it.
"""
import ctypes
import hashlib
import json
import os
import platform
import re
import struct
import subprocess
import sys
import uuid
from datetime import datetime

import psutil

from llm_cpu_inference import PROJECT_ROOT
from llm_cpu_inference.models import MODELS, MODELS_DIR, model_path, quantization_of
from llm_cpu_inference.log import get_logger
from llm_cpu_inference.power import power_state
from llm_cpu_inference.results_io import write_json_atomic

METADATA_SCHEMA_VERSION = 2
SHA256_CACHE = os.path.join(MODELS_DIR, ".sha256_cache.json")  # models/ is git-ignored
_dirty_warned = False

_HARDWARE_QUERY = (
    "[Console]::OutputEncoding=[Text.Encoding]::UTF8; "
    "[pscustomobject]@{"
    "cpu=@(Get-CimInstance Win32_Processor | Select-Object Name,Manufacturer,Description,NumberOfCores,"
    "NumberOfLogicalProcessors,MaxClockSpeed,L2CacheSize,L3CacheSize);"
    "memory=@(Get-CimInstance Win32_PhysicalMemory | Select-Object DeviceLocator,Manufacturer,PartNumber,"
    "Capacity,Speed,ConfiguredClockSpeed,ConfiguredVoltage);"
    "gpu=@(Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion);"
    "os=@(Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,BuildNumber);"
    "board=@(Get-CimInstance Win32_BaseBoard | Select-Object Manufacturer,Product)"
    "} | ConvertTo-Json -Depth 4 -Compress"
)


def now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def new_run_id():
    """Sortable identifier: local start time plus a short random suffix."""
    return f"{datetime.now():%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}"


def _run(command, encoding="utf-8", timeout=60):
    """Run a command and return its stdout, or None (with the reason logged) if it fails."""
    try:
        completed = subprocess.run(command, capture_output=True, timeout=timeout, cwd=PROJECT_ROOT)
    except (OSError, subprocess.TimeoutExpired) as exc:
        get_logger().warning("Command %s failed: %s", command[0], exc)
        return None
    if completed.returncode != 0:
        get_logger().warning("Command %s exited with code %d", command[0], completed.returncode)
        return None
    return completed.stdout.decode(encoding, errors="replace").strip()


def detect_topology():
    """Physical/logical core counts and the P-core/E-core split (hybrid CPUs).

    Uses GetLogicalProcessorInformationEx(RelationProcessorCore): each record is one physical
    core with its EfficiencyClass. More than one distinct class means a hybrid design; the
    highest class is the performance (P) cores.
    """
    topology = {
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_processors": psutil.cpu_count(logical=True),
        "smt": None, "hybrid": None, "p_cores": None, "e_cores": None,
        "cores_per_efficiency_class": None, "source": "psutil",
    }
    if not sys.platform.startswith("win"):
        return topology
    try:
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        get_info = kernel32.GetLogicalProcessorInformationEx
        get_info.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD)]
        get_info.restype = wintypes.BOOL

        size = wintypes.DWORD(0)
        get_info(0, None, ctypes.byref(size))  # RelationProcessorCore = 0; first call returns the size
        buffer = ctypes.create_string_buffer(size.value)
        if not get_info(0, buffer, ctypes.byref(size)):
            raise OSError(ctypes.get_last_error(), "GetLogicalProcessorInformationEx failed")

        raw = buffer.raw[:size.value]
        cores = []  # (efficiency_class, logical processors on the core)
        offset = 0
        while offset < len(raw):
            _relationship, record_size = struct.unpack_from("<II", raw, offset)
            # PROCESSOR_RELATIONSHIP: BYTE Flags; BYTE EfficiencyClass; BYTE Reserved[20];
            # WORD GroupCount; GROUP_AFFINITY GroupMask[] (16 bytes each, KAFFINITY first)
            _flags, efficiency = struct.unpack_from("<BB", raw, offset + 8)
            (group_count,) = struct.unpack_from("<H", raw, offset + 30)
            logical = 0
            for group in range(group_count):
                (mask,) = struct.unpack_from("<Q", raw, offset + 32 + 16 * group)
                logical += bin(mask).count("1")
            cores.append((efficiency, logical))
            offset += record_size

        classes = {}
        for efficiency, _ in cores:
            classes[efficiency] = classes.get(efficiency, 0) + 1
        hybrid = len(classes) > 1
        p_cores = classes[max(classes)] if hybrid else len(cores)
        topology.update({
            "physical_cores": len(cores),
            "logical_processors": sum(logical for _, logical in cores),
            "smt": any(logical > 1 for _, logical in cores),
            "hybrid": hybrid,
            "p_cores": p_cores,
            "e_cores": len(cores) - p_cores,
            "cores_per_efficiency_class": {str(k): v for k, v in sorted(classes.items())},
            "source": "GetLogicalProcessorInformationEx",
        })
    except (OSError, struct.error, AttributeError) as exc:
        get_logger().warning("Core topology query failed (%s); hybrid/P-core fields are unknown.", exc)
    return topology


def _first(items):
    if isinstance(items, list):
        return items[0] if items else {}
    return items or {}


def query_hardware():
    """CPU model, memory modules (with the configured speed), GPU, OS and board via CIM."""
    hardware = {"cpu": None, "memory": None, "gpu": None, "os": None, "board": None}
    if not sys.platform.startswith("win"):
        hardware["cpu"] = {"model": platform.processor()}
        return hardware
    output = _run(["powershell", "-NoProfile", "-NonInteractive", "-Command", _HARDWARE_QUERY])
    if not output:
        get_logger().warning("Hardware inventory (CIM) is unavailable; hardware fields are null.")
        return hardware
    try:
        data = json.loads(output)
    except json.JSONDecodeError as exc:
        get_logger().warning("Hardware inventory could not be parsed: %s", exc)
        return hardware

    cpu = _first(data.get("cpu"))
    hardware["cpu"] = {
        "model": (cpu.get("Name") or "").strip(),
        "manufacturer": cpu.get("Manufacturer"),
        "family_model_stepping": cpu.get("Description"),
        "nominal_mhz": cpu.get("MaxClockSpeed"),
        "l2_cache_kb": cpu.get("L2CacheSize"),
        "l3_cache_kb": cpu.get("L3CacheSize"),
    }
    modules = [{
        "slot": m.get("DeviceLocator"),
        "manufacturer": (m.get("Manufacturer") or "").strip(),
        "part_number": (m.get("PartNumber") or "").strip(),
        "capacity_gb": round(int(m.get("Capacity") or 0) / 2 ** 30, 2),
        "rated_speed_mhz": m.get("Speed"),
        "configured_speed_mhz": m.get("ConfiguredClockSpeed"),
        "configured_voltage_mv": m.get("ConfiguredVoltage"),
    } for m in data.get("memory") or []]
    speeds = sorted({m["configured_speed_mhz"] for m in modules if m["configured_speed_mhz"]})
    hardware["memory"] = {
        "total_gb": round(psutil.virtual_memory().total / 2 ** 30, 2),
        "modules": modules,
        # The speed the modules actually run at (XMP off => JEDEC speed), not the rated speed
        "configured_speed_mhz": speeds[0] if len(speeds) == 1 else speeds or None,
    }
    hardware["gpu"] = [{"name": g.get("Name"), "driver": g.get("DriverVersion")} for g in data.get("gpu") or []]
    os_info = _first(data.get("os"))
    hardware["os"] = {"caption": os_info.get("Caption"), "version": os_info.get("Version"),
                      "build": os_info.get("BuildNumber")}
    board = _first(data.get("board"))
    hardware["board"] = {"manufacturer": board.get("Manufacturer"), "product": board.get("Product")}
    return hardware


def git_info():
    """Commit of the code that is running. `dirty` ignores results/, which the campaign itself writes."""
    commit = _run(["git", "rev-parse", "HEAD"])
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    diff = _run(["git", "diff", "--name-only", "HEAD"])  # Tracked files that differ from the commit
    modified = None
    if diff is not None:
        modified = [path for path in diff.splitlines() if path.strip() and not path.startswith("results/")]
    global _dirty_warned
    if modified and not _dirty_warned:
        _dirty_warned = True  # Once per process; the state is also stored in every metadata file
        get_logger().warning("Working tree has uncommitted changes in tracked files: %s. The git commit in "
                             "the metadata does not fully describe the running code.", ", ".join(modified))
    return {"commit": commit, "branch": branch, "dirty": bool(modified) if modified is not None else None,
            "modified_tracked_files": modified}


def software_versions():
    import codecarbon
    import llama_cpp
    import numpy
    import yaml

    return {
        "python": platform.python_version(),
        "llama_cpp_python": llama_cpp.__version__,
        "codecarbon": codecarbon.__version__,
        "numpy": numpy.__version__,
        "psutil": psutil.__version__,
        "pyyaml": yaml.__version__,
        "llama_cpp_system_info": llama_cpp.llama_print_system_info().decode("utf-8", errors="replace").strip(),
        "llama_cpp_gpu_offload_supported": bool(llama_cpp.llama_supports_gpu_offload()),
    }


def file_sha256(path):
    """SHA256 of a model file, cached by (size, mtime) so each file is hashed only once."""
    stat = os.stat(path)
    name = os.path.basename(path)
    cache = {}
    if os.path.exists(SHA256_CACHE):
        try:
            with open(SHA256_CACHE, encoding="utf-8") as f:
                cache = json.load(f)
        except (OSError, json.JSONDecodeError):
            cache = {}
    entry = cache.get(name)
    if entry and entry.get("size") == stat.st_size and entry.get("mtime_ns") == stat.st_mtime_ns:
        return entry["sha256"]

    get_logger().info("Hashing %s (%.2f GB, first time only)...", name, stat.st_size / 1e9)
    with open(path, "rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    cache[name] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest}
    write_json_atomic(SHA256_CACHE, cache)
    return digest


def model_record(key):
    info = MODELS[key]
    path = model_path(key)
    return {
        "filename": info["filename"],
        "repo_id": info["repo_id"],
        "method": info["method"],
        "lineage": info.get("lineage"),
        "params_b": info["params_b"],
        "is_base_model": info["base"],
        "quantization": quantization_of(info["filename"]),
        "size_bytes": os.path.getsize(path),
        "sha256": file_sha256(path),
    }


def collect_run_metadata(run_id, experiments, config, model_keys, topology, resolved_threads, session=None):
    """Static part of the metadata, collected once at the start of a run (one invocation)."""
    inference = config["inference"]
    git = git_info()
    return {
        "schema_version": METADATA_SCHEMA_VERSION,
        "run_id": run_id,
        "session_id": (session or {}).get("session_id"),
        "session": session,
        "status": "running",
        "final_mode": bool(config["_meta"].get("final")),
        "git_dirty": git["dirty"],
        "timestamp_start": now_iso(),
        "timestamp_end": None,
        "experiments": list(experiments),
        "smoke": bool(config["_meta"]["smoke"]),
        "git": git,
        "hardware": {**query_hardware(), "topology": topology, "power": power_state()},
        "software": software_versions(),
        "inference": {
            "n_ctx": inference["n_ctx"],
            "seed": inference["seed"],
            "n_gpu_layers": 0,
            "use_mmap": inference["use_mmap"],
            "decoding": "greedy (argmax), end-of-generation tokens suppressed for a fixed output length",
            "n_threads": resolved_threads,
        },
        "models": {key: model_record(key) for key in model_keys},
        "config": {k: v for k, v in config.items() if k != "_meta"},
        "config_file": config["_meta"],
        "energy_tracking": None,
        "frequency_tracking": None,
        "thermal": None,
        "session_start": None,
        "warmup_load": None,
        "executed_units": [],
        "idle_baseline": None,
        "carbon": None,
    }


def write_run_metadata(metadata_dir, metadata):
    path = os.path.join(metadata_dir, f"{metadata['run_id']}.json")
    write_json_atomic(path, metadata)
    return path

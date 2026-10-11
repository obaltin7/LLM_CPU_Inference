"""Loads configs/experiment.yaml, the single source of every experiment parameter."""
import copy
import hashlib
import json
import os

import yaml

from llm_cpu_inference import PROJECT_ROOT
from llm_cpu_inference.log import get_logger
from llm_cpu_inference.models import MODELS, available_models

DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "experiment.yaml")

# Symbolic thread counts, resolved against the detected core topology at run time
THREAD_ALIASES = ("physical", "p_cores", "logical")
MIN_PROMPTS = 3


def _deep_merge(base, override):
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path=None, smoke=False, final=False):
    """Read the YAML file; with smoke=True the `smoke:` block overrides the main values.

    final=True marks a final (thesis) run: it cannot be combined with smoke, and the session
    refuses to start unless every pre-condition holds (clean git tree, expected power plan,
    real temperature sensor).
    """
    if smoke and final:
        raise ValueError("--final and --smoke cannot be combined")
    path = os.path.abspath(path or DEFAULT_CONFIG_PATH)
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    config = yaml.safe_load(raw)
    smoke_overrides = config.pop("smoke", None) or {}
    if smoke:
        config = _deep_merge(config, smoke_overrides)
    config["_meta"] = {
        "path": os.path.relpath(path, PROJECT_ROOT).replace("\\", "/"),
        "sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "smoke": bool(smoke),
        "final": bool(final),
    }
    _validate(config)
    return config


def _validate(config):
    for section in ("paths", "campaign", "power", "models", "inference", "prompts", "energy", "thread_scaling",
                    "cognitive", "thermal", "idle", "carbon", "heat_test", "session_start", "warmup_load"):
        if section not in config:
            raise ValueError(f"Config section '{section}' is missing")
    unknown = [m for m in config["models"] + config["thread_scaling"]["models"] if m not in MODELS]
    if unknown:
        raise ValueError(f"Unknown model keys in config: {unknown} (known: {list(MODELS)})")
    prompt_ids = [p["id"] for p in config["prompts"]]
    if len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError("Prompt ids must be unique")
    if len(prompt_ids) < MIN_PROMPTS and not config["_meta"]["smoke"]:
        raise ValueError(f"At least {MIN_PROMPTS} fixed prompts are required for final runs, "
                         f"got {len(prompt_ids)}")


def unit_fingerprint(config, experiment):
    """Hash of the parameters that define how a unit of `experiment` is measured.

    A session records the fingerprint of each experiment when it is created; resuming the
    session with different measurement parameters is refused, so one session never mixes two
    protocols. Model and thread lists are left out on purpose: the realised unit list is stored
    in the session itself. Thermal safety limits are left out too: tightening them must not
    invalidate measured data.
    """
    section = {k: v for k, v in config[experiment].items() if k not in ("models", "threads")}
    relevant = {"experiment": section, "inference": config["inference"]}
    if experiment != "cognitive":
        relevant.update({key: config[key] for key in ("prompts", "idle", "carbon")})
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def resolve_thread_count(value, topology):
    """Turn an integer or one of THREAD_ALIASES into a thread count."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if value == "physical":
        return topology["physical_cores"]
    if value == "logical":
        return topology["logical_processors"]
    if value == "p_cores":
        # On a non-hybrid CPU every core is a performance core
        return topology["p_cores"] or topology["physical_cores"]
    raise ValueError(f"Invalid thread count '{value}' (use an integer or one of {THREAD_ALIASES})")


def resolve_thread_list(spec, topology):
    """Resolve the configured thread list: aliases expanded, duplicates removed, sorted, capped."""
    logical = topology["logical_processors"]
    resolved = sorted({resolve_thread_count(item, topology) for item in spec})
    dropped = [n for n in resolved if n > logical or n < 1]
    if dropped:
        get_logger().warning("Thread counts %s are outside 1..%d logical processors and are skipped.",
                             dropped, logical)
    return [n for n in resolved if 1 <= n <= logical]


def select_models(configured, requested=None):
    """Models to run: the configured list (or a requested subset) restricted to files on disk."""
    wanted = list(requested) if requested else list(configured)
    present = available_models(wanted)
    for key in wanted:
        if key not in present:
            get_logger().error("Model file for '%s' is missing (%s); run 'python scripts/download_models.py'. "
                               "It is skipped.", key, MODELS[key]["filename"])
    return present


def result_paths(config, results_dir=None):
    """Locations of the Phase 2 result files (new schema; Phase 1 lives in results/phase1/)."""
    root = os.path.abspath(results_dir or os.path.join(PROJECT_ROOT, config["paths"]["results_dir"]))
    return {
        "root": root,
        "metadata": os.path.join(root, "metadata"),
        "nll": os.path.join(root, "nll"),
        "logs": os.path.join(root, "logs"),
        "energy_iterations": os.path.join(root, "energy_iterations.csv"),
        "energy_summary": os.path.join(root, "energy_summary.csv"),
        "thread_scaling_iterations": os.path.join(root, "thread_scaling_iterations.csv"),
        "thread_scaling_summary": os.path.join(root, "thread_scaling_summary.csv"),
        "cognitive": os.path.join(root, "cognitive_results.csv"),
        "cognitive_full": os.path.join(root, "cognitive_results_full.csv"),  # merged full-set rows (night_run a2_rest)
        "state": os.path.join(root, "campaign_state.json"),
    }

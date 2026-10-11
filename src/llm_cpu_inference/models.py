"""Model registry of the GGUF files used in the experiments, and the download helper behind
scripts/download_models.py."""
import os
import re

from huggingface_hub import hf_hub_download

from llm_cpu_inference import PROJECT_ROOT

# models/ at the repository root is used regardless of the working directory
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Model registry. The key is the short name used by every script and by configs/experiment.yaml.
#   method   : compression applied to obtain the model (the comparison axis)
#   lineage  : the model family the file derives from (see the model cards)
#   params_b : parameter count in billions
#   base     : True for a base model without instruct fine-tuning (matters when comparing perplexity)
MODELS = {
    # Quantization only: dense Llama-3-8B-Instruct
    "8B-Q4_K_M": {
        "repo_id": "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF",
        "filename": "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
        "method": "quantization", "params_b": 8.0, "base": False,
        "lineage": "Llama-3-8B-Instruct (dense)",
    },
    # Intermediate quantization levels of the same converter/repository (night run, Pareto gap Q4 <-> Q8)
    "8B-Q5_K_M": {
        "repo_id": "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF",
        "filename": "Meta-Llama-3-8B-Instruct.Q5_K_M.gguf",
        "method": "quantization", "params_b": 8.0, "base": False,
    },
    "8B-Q6_K": {
        "repo_id": "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF",
        "filename": "Meta-Llama-3-8B-Instruct.Q6_K.gguf",
        "method": "quantization", "params_b": 8.0, "base": False,
    },
    "8B-Q8_0": {
        "repo_id": "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF",
        "filename": "Meta-Llama-3-8B-Instruct.Q8_0.gguf",
        "method": "quantization", "params_b": 8.0, "base": False,
        "lineage": "Llama-3-8B-Instruct (dense)",
    },
    # Structured pruning + knowledge distillation: Meta produced Llama-3.2 1B/3B by pruning and then
    # distilling with logits from Llama-3.1 8B and 70B. The two methods cannot be separated.
    "3B-Q4_K_M": {
        "repo_id": "QuantFactory/Llama-3.2-3B-Instruct-GGUF",
        "filename": "Llama-3.2-3B-Instruct.Q4_K_M.gguf",
        "method": "pruning+distillation", "params_b": 3.2, "base": False,
        "lineage": "Llama-3.2-3B-Instruct (pruned and distilled by Meta; teachers: Llama-3.1 8B/70B)",
    },
    "3B-Q8_0": {
        "repo_id": "QuantFactory/Llama-3.2-3B-Instruct-GGUF",
        "filename": "Llama-3.2-3B-Instruct.Q8_0.gguf",
        "method": "pruning+distillation", "params_b": 3.2, "base": False,
        "lineage": "Llama-3.2-3B-Instruct (pruned and distilled by Meta; teachers: Llama-3.1 8B/70B)",
    },
    "1B-Q4_K_M": {
        "repo_id": "QuantFactory/Llama-3.2-1B-Instruct-GGUF",
        "filename": "Llama-3.2-1B-Instruct.Q4_K_M.gguf",
        "method": "pruning+distillation", "params_b": 1.2, "base": False,
        "lineage": "Llama-3.2-1B-Instruct (pruned and distilled by Meta; teachers: Llama-3.1 8B/70B)",
    },
    "1B-Q8_0": {
        "repo_id": "QuantFactory/Llama-3.2-1B-Instruct-GGUF",
        "filename": "Llama-3.2-1B-Instruct.Q8_0.gguf",
        "method": "pruning+distillation", "params_b": 1.2, "base": False,
        "lineage": "Llama-3.2-1B-Instruct (pruned and distilled by Meta; teachers: Llama-3.1 8B/70B)",
    },
    # Semi-structured 2:4 sparsity (50 % of the weights in the transformer blocks are zero), obtained
    # with SparseGPT and then recovered with SquareHead knowledge distillation (13B tokens), so this is
    # not pruning alone. GGUF stores the zeros densely and llama.cpp multiplies densely: no speed or
    # energy gain is expected. It is a base model: its perplexity is not directly comparable to the
    # instruct models.
    "Sparse8B-Q4_K_M": {
        "repo_id": "QuantFactory/Sparse-Llama-3.1-8B-2of4-GGUF",
        "filename": "Sparse-Llama-3.1-8B-2of4.Q4_K_M.gguf",
        "method": "2:4 sparse pruning+distillation", "params_b": 8.0, "base": True,
        "lineage": "Llama-3.1-8B base (SparseGPT 2:4 + SquareHead distillation, by Neural Magic)",
    },
    "Sparse8B-Q8_0": {
        "repo_id": "QuantFactory/Sparse-Llama-3.1-8B-2of4-GGUF",
        "filename": "Sparse-Llama-3.1-8B-2of4.Q8_0.gguf",
        "method": "2:4 sparse pruning+distillation", "params_b": 8.0, "base": True,
        "lineage": "Llama-3.1-8B base (SparseGPT 2:4 + SquareHead distillation, by Neural Magic)",
    },
}

# Reverse lookup from file name to registry key (used to label result rows)
_KEY_BY_FILENAME = {info["filename"]: key for key, info in MODELS.items()}


def model_path(key):
    return os.path.join(MODELS_DIR, MODELS[key]["filename"])


def available_models(keys=None):
    """Registry keys whose model file exists on disk."""
    return [k for k in (keys or MODELS) if os.path.exists(model_path(k))]


def model_info(filename):
    """Key, method and parameter count for a file name (empty fields if it is not registered)."""
    key = _KEY_BY_FILENAME.get(filename)
    info = MODELS.get(key, {})
    return {"model_key": key or "", "method": info.get("method", ""), "params_b": info.get("params_b", "")}


def quantization_of(filename):
    """Quantization level encoded in the file name (e.g. Q4_K_M)."""
    match = re.search(r"\.(Q\d\w*)\.gguf$", filename)
    return match.group(1) if match else ""


def download_missing_models(keys, force=False):
    """Download the models that are not yet in models/."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    for key in keys:
        info = MODELS[key]
        target_path = model_path(key)

        if os.path.exists(target_path) and not force:
            size_gb = os.path.getsize(target_path) / 1e9
            print(f"Already present, skipping: {info['filename']} ({size_gb:.2f} GB)")
            continue

        # Interrupted downloads resume; the file appears under models/ only once it is complete.
        print(f"Downloading: {info['filename']}...")
        hf_hub_download(
            repo_id=info["repo_id"],
            filename=info["filename"],
            local_dir=MODELS_DIR,
            force_download=force
        )
        print(f"Done: {info['filename']}")

    print("All models are ready.")

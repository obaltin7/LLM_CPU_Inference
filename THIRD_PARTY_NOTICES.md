# Third-Party Components and Licenses

This repository is licensed under the Apache License 2.0 (`LICENSE`). The components below are **not included**
in the repository; they are obtained during installation or download and are subject to their own licenses.

## Software

| Component | Use | License |
|---|---|---|
| llama.cpp / ggml (ggml-org) | Inference engine | MIT |
| llama-cpp-python 0.3.36 (abetlen) | Python bindings, prebuilt CPU wheel | MIT |
| CodeCarbon 3.3.1 | CPU energy (Windows EMI / RAPL) and carbon estimation | MIT |
| LibreHardwareMonitorLib 0.9.6 | CPU package temperature (downloaded into `third_party/` with `scripts/fetch_lhm.py`; not kept in the repository) | MPL-2.0 |
| PawnIO (namazso) | Kernel driver for LibreHardwareMonitor (installed separately) | its own license (see the project page) |
| pythonnet 3.0.5 | Loading the .NET library from Python | MIT |
| gguf 0.19.0 (ggml-org) | Reading GGUF headers | MIT |
| numpy, pandas, matplotlib, psutil, PyYAML, huggingface_hub, python-docx | Analysis, figure and table generation | BSD-3 / BSD-3 / PSF-based / BSD-3 / MIT / Apache-2.0 / MIT |

## Models (Hugging Face, GGUF; downloaded with `scripts/download_models.py`, not kept in the repository)

| Model | Repository | License |
|---|---|---|
| Meta-Llama-3-8B-Instruct (Q4_K_M, Q5_K_M, Q6_K, Q8_0) | QuantFactory/Meta-Llama-3-8B-Instruct-GGUF | Meta Llama 3 Community License |
| Llama-3.2-3B-Instruct (Q4_K_M, Q8_0) | QuantFactory/Llama-3.2-3B-Instruct-GGUF | Llama 3.2 Community License |
| Llama-3.2-1B-Instruct (Q4_K_M, Q8_0) | QuantFactory/Llama-3.2-1B-Instruct-GGUF | Llama 3.2 Community License |
| Sparse-Llama-3.1-8B-2of4 (Q4_K_M, Q8_0; Neural Magic / Red Hat AI) | QuantFactory/Sparse-Llama-3.1-8B-2of4-GGUF | Llama 3.1 Community License |

QuantFactory, which publishes the GGUF conversions, redistributes the model weights under the respective Meta
licenses; the "Built with Llama" attribution obligation stems from these licenses. The file SHA-256 values are
recorded in `results/metadata/<run_id>.json` and in `docs/findings_report.md` §6.1.

## Dataset

| Data | Source | License |
|---|---|---|
| WikiText-2 (raw, test) — `wikitext-2-raw-v1.zip` | ggml-org/ci (Hugging Face); original by Merity et al. 2016 | CC BY-SA 3.0 |

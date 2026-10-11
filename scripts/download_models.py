"""Download the GGUF models that are missing from models/ (see the registry in llm_cpu_inference.models)."""
import argparse

import _paths  # noqa: F401  (makes src/ importable without installing the package)
from llm_cpu_inference.models import MODELS, download_missing_models


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download missing GGUF models from Hugging Face into models/.")
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS),
                        help="Models to download (default: all)")
    parser.add_argument("--force", action="store_true", help="Download again even if the file exists")
    args = parser.parse_args(argv)
    download_missing_models(args.models, force=args.force)


if __name__ == "__main__":
    main()

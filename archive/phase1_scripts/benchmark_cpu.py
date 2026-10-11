import time
import psutil
import os
from llama_cpp import Llama


def get_current_ram_usage():
    """Returns the current RAM usage of the process (in GB)."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024 * 1024)


def benchmark_quantization(model_path, prompt, max_tokens=100):
    print(f"--- Test Starting: {os.path.basename(model_path)} ---")

    # 1. RAM before the model is loaded
    ram_before = get_current_ram_usage()

    # Initialise the model on the CPU
    print("Loading model into memory...")
    llm = Llama(
        model_path=model_path,
        n_ctx=512,
        seed=42,  # Seed fixed so that the outputs are identical in every test
        use_mmap=False,  # <--- ADDED LINE: disables memory mapping, forces the model into RAM
        verbose=False
    )

    # 2. RAM after the model is loaded (peak)
    ram_after = get_current_ram_usage()
    model_ram_cost = ram_after - ram_before

    print(f"System Memory Cost (RAM): {model_ram_cost:.2f} GB")

    # 3. Inference and speed test
    print("Starting text generation...")
    start_time = time.time()

    output = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=0.0,  # Creativity fully disabled (greedy decoding)
        echo=False
    )

    end_time = time.time()

    # Computing the metrics
    generation_time = end_time - start_time
    generated_tokens = output['usage']['completion_tokens']
    tokens_per_sec = generated_tokens / generation_time

    print(f"Generated Tokens: {generated_tokens}")
    print(f"Elapsed Time: {generation_time:.2f} seconds")
    print(f"Speed (Tokens/sec): {tokens_per_sec:.2f} tokens/s")
    print("-" * 50 + "\n")

    return {
        "model": os.path.basename(model_path),
        "ram_gb": model_ram_cost,
        "tokens_per_sec": tokens_per_sec
    }


if __name__ == "__main__":
    # Add the file paths of the downloaded models here
    models_to_test = [
        "./models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
        "./models/Meta-Llama-3-8B-Instruct.Q8_0.gguf"
    ]

    test_prompt = "Explain the architecture of a CPU in detail."

    results = []
    for model in models_to_test:
        if os.path.exists(model):
            res = benchmark_quantization(model, test_prompt)
            results.append(res)
        else:
            print(f"Error: {model} not found. Please check the file path.")

    # Tabulating the results (for thesis reporting)
    print("=== COMPARATIVE RESULTS ===")
    for r in results:
        print(f"Model: {r['model']} | RAM Usage: {r['ram_gb']:.2f} GB | Speed: {r['tokens_per_sec']:.2f} t/s")
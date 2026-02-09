#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TTS Benchmark Script

Benchmarks TTS models with configurable parameters and saves results to JSON.

Usage:
    python benchmarks/run_benchmark.py --description "torch.compile optimization test"
    python benchmarks/run_benchmark.py --description "baseline" --models 1.7B
    python benchmarks/run_benchmark.py --help
"""
import argparse
import json
import requests
import time
import sys
import subprocess
from datetime import datetime
from pathlib import Path

# Configuration
API_URL = "http://localhost:7860"
RESULTS_DIR = Path(__file__).parent / "results"

# Russian test texts of varying lengths
TEST_CASES = [
    ("short", "Привет, это быстрый тест системы."),
    ("medium", "Искусственный интеллект меняет наш мир каждый день. Голосовые помощники становятся умнее и полезнее."),
    ("long", "Технологии синтеза речи достигли невероятного уровня качества. Современные нейронные сети способны генерировать голос, практически неотличимый от человеческого. Это открывает множество возможностей для создания контента."),
]


def estimate_audio_duration(text: str) -> float:
    """Estimate audio duration based on word count (~2 words/sec for Russian)"""
    words = len(text.split())
    return words / 2.0


def get_gpu_memory() -> dict:
    """Get GPU memory usage using nvidia-smi"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            if len(parts) >= 3:
                return {
                    "used_mb": int(parts[0].strip()),
                    "total_mb": int(parts[1].strip()),
                    "free_mb": int(parts[2].strip()),
                }
    except Exception:
        pass
    return {"used_mb": None, "total_mb": None, "free_mb": None}


def check_server() -> dict:
    """Check server health and return status info"""
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.json()
    except Exception as e:
        return {"status": "error", "error": str(e)}


def get_models_status() -> dict:
    """Get status of all models"""
    try:
        r = requests.get(f"{API_URL}/models/status", timeout=5)
        return r.json().get("models", {})
    except Exception:
        return {}


def load_model(model_id: str, warmup: bool = False, warmup_lang: str = None,
               warmup_voice: str = None, quantization: str = "none",
               enable_optimizations: bool = True, torch_compile: bool = True,
               cuda_graphs: bool = True, compile_codebook: bool = True,
               fast_codebook: bool = True, attention: str = "auto") -> tuple[bool, str]:
    """Load model with optional warmup, voice, quantization, and optimizations"""
    url = f"{API_URL}/models/{model_id}/load"
    params = []
    if warmup:
        params.append("warmup=batch")
        if warmup_lang:
            params.append(f"warmup_lang={warmup_lang}")
        if warmup_voice:
            params.append(f"warmup_voice={warmup_voice}")
    else:
        params.append("warmup=none")
    if quantization and quantization != "none":
        params.append(f"quantization={quantization}")
    params.append(f"enable_optimizations={'true' if enable_optimizations else 'false'}")
    params.append(f"torch_compile={'true' if torch_compile else 'false'}")
    params.append(f"cuda_graphs={'true' if cuda_graphs else 'false'}")
    params.append(f"compile_codebook={'true' if compile_codebook else 'false'}")
    params.append(f"fast_codebook={'true' if fast_codebook else 'false'}")
    params.append(f"attention={attention}")
    if params:
        url += "?" + "&".join(params)
    try:
        r = requests.post(url, timeout=600)
        return r.status_code == 200, r.json().get("message", "")
    except Exception as e:
        return False, str(e)


def unload_model(model_id: str) -> bool:
    """Unload model"""
    try:
        r = requests.post(f"{API_URL}/models/{model_id}/unload", timeout=60)
        return r.status_code == 200
    except Exception:
        return False


def run_synthesis(text: str, voice: str, model: str, language: str = "Russian",
                  do_sample: bool = False) -> tuple[float, bool, str]:
    """Run a single synthesis and return (time, success, error)"""
    start = time.time()
    try:
        r = requests.post(
            f"{API_URL}/synthesize_speech/",
            json={
                "text": text,
                "voice": voice,
                "language": language,
                "model": model,
                "do_sample": do_sample,
            },
            timeout=300
        )
        elapsed = time.time() - start
        if r.status_code == 200:
            return elapsed, True, ""
        return elapsed, False, f"HTTP {r.status_code}"
    except Exception as e:
        return time.time() - start, False, str(e)


def benchmark_model(model_id: str, voice: str, language: str = "Russian",
                    runs: int = 3, do_sample: bool = False) -> tuple[list[dict], dict]:
    """Benchmark a single model with all test cases. Returns (results, memory_stats)"""
    results = []
    peak_memory_mb = 0

    # Get initial memory
    initial_mem = get_gpu_memory()

    for case_name, text in TEST_CASES:
        words = len(text.split())
        print(f"  Testing: {case_name} ({words} words)")

        times = []
        for run in range(runs):
            elapsed, success, error = run_synthesis(text, voice, model_id, language=language, do_sample=do_sample)
            if success:
                times.append(elapsed)
                # Check memory after each run
                mem = get_gpu_memory()
                if mem["used_mb"] and mem["used_mb"] > peak_memory_mb:
                    peak_memory_mb = mem["used_mb"]
                print(f"    Run {run+1}: {elapsed:.2f}s")
            else:
                print(f"    Run {run+1}: ERROR - {error}")

        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            est_duration = estimate_audio_duration(text)
            rtf = min_time / est_duration if est_duration > 0 else 0

            results.append({
                "case": case_name,
                "words": words,
                "runs": len(times),
                "times": times,
                "avg_time": round(avg_time, 3),
                "min_time": round(min_time, 3),
                "rtf": round(rtf, 3),
            })
            print(f"    => Avg: {avg_time:.2f}s, Min: {min_time:.2f}s, RTF: {rtf:.2f}")
        else:
            results.append({
                "case": case_name,
                "words": words,
                "runs": 0,
                "times": [],
                "avg_time": None,
                "min_time": None,
                "rtf": None,
                "error": "All runs failed"
            })

    memory_stats = {
        "peak_used_mb": peak_memory_mb,
        "total_mb": initial_mem.get("total_mb"),
    }

    return results, memory_stats


def run_benchmark(description: str, models: list[str], voice: str,
                  language: str = "Russian", runs: int = 3, do_sample: bool = False,
                  warmup: bool = False, warmup_lang: str = None, quantization: str = "none",
                  enable_optimizations: bool = True, torch_compile: bool = True,
                  cuda_graphs: bool = True, compile_codebook: bool = True,
                  fast_codebook: bool = True, attention: str = "auto") -> dict:
    """Run full benchmark and return results"""

    print("=" * 70)
    print("TTS Benchmark")
    print(f"Description: {description}")
    print(f"Voice: {voice} | Language: {language} | Models: {models} | do_sample: {do_sample}")
    print(f"optimizations: {enable_optimizations} | compile: {torch_compile} | cuda_graphs: {cuda_graphs}")
    print(f"compile_codebook: {compile_codebook} | fast_codebook: {fast_codebook} | attention: {attention}")
    if quantization != "none":
        print(f"Quantization: {quantization}")
    if warmup and warmup_lang:
        print(f"Warmup language: {warmup_lang}")
    print("=" * 70)

    # Check server
    health = check_server()
    if health.get("status") != "healthy":
        print(f"Server not healthy: {health}")
        sys.exit(1)
    print(f"Server status: {health.get('status')}")

    # Get GPU info
    gpu_mem = get_gpu_memory()
    print(f"GPU memory: {gpu_mem.get('used_mb', '?')}MB / {gpu_mem.get('total_mb', '?')}MB")

    # Build results structure
    results = {
        "description": description,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "voice": voice,
            "language": language,
            "models": models,
            "runs_per_test": runs,
            "do_sample": do_sample,
            "warmup": warmup,
            "warmup_lang": warmup_lang,
            "quantization": quantization,
            "enable_optimizations": enable_optimizations,
            "torch_compile": torch_compile,
            "cuda_graphs": cuda_graphs,
            "compile_codebook": compile_codebook,
            "fast_codebook": fast_codebook,
            "attention": attention,
            "test_cases": [{"name": name, "words": len(text.split())} for name, text in TEST_CASES],
        },
        "gpu": {
            "total_memory_mb": gpu_mem.get("total_mb"),
            "initial_used_mb": gpu_mem.get("used_mb"),
        },
        "models": {}
    }

    for model_id in models:
        print()
        print(f"{'='*70}")
        print(f"MODEL: {model_id}")
        print(f"{'='*70}")

        # Load model
        # Use benchmark voice for warmup to ensure CUDA graphs match actual usage
        warmup_voice = voice if warmup else None
        print(f"Loading {model_id} (warmup={warmup}, lang={warmup_lang or 'default'}, voice={warmup_voice or 'silence'}, quant={quantization}, opts={enable_optimizations}, attention={attention})...")
        load_start = time.time()
        success, msg = load_model(model_id, warmup=warmup, warmup_lang=warmup_lang,
                                   warmup_voice=warmup_voice, quantization=quantization,
                                   enable_optimizations=enable_optimizations,
                                   torch_compile=torch_compile, cuda_graphs=cuda_graphs,
                                   compile_codebook=compile_codebook,
                                   fast_codebook=fast_codebook, attention=attention)
        load_time = time.time() - load_start

        if not success:
            print(f"Failed to load {model_id}: {msg}")
            results["models"][model_id] = {"error": f"Failed to load: {msg}"}
            continue

        # Get memory after loading
        mem_after_load = get_gpu_memory()
        model_memory_mb = mem_after_load.get("used_mb", 0)
        print(f"Loaded in {load_time:.1f}s: {msg}")
        print(f"GPU memory after load: {model_memory_mb}MB")
        print()

        # Run benchmarks
        model_results, memory_stats = benchmark_model(model_id, voice, language=language, runs=runs, do_sample=do_sample)

        # Calculate overall RTF
        valid_results = [r for r in model_results if r.get("rtf") is not None]
        overall_rtf = sum(r["rtf"] for r in valid_results) / len(valid_results) if valid_results else None

        results["models"][model_id] = {
            "load_time": round(load_time, 2),
            "memory_after_load_mb": model_memory_mb,
            "peak_memory_mb": memory_stats.get("peak_used_mb"),
            "results": model_results,
            "overall_rtf": round(overall_rtf, 3) if overall_rtf else None,
        }

        print(f"\nMemory: loaded={model_memory_mb}MB, peak={memory_stats.get('peak_used_mb')}MB")

        # Unload model
        print(f"\nUnloading {model_id}...")
        unload_model(model_id)

    return results


def print_summary(results: dict):
    """Print benchmark summary"""
    print()
    print("=" * 70)
    print("SUMMARY")
    print(f"Description: {results['description']}")
    print(f"Attention: {results['config'].get('attention', 'auto')}")
    print("=" * 70)

    for model_id, model_data in results["models"].items():
        if "error" in model_data:
            print(f"\n{model_id}: ERROR - {model_data['error']}")
            continue

        print(f"\n{model_id}:")
        mem_load = model_data.get("memory_after_load_mb", "?")
        mem_peak = model_data.get("peak_memory_mb", "?")
        print(f"  GPU Memory: {mem_load}MB (loaded) / {mem_peak}MB (peak)")
        print(f"  {'Case':<10} {'Words':<8} {'Avg(s)':<10} {'Min(s)':<10} {'RTF':<8}")
        print(f"  {'-'*46}")

        for r in model_data["results"]:
            if r.get("avg_time") is not None:
                print(f"  {r['case']:<10} {r['words']:<8} {r['avg_time']:<10.2f} {r['min_time']:<10.2f} {r['rtf']:<8.2f}")
            else:
                print(f"  {r['case']:<10} {r['words']:<8} {'FAILED':<10}")

        overall_rtf = model_data.get("overall_rtf")
        if overall_rtf:
            print(f"  {'-'*46}")
            print(f"  Overall RTF: {overall_rtf:.2f}")
            if overall_rtf < 1.0:
                print(f"  => Faster than real-time ({(1-overall_rtf)*100:.1f}% faster)")
            else:
                print(f"  => Slower than real-time ({(overall_rtf-1)*100:.1f}% slower)")

    print()
    print("=" * 70)


def save_results(results: dict, output_path: Path = None, label: str = None) -> Path:
    """Save results to JSON file"""
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if label:
            # Sanitize label for filename (replace spaces with underscores, remove special chars)
            safe_label = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)
            output_path = RESULTS_DIR / f"benchmark_{timestamp}_{safe_label}.json"
        else:
            output_path = RESULTS_DIR / f"benchmark_{timestamp}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="TTS Benchmark Script")
    parser.add_argument(
        "--description", "-d",
        required=True,
        help="Description of this benchmark run (saved in results)"
    )
    parser.add_argument(
        "--models", "-m",
        nargs="+",
        default=["0.6B", "1.7B"],
        help="Models to benchmark (default: 0.6B 1.7B)"
    )
    parser.add_argument(
        "--voice", "-v",
        default="ponda",
        help="Voice to use (default: ponda)"
    )
    parser.add_argument(
        "--language",
        default="Russian",
        help="Language for synthesis (default: Russian)"
    )
    parser.add_argument(
        "--runs", "-r",
        type=int,
        default=3,
        help="Number of runs per test case (default: 3)"
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable sampling (default: disabled for deterministic output)"
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Enable warmup when loading models"
    )
    parser.add_argument(
        "--warmup-lang", "-w",
        default="Russian",
        help="Language for warmup phrases when --warmup is enabled (default: Russian)"
    )
    parser.add_argument(
        "--quantization", "-q",
        default="none",
        choices=["none", "int8", "float8"],
        help="Quantization mode (default: none)"
    )
    parser.add_argument(
        "--no-optimizations",
        action="store_true",
        help="Disable all streaming optimizations (compile, CUDA graphs, codebook)"
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile on decoder (enabled by default)"
    )
    parser.add_argument(
        "--no-cuda-graphs",
        action="store_true",
        help="Disable CUDA graphs (enabled by default)"
    )
    parser.add_argument(
        "--no-compile-codebook",
        action="store_true",
        help="Disable torch.compile on codebook predictor (enabled by default)"
    )
    parser.add_argument(
        "--no-fast-codebook",
        action="store_true",
        help="Disable fast codebook generation (enabled by default)"
    )
    parser.add_argument(
        "--attention", "-a",
        default="auto",
        choices=["auto", "sage_attn", "flex_attn", "flash2_attn", "sdpa", "eager"],
        help="Attention implementation (default: auto)"
    )
    parser.add_argument(
        "--label", "-l",
        help="Optional label to append to result filename"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output file path (default: benchmarks/results/benchmark_<timestamp>.json)"
    )

    args = parser.parse_args()

    # Run benchmark
    results = run_benchmark(
        description=args.description,
        models=args.models,
        voice=args.voice,
        language=args.language,
        runs=args.runs,
        do_sample=args.do_sample,
        warmup=args.warmup,
        warmup_lang=args.warmup_lang if args.warmup else None,
        quantization=args.quantization,
        enable_optimizations=not args.no_optimizations,
        torch_compile=not args.no_compile,
        cuda_graphs=not args.no_cuda_graphs,
        compile_codebook=not args.no_compile_codebook,
        fast_codebook=not args.no_fast_codebook,
        attention=args.attention,
    )

    # Print summary
    print_summary(results)

    # Save results
    output_path = save_results(results, args.output, label=args.label)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()

"""
High-Throughput Concurrency & Prefix-Caching Benchmark Harness for vLLM
Model: 8B AWQ Quantized (casperhansen/llama-3.1-8b-instruct-awq or Qwen2.5-7B-Instruct-AWQ)
Hardware: NVIDIA GeForce RTX 3060 (12GB VRAM)
Target: Ampere Memory Bandwidth & PagedAttention KV-Cache Saturation
"""

import asyncio
import time
import argparse
import statistics
import json
from pathlib import Path
import httpx
from rich.console import Console
from rich.table import Table

console = Console()

async def send_stream_request(client: httpx.AsyncClient, url: str, model: str, prompt: str, max_tokens: int):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.2
    }
    
    start_time = time.perf_counter()
    first_token_time = None
    token_timestamps = []
    
    try:
        async with client.stream("POST", f"{url}/v1/chat/completions", json=payload, timeout=120.0) as response:
            if response.status_code != 200:
                text = await response.aread()
                return {"status": "error", "error": f"HTTP {response.status_code}: {text.decode('utf-8', errors='ignore')[:100]}"}
                
            async for line in response.aiter_lines():
                now = time.perf_counter()
                if line.startswith("data: ") and line != "data: [DONE]":
                    if first_token_time is None:
                        first_token_time = now
                    token_timestamps.append(now)
                    
        end_time = time.perf_counter()
        total_time = end_time - start_time
        num_tokens = len(token_timestamps)
        ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else 0
        
        itls = []
        for i in range(1, len(token_timestamps)):
            itls.append((token_timestamps[i] - token_timestamps[i-1]) * 1000)
            
        mean_itl_ms = statistics.mean(itls) if itls else 0
        
        return {
            "status": "success",
            "total_time": total_time,
            "ttft_ms": ttft_ms,
            "itl_ms": mean_itl_ms,
            "tokens": num_tokens,
            "tokens_per_sec": num_tokens / total_time if total_time > 0 else 0
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

async def run_concurrency_tier(url: str, model: str, concurrency: int, num_requests: int, max_tokens: int, prompt: str):
    sem = asyncio.Semaphore(concurrency)
    
    async with httpx.AsyncClient() as client:
        async def bounded_request():
            async with sem:
                return await send_stream_request(client, url, model, prompt, max_tokens)
                
        start_time = time.perf_counter()
        tasks = [bounded_request() for _ in range(num_requests)]
        results = await asyncio.gather(*tasks)
        wall_time = time.perf_counter() - start_time
        
    successes = [r for r in results if r.get("status") == "success"]
    failures = [r for r in results if r.get("status") == "error"]
    
    if not successes:
        return {
            "concurrency": concurrency,
            "success": 0,
            "total": num_requests,
            "error": failures[0]["error"] if failures else "Unknown error"
        }
        
    ttfts = [r["ttft_ms"] for r in successes]
    itls = [r["itl_ms"] for r in successes if r["itl_ms"] > 0]
    total_tokens = sum(r["tokens"] for r in successes)
    system_tps = total_tokens / wall_time if wall_time > 0 else 0
    
    ttfts_sorted = sorted(ttfts)
    p50_ttft = statistics.median(ttfts)
    p90_ttft = ttfts_sorted[int(len(ttfts_sorted) * 0.90)] if len(ttfts_sorted) > 1 else ttfts_sorted[0]
    p99_ttft = ttfts_sorted[int(len(ttfts_sorted) * 0.99)] if len(ttfts_sorted) > 1 else ttfts_sorted[-1]
    
    return {
        "concurrency": concurrency,
        "success": len(successes),
        "total": num_requests,
        "wall_time_s": wall_time,
        "total_tokens": total_tokens,
        "system_tps": system_tps,
        "p50_ttft_ms": p50_ttft,
        "p90_ttft_ms": p90_ttft,
        "p99_ttft_ms": p99_ttft,
        "mean_itl_ms": statistics.mean(itls) if itls else 0
    }

async def main():
    parser = argparse.ArgumentParser(description="AWQ 8B Inference Concurrency Sweep")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", default="casperhansen/llama-3.1-8b-instruct-awq")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--tiers", type=str, default="1,5,10,15,20", help="Comma-separated concurrency tiers")
    parser.add_argument("--requests-per-tier", type=int, default=20)
    parser.add_argument("--output", default="benchmarks/awq_8b_results.json")
    args = parser.parse_args()
    
    tiers = [int(x.strip()) for x in args.tiers.split(",")]
    
    console.print("[bold yellow]══════════════════════════════════════════════════════════════[/bold yellow]")
    console.print(f"[bold cyan]vLLM 8B AWQ Concurrency Stress Test — RTX 3060 (12GB)[/bold cyan]")
    console.print(f"Endpoint: [green]{args.url}[/green] | Model: [green]{args.model}[/green]")
    console.print(f"Concurrency Tiers: [magenta]{tiers}[/magenta] | Max Tokens: [magenta]{args.max_tokens}[/magenta]")
    console.print("[bold yellow]══════════════════════════════════════════════════════════════[/bold yellow]\n")
    
    shared_prompt = (
        "You are an AI systems performance engineer. Explain the mechanics of "
        "continuous batching in high-throughput LLM serving. Detail how token iteration scheduling "
        "eliminates bubble time caused by variance in sequence generation lengths. Give 3 technical points."
    )
    
    tier_results = []
    
    table = Table(title="AWQ 8B High-Concurrency Benchmark Results (RTX 3060 12GB)")
    table.add_column("Concurrency", justify="center", style="cyan")
    table.add_column("Success Rate", justify="center", style="green")
    table.add_column("Throughput (tok/s)", justify="right", style="bold yellow")
    table.add_column("P50 TTFT (ms)", justify="right", style="magenta")
    table.add_column("P90 TTFT (ms)", justify="right", style="magenta")
    table.add_column("Mean ITL (ms)", justify="right", style="blue")
    
    for c in tiers:
        num_reqs = max(args.requests_per_tier, c)
        console.print(f"Testing concurrency: [bold cyan]{c}[/bold cyan] ({num_reqs} requests)...")
        res = await run_concurrency_tier(args.url, args.model, c, num_reqs, args.max_tokens, shared_prompt)
        tier_results.append(res)
        
        if "error" in res and res["success"] == 0:
            table.add_row(str(c), f"0/{num_reqs} (FAIL)", "0.0", "-", "-", "-")
        else:
            table.add_row(
                str(c),
                f"{res['success']}/{res['total']}",
                f"{res['system_tps']:.1f}",
                f"{res['p50_ttft_ms']:.1f}",
                f"{res['p90_ttft_ms']:.1f}",
                f"{res['mean_itl_ms']:.1f}"
            )
            
    console.print("\n")
    console.print(table)
    
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(tier_results, f, indent=2)
    console.print(f"\n[green]✓ Benchmark results saved to {out_path}[/green]")

if __name__ == "__main__":
    asyncio.run(main())

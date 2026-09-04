"""
High-Performance Model Serving & Inference Benchmark Script
Measures Time-to-First-Token (TTFT), Inter-Token Latency (ITL), and Tokens/Sec under Concurrency.
Target: Local vLLM / OpenAI-compatible endpoint on RTX 3060 (12GB VRAM).
"""

import asyncio
import time
import argparse
import statistics
import httpx
from rich.console import Console
from rich.table import Table

console = Console()

async def send_inference_request(client: httpx.AsyncClient, url: str, model: str, prompt: str, max_tokens: int):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.7
    }
    
    start_time = time.perf_counter()
    first_token_time = None
    token_timestamps = []
    
    try:
        async with client.stream("POST", f"{url}/v1/chat/completions", json=payload, timeout=60.0) as response:
            if response.status_code != 200:
                return {"status": "error", "error": f"HTTP {response.status_code}"}
                
            async for chunk in response.aiter_text():
                now = time.perf_counter()
                if first_token_time is None and "content" in chunk:
                    first_token_time = now
                token_timestamps.append(now)
                
        end_time = time.perf_counter()
        total_time = end_time - start_time
        num_tokens = len(token_timestamps)
        ttft = (first_token_time - start_time) * 1000 if first_token_time else 0
        
        return {
            "status": "success",
            "total_time": total_time,
            "ttft_ms": ttft,
            "tokens": num_tokens,
            "tokens_per_sec": num_tokens / total_time if total_time > 0 else 0
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

async def run_benchmark(url: str, model: str, concurrency: int, total_requests: int, max_tokens: int):
    console.print(f"[bold cyan]Starting AI Inference Benchmark on {url}[/bold cyan]")
    console.print(f"Model: [green]{model}[/green] | Concurrency: [yellow]{concurrency}[/yellow] | Total Requests: [yellow]{total_requests}[/yellow]")
    
    test_prompt = "Explain the difference between PagedAttention and standard attention in transformer model serving in 3 concise bullet points."
    
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient() as client:
        async def bounded_request():
            async with sem:
                return await send_inference_request(client, url, model, test_prompt, max_tokens)
                
        overall_start = time.perf_counter()
        tasks = [bounded_request() for _ in range(total_requests)]
        results = await asyncio.gather(*tasks)
        overall_time = time.perf_counter() - overall_start
        
    successes = [r for r in results if r.get("status") == "success"]
    failures = [r for r in results if r.get("status") == "error"]
    
    if not successes:
        console.print(f"[bold red]All requests failed![/bold red] Errors: {[f['error'] for f in failures[:3]]}")
        return
        
    ttfts = [r["ttft_ms"] for r in successes]
    tpss = [r["tokens_per_sec"] for r in successes]
    total_tokens_generated = sum(r["tokens"] for r in successes)
    system_throughput = total_tokens_generated / overall_time
    
    table = Table(title="Inference Benchmark Summary (RTX 3060 12GB)")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Successful Requests", f"{len(successes)} / {total_requests}")
    table.add_row("Overall Wall Time", f"{overall_time:.2f} s")
    table.add_row("System Throughput (Tokens/s)", f"{system_throughput:.2f} tok/s")
    table.add_row("Mean TTFT (Time to First Token)", f"{statistics.mean(ttfts):.2f} ms")
    table.add_row("Median TTFT", f"{statistics.median(ttfts):.2f} ms")
    table.add_row("P95 TTFT", f"{sorted(ttfts)[int(len(ttfts) * 0.95)]:.2f} ms")
    table.add_row("Mean Per-Stream Tokens/s", f"{statistics.mean(tpss):.2f} tok/s")
    
    console.print(table)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", default="casperhansen/llama-3.1-8b-instruct-awq")
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--total-requests", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()
    
    asyncio.run(run_benchmark(args.url, args.model, args.concurrency, args.total_requests, args.max_tokens))

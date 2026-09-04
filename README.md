# AI Systems & Inference Platform Lab (RTX 3060 12GB)
> High-performance inference engine benchmarking, model quantization (AWQ/FP8), and continuous batching lab for Sharvin Neve.

---

## 🛠️ Hardware Specification
* **GPU**: NVIDIA GeForce RTX 3060 (12GB GDDR6 VRAM)
* **Architecture**: Ampere (Compute Capability 8.6)
* **Target Workloads**: 
  * Meta-Llama-3.1-8B-Instruct (4-bit AWQ / GPTQ)
  * Mistral-7B-Instruct-v0.3
  * Real-Time WebRTC Multimodal Ingestion (InferStream-X)

---

## 🚀 Setup Runbook (WSL2 Ubuntu)

### 1. Environment Activation
```bash
source ~/ai-systems-venv/bin/activate
```
*(Environment variables `VLLM_WSL2_ENABLE_PIN_MEMORY=1` and `VLLM_USE_FLASHINFER_SAMPLER=0` are already permanently configured in `~/.bashrc`)*

### 2. Launch Local vLLM Inference Server
```bash
# Serves high-throughput LLM with PagedAttention continuous batching on RTX 3060:
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --port 8000 \
  --gpu-memory-utilization 0.70 \
  --max-model-len 4096

# Or for Llama-3.1-8B-Instruct AWQ 4-bit:
vllm serve casperhansen/llama-3.1-8b-instruct-awq \
  --quantization awq \
  --port 8000 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096
```

### 3. Run Concurrent Token Generation Benchmark
```bash
# In another terminal:
python /mnt/e/Projects/ai-systems-lab/benchmark_serving.py --concurrency 5 --total-requests 20
```

---

## 📊 Live Verified Benchmarks (RTX 3060 12GB VRAM)

| Metric | Measured Value | Notes |
| :--- | :--- | :--- |
| **Aggregate System Throughput** | **755.41 tok/s** | Continuous batching over 5 concurrent streams |
| **Median Time to First Token (TTFT)** | **49.18 ms** | Sub-50ms ultra-low latency response |
| **Mean Per-Stream Generation** | **158.52 tok/s** | Sustained token generation speed |
| **Allocated KV-Cache** | **7.11 GiB** | PagedAttention dynamic block pool |
| **Success Rate** | **100% (20/20)** | Zero dropped requests or timeouts |


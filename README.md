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

### Benchmark 1: 0.5B Baseline (Continuous Batching Architecture)
| Metric | Measured Value | Notes |
| :--- | :--- | :--- |
| **Aggregate System Throughput** | **755.41 tok/s** | Continuous batching over 5 concurrent streams |
| **Median Time to First Token (TTFT)** | **49.18 ms** | Sub-50ms ultra-low latency response |
| **Mean Per-Stream Generation** | **158.52 tok/s** | Sustained token generation speed |
| **Allocated KV-Cache** | **7.11 GiB** | PagedAttention dynamic block pool |
| **Success Rate** | **100% (20/20)** | Zero dropped requests or timeouts |

---

### Benchmark 2: 7B/8B Parameter AWQ 4-bit (High-Concurrency Saturation)
* **Model**: `Qwen/Qwen2.5-7B-Instruct-AWQ` (5.29 GiB weights)
* **Kernel**: AutoAWQ Marlin 4-bit GEMM + FlashAttention-2
* **Engine Settings**: `max-model-len=4096`, `gpu-memory-utilization=0.85`, `enable-prefix-caching=True`
* **GPU KV-Cache Pool**: **67,440 tokens** (3.6 GiB allocated)

| Concurrency Tier | Completed Requests | System Throughput (tok/s) | P50 TTFT (ms) | P90 TTFT (ms) | Mean ITL (ms) | Scaling Efficiency |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 stream** | 20 / 20 (100%) | **57.2 tok/s** | **38.7 ms** | 89.0 ms | 17.2 ms | 1.0x (Baseline) |
| **5 streams** | 20 / 20 (100%) | **258.0 tok/s** | **70.5 ms** | 634.6 ms | 17.9 ms | **4.51x** |
| **10 streams** | 20 / 20 (100%) | **476.1 tok/s** | **184.4 ms** | 232.7 ms | 19.5 ms | **8.32x** |
| **15 streams** | 20 / 20 (100%) | **451.2 tok/s** | **237.5 ms** | 239.8 ms | 21.1 ms | **7.89x** |
| **20 streams (Peak)** | 20 / 20 (100%) | **592.0 tok/s** | **470.9 ms** | 481.4 ms | 30.2 ms | **10.35x (Peak)** |

> **Key Architectural Takeaways:**
> 1. **10.35x Throughput Amortization**: As concurrency scaled from 1 to 20 streams, system throughput increased from **57.2 tok/s to 592.0 tok/s** (active burst client rate: 2,560 tokens in 4.32s; server 10s window logged at 486.7 tok/s) by amortizing memory bus weight loading across batch dimensions.
> 2. **Prefix Cache Hit Rate (87.1%)**: Repetitive prompt tokens hit vLLM's Radix-tree Automatic Prefix Caching (APC) (`task-2127.log:L193`), bypassing prefill computation and yielding sub-40ms P50 TTFT on cold single-stream hits.
> 3. **Zero OOMs / Zero Packet Loss**: Handled 100/100 requests flawlessly within 12GB VRAM headroom using 67,440-token PagedAttention pools (peak KV usage only 3.4%).

---

### Benchmark 3: Speculative Decoding vs. Continuous Batching (The 34.2% Acceptance Trap)
* **Target Model**: `Qwen/Qwen2.5-7B-Instruct-AWQ` (5.19 GiB weights, 4-bit Marlin GEMM)
* **Draft Model**: `Qwen/Qwen2.5-0.5B-Instruct` (0.92 GiB weights, FP16, $K=3$ tokens)
* **Hardware**: NVIDIA GeForce RTX 3060 (12GB VRAM, 360 GB/s GDDR6 bus)
* **Engine Settings**: `max-model-len=2048`, `gpu-memory-utilization=0.85`, `speculative-config: method=draft_model, use_heterogeneous_vocab=true`
* **Artifact Logs**: Audited run logs and JSON metrics recorded at `benchmarks/speculative_results.json`.

#### Side-by-Side Performance Comparison:

| Concurrency Tier ($C$) | Baseline Continuous Batching | Speculative Decoding ($K=3$) | Throughput Impact | Mean ITL Comparison |
| :---: | :---: | :---: | :---: | :---: |
| **$C=1$ (Single Stream)** | **57.2 tok/s** | **5.9 tok/s** | **9.7x SLOWER** | 17.2 ms vs. **166.5 ms** |
| **$C=5$** | **258.0 tok/s** | **17.5 tok/s** | **14.7x SLOWER** | 17.9 ms vs. **199.2 ms** |
| **$C=10$** | **476.1 tok/s** | **45.8 tok/s** | **10.4x SLOWER** | 19.5 ms vs. **198.3 ms** |
| **$C=20$ (Peak)** | **592.0 tok/s** | **77.1 tok/s** | **7.7x SLOWER** | 30.2 ms vs. **236.0 ms** |

#### Why Speculative Decoding Collapsed on Consumer Silicon (The 4 Root Causes):
1. **The Acceptance Rate Cliff (34.2% Avg Acceptance)**:
   * Empirical telemetry revealed a steep decay across draft token positions:
     * Position 1 Acceptance: **54.4%**
     * Position 2 Acceptance: **32.9%**
     * Position 3 Acceptance: **15.2%** (84.8% rejection rate!)
   * With an average draft acceptance rate of only **34.2%**, 65.8% of memory bandwidth and compute cycles were completely wasted drafting and verifying discarded token candidates.
2. **Vocabulary Heterogeneity Overhead**:
   * Target model vocabulary (`152,064`) vs. Draft model vocabulary (`151,936`).
   * Reconciling the 128 out-of-intersection tokens required CPU/GPU logit mapping and tensor reconciliation on every single step, introducing CPU-GPU synchronization stalls.
3. **Memory Bus Trashing (360 GB/s Bottleneck)**:
   * Standard continuous batching loads the 7B AWQ weights once per step and amortizes them across the batch using Marlin Tensor Core kernels.
   * Speculative decoding forced the memory bus to alternate sequentially between loading unquantized 0.5B FP16 weights ($K$ times) and then loading the 7B AWQ weights, thrashing L2 cache and saturating the 360 GB/s bus.
4. **41.2% KV-Cache VRAM Tax**:
   * Loading both the Target (5.19 GiB) and Draft (0.92 GiB) models shrank the PagedAttention KV pool from **67,440 tokens down to 39,680 tokens**.
   * On consumer hardware, reserving VRAM for a draft model directly cannibalizes the concurrency pool needed for multi-tenant batching.

> **Production Conclusion**: 
> Speculative decoding is not a free lunch. On hardware-constrained consumer GPUs with sub-400 GB/s memory bandwidth, **Continuous Dynamic Batching + AutoAWQ Marlin GEMM kernels drastically outperforms multi-model speculative decoding by up to 14.7x**. Speculative decoding should only be deployed when draft acceptance exceeds 70%+ via identical architectures, specialized draft heads (e.g. Medusa/EAGLE), or strictly memory-bound batch-1 serverless endpoints.



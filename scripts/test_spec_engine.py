import sys
from vllm import LLM, SamplingParams

try:
    print("Testing Speculative with use_heterogeneous_vocab=True...")
    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct-AWQ",
        quantization="awq",
        gpu_memory_utilization=0.85,
        max_model_len=2048,
        speculative_config={
            "method": "draft_model",
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "num_speculative_tokens": 3,
            "use_heterogeneous_vocab": True,
        },
        enforce_eager=True,
    )
    print("SUCCESS: Engine initialized with heterogeneous vocab!")
    prompts = ["Explain the concept of memory bandwidth in GPU computing:"]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=64)
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        print("Generated text:", output.outputs[0].text[:100])
except Exception as e:
    print("ERROR:", e)
    import traceback
    traceback.print_exc()

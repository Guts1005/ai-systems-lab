import vllm
import vllm.config
from vllm.engine.arg_utils import EngineArgs

print("vLLM version:", vllm.__version__)
spec_configs = [a for a in dir(vllm.config) if "spec" in a.lower()]
print("Speculative configs in vllm.config:", spec_configs)

engine_args = [f for f in dir(EngineArgs) if "spec" in f.lower()]
print("Speculative fields in EngineArgs:", engine_args)

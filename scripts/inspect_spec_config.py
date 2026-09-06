import inspect
from vllm.config.speculative import SpeculativeConfig

print(inspect.signature(SpeculativeConfig.__init__))

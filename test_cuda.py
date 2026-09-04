import torch
print("=" * 60)
print(f"PyTorch Version   : {torch.__version__}")
print(f"CUDA Available    : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name       : {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM        : {round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)} GB")
    print(f"CUDA Version      : {torch.version.cuda}")
    print(f"cuDNN Version     : {torch.backends.cudnn.version()}")
    x = torch.randn(1000, 1000, device="cuda")
    y = x @ x
    print("Tensor MatMul Test: PASSED on RTX 3060 CUDA Core!")
import vllm
from vllm.utils.platform_utils import is_uva_available, is_pin_memory_available
print(f"vLLM Version      : {vllm.__version__}")
print(f"Pin Memory Avail  : {is_pin_memory_available()}")
print(f"UVA Available     : {is_uva_available()}")
print("=" * 60)

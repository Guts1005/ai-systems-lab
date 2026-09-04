#!/usr/bin/env bash
set -e

echo "=== [1/4] Checking NVIDIA Driver & CUDA in WSL2 ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "nvidia-smi not found. Ensure NVIDIA Windows drivers are installed."
    exit 1
fi

echo "=== [2/4] Ensuring uv is installed ==="
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "=== [3/4] Creating/Activating Python 3.11 Virtual Environment ==="
VENV_PATH="$HOME/ai-systems-venv"
if [ ! -d "$VENV_PATH" ]; then
    uv venv --python 3.11 "$VENV_PATH"
fi
source "$VENV_PATH/bin/activate"

echo "=== [4/4] Installing PyTorch with CUDA 12.4 & vLLM ==="
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
uv pip install vllm httpx aiohttp pydantic rich

echo "=== Verification ==="
python -c "import torch; print('PyTorch Version:', torch.__version__, '| CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0))"

echo "=== Setup Complete! ==="
echo "Activate anytime with: source $HOME/ai-systems-venv/bin/activate"
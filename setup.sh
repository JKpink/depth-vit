#!/bin/bash
# 一键环境搭建

echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "Installing PyTorch (CUDA 12.1)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q

echo "Installing dependencies..."
pip install -r requirements.txt -q

echo "Verify:"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

echo ""
echo "Done. Activate with: source venv/bin/activate"
echo "Train:      python src/train.py --plan A --data data/nyu"
echo "Eval:       python src/eval.py --plan A --data data/nyu --checkpoint outputs/plan_A/best.pt"
echo "Test:       python -m pytest tests/ -v"

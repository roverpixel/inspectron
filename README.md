# inspectron
visual language model simulation

# AttnScope 🔬

An interactive visualizer and minimal PyTorch implementation of a decoder-only Large Language Model (Transformer).

## Features
- **Minimal PyTorch LLM (`model.py`)**: ~120 lines of self-contained, readable code implementing multi-head causal self-attention, pre-layer norm residuals, training loop, and autoregressive generation.
- **Interactive Visualizer (`web/index.html`)**: Single-file web app illustrating token embedding, causal masking, layer-by-layer tensor shapes, parameter counts, and the architectural differences between GPT-2 and LLaMA (RoPE, RMSNorm, SwiGLU).

## Getting Started

### 1. Run the PyTorch Model
```bash
pip install -r requirements.txt
python model.py

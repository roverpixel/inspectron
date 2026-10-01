# Inspectron 🔬

**Inspectron** is an interactive visualizer and educational simulation of a decoder-only Large Language Model (Transformer). It provides an intuitive, hands-on way to explore how transformers process tokens, compute multi-head causal self-attention, train on text corpora, and predict subsequent tokens in real time.

> 🐳 **Note:** Inspectron is packaged as a **Docker Compose** application! Everything needed—including PyTorch, Flask, and the interactive web frontend—runs seamlessly with a single `docker compose up` command.

---

## Features

- 🧠 **Interactive Architecture Explorer:** Inspect every layer of the transformer pipeline—Token & Positional Embeddings, LayerNorm, Multi-Head Causal Self-Attention ($Q, K, V$), Feed-Forward Networks (MLP), and Logit Projections.
- ⚖️ **Architecture Comparisons:** Toggle between classic **GPT-2** style (Learned Positional Embeddings, LayerNorm, standard GELU MLP) and modern **LLaMA** style (RoPE rotary embeddings, RMSNorm, SwiGLU activation).
- 📈 **Real-Time Training Dashboard:** Trigger asynchronous background training loops directly from the UI. Monitor live step counts, learning rates, and a dynamically updating loss curve.
- 🎯 **Next-Token Prediction & Attention Heatmaps:** Type custom prompts or choose preset examples to view the predicted next token, top candidate probability distributions, logits, and token-to-token causal attention heatmaps.
- 🔍 **Grammar & Anomaly Inspector:** Analyze text using token-level surprisal ($-\log P(x)$) and perplexity scoring. Highlights anomalous tokens, rare transitions, or grammar quirks with interactive alternative candidate recommendations.
- 📚 **Multi-Corpus Dataset Switching:** Dynamically swap between training corpora and pre-trained checkpoints without restarting:
  - **Simple Wikipedia:** Clean, modern factual English.
  - **Classic Literature:** Ornate, dramatic prose and poetry (Poe, Shakespeare).
  - **Combined Dataset:** A blend of classical and modern text.
- 🚀 **Zero Host Setup Needed:** Fully containerized with PyTorch (supports CUDA if available, automatically falls back to CPU).

---

## Getting Started with Docker Compose

The easiest and recommended way to run Inspectron is with **Docker Compose**.

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) (v20.10+)
- [Docker Compose](https://docs.docker.com/compose/) (v2.0+)

### Quickstart

1. **Clone the repository:**
   ```bash
   git clone git@github.com:roverpixel/inspectron.git
   cd inspectron
   ```

2. **Launch the application with Docker Compose:**
   ```bash
   docker compose up
   ```
   *(Or run in background with `docker compose up -d`)*

3. **Open the Web Interface:**
   Navigate to [http://localhost:7001](http://localhost:7001) in your browser.

4. **Stop the container:**
   ```bash
   docker compose down
   ```

---

## Makefile Helpers

A `makefile` is provided for common Docker operations:

| Command | Description |
|---|---|
| `make start` | Start the app in the background (`docker compose up -d`) |
| `make stop` | Stop the app and remove volumes (`docker compose down --volumes`) |
| `make build` | Rebuild the Docker container image |
| `make log` | Tail live container logs (`docker compose logs -f web`) |
| `make enter` | Open an interactive bash shell inside the running container |
| `make help` | Show available make targets and URLs |

---

## GPU Acceleration (Optional)

By default, Inspectron runs comfortably on the CPU. If you have an NVIDIA GPU with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed, you can enable hardware acceleration:

In `docker-compose.yml`, uncomment the `deploy` block:

```yaml
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

Then rebuild and restart:
```bash
docker compose up --build
```

---

## Alternative: Local Python Setup (Without Docker)

If you prefer to run directly on your host machine without Docker:

1. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the Flask server:**
   ```bash
   python app.py
   ```
   Access the dashboard at [http://localhost:7001](http://localhost:7001).

4. **CLI Model Test (Optional):**
   You can also test the standalone model and training loop directly from the command line:
   ```bash
   python web/model.py
   ```

---

## Project Structure

```
inspectron/
├── app.py                  # Flask server and REST API endpoints
├── docker-compose.yml      # Docker Compose configuration (Port 7001)
├── Dockerfile              # PyTorch + CUDA / CPU container definition
├── makefile                # Make shortcuts (start, stop, logs, shell)
├── requirements.txt        # Python package dependencies
├── AGENTS.md               # Architecture guidance and developer notes
├── data/                   # Corpora and pre-trained weights
│   ├── corpus_simple_wikipedia.txt
│   ├── corpus_literature.txt
│   ├── corpus_combined.txt
│   ├── model_weights_simple_wikipedia.pt
│   ├── model_weights_literature.pt
│   └── model_weights.pt
└── web/
    ├── index.html          # Interactive visualizer, dashboard & inspector UI
    └── model.py            # Minimal Transformer implementation & ModelManager
```

---

## REST API Overview

Inspectron exposes a REST API for programmatic interaction:

| Endpoint | Method | Description |
|---|---|---|
| `/` | `GET` | Serves the web visualizer application |
| `/api/status` | `GET` | Returns model hyperparameters, current step, loss, and corpus status |
| `/api/predict` | `POST` | Computes next-token probability distribution and attention maps for a prompt |
| `/api/generate` | `POST` | Autoregressively generates text continuation |
| `/api/train` | `POST` | Starts asynchronous training loop in background thread (`steps`, `learning_rate`) |
| `/api/train/status` | `GET` | Polls active training status and historical loss metrics |
| `/api/grammar/inspect`| `POST` | Analyzes text for per-token surprisal, anomalies, and expected alternatives |
| `/api/corpus/list` | `GET` | Lists available datasets and checkpoints |
| `/api/corpus/select` | `POST` | Switches active corpus and loads corresponding checkpoint |

---

## License

This project is licensed under the [MIT License](LICENSE).

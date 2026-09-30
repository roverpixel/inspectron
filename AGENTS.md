# Agent Instructions

## Project Overview
This project, `inspectron`, is a demonstration of a minimal, decoder-only Large Language Model (Transformer). The goal is to provide a visual representation of the model. The web page (`web/index.html`) is intended to show graphics that display the static architecture and the dynamic state of the model both during training and execution.

## Infrastructure
- **Server:** We use a Flask web server (`app.py`) to serve the frontend and provide a foundation for future backend communication.
- **Docker:** The application runs in a Docker container using `docker-compose`. Port 7001 is mapped. The container uses a PyTorch base image with CUDA support (falling back to CPU if unavailable).

## Future Work & Guidance for Agents
As you contribute to this repository, keep the following goals and instructions in mind:

1. **Communication Layer:**
   - The primary next step is to establish a communication layer between the PyTorch model (`web/model.py`) and the web frontend.
   - You should consider implementing WebSockets (e.g., using `Flask-SocketIO` or similar) or REST API endpoints to stream the model's state (loss, token probabilities, attention matrices) to the frontend in real time during training or generation.

2. **Backend/Model Integration:**
   - Currently, `model.py` is a standalone script. It will need to be integrated with `app.py` so that training or inference can be triggered and monitored via the web interface.
   - Take care to manage the training loop asynchronously (e.g., in a background thread or a separate worker) so it doesn't block the Flask server.

3. **Frontend Visualizations:**
   - The frontend (`web/index.html`) contains placeholders and static visualizers for the transformer architecture.
   - Future agents should update the frontend JavaScript to dynamically ingest data from the backend and animate the training process (e.g., updating loss charts, live attention heatmaps, and next-token probability bars).

4. **Environment:**
   - All components should run seamlessly via `docker compose up`. If you introduce new dependencies, ensure they are added to `requirements.txt` and that the `Dockerfile` rebuilds correctly.
   - Try to maintain support for both CUDA (when available) and CPU, as the application should be portable.

5. **Code Quality:**
   - Ensure that the minimal, educational nature of the code is preserved. `model.py` is designed to be readable (~120 lines). Avoid over-complicating the PyTorch implementation unless necessary for the visualization.

When making changes, refer back to these guidelines to ensure they align with the project's educational and visual goals.

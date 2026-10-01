from flask import Flask, send_from_directory, request, jsonify
import os
import sys

# Ensure web directory is in path for module imports
sys.path.insert(0, os.path.dirname(__file__))

from web.model import get_model_manager

app = Flask(__name__, static_folder='web', static_url_path='')

# Initialize or retrieve model manager instance
mgr = get_model_manager()

@app.route('/')
def index():
    return send_from_directory('web', 'index.html')

@app.route('/api/status', methods=['GET'])
def get_status():
    """Returns general model info, hyperparameters, and training state."""
    return jsonify(mgr.get_info())

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predicts next token probabilities and returns attention map for prompt."""
    data = request.get_json(force=True, silent=True) or {}
    prompt = data.get('prompt', '')
    temperature = data.get('temperature', 1.0)
    top_k = data.get('top_k', 6)

    try:
        result = mgr.predict_next(prompt=prompt, temperature=temperature, top_k=top_k)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/generate', methods=['POST'])
def generate():
    """Autoregressively generates multiple tokens from prompt."""
    data = request.get_json(force=True, silent=True) or {}
    prompt = data.get('prompt', '')
    max_tokens = min(int(data.get('max_tokens', 10)), 100)
    temperature = data.get('temperature', 0.7)

    try:
        result = mgr.generate_text(prompt=prompt, max_new_tokens=max_tokens, temperature=temperature)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/train', methods=['POST'])
def trigger_training():
    """Triggers background training steps."""
    data = request.get_json(force=True, silent=True) or {}
    steps = int(data.get('steps', 100))
    learning_rate = float(data.get('learning_rate', 1e-3))

    if mgr.is_training:
        return jsonify({"status": "already_training", "message": "Model is already training."}), 409

    success = mgr.start_background_train(steps=steps, lr=learning_rate)
    return jsonify({"status": "training_started" if success else "failed", "steps": steps})

@app.route('/api/train/status', methods=['GET'])
def train_status():
    """Polls training status and recent loss history."""
    return jsonify({
        "is_training": mgr.is_training,
        "step": mgr.current_step,
        "loss": mgr.current_loss,
        "training_history": mgr.training_history
    })

@app.route('/api/grammar/inspect', methods=['POST'])
def inspect_grammar():
    """Analyzes text for per-token surprisal, anomalies, and expected alternatives."""
    data = request.get_json(force=True, silent=True) or {}
    text = data.get('text', '')
    try:
        result = mgr.inspect_grammar(text=text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/corpus/list', methods=['GET'])
def get_corpora():
    """Returns available corpora with metadata and active status."""
    return jsonify({
        "active_corpus": getattr(mgr, "active_corpus_id", "simple_wikipedia"),
        "corpora": mgr.get_corpora_list()
    })

@app.route('/api/corpus/select', methods=['POST'])
def select_corpus():
    """Switches the active training corpus and loads corresponding checkpoint."""
    data = request.get_json(force=True, silent=True) or {}
    corpus_id = data.get('corpus_id', 'simple_wikipedia')
    if mgr.is_training:
        return jsonify({"error": "Cannot switch corpus while training is running."}), 409

    success, msg = mgr.switch_corpus(corpus_id)
    if not success:
        return jsonify({"error": msg}), 400

    return jsonify({
        "status": "success",
        "message": msg,
        "active_corpus": mgr.active_corpus_id,
        "info": mgr.get_info()
    })

if __name__ == '__main__':
    # Run server on 0.0.0.0:7001
    app.run(host='0.0.0.0', port=7001, debug=True)

from flask import Flask, request, jsonify
import threading

from models.brainformer import BrainFormer
from data.loader import load_data
from utils.persistence import load_model
from sampling.text_generation import generate_text
from config.base_config import device

app = Flask(__name__)

# Global model ve vocablar
model = None
vocab = None
word_to_id = None
id_to_word = None

def load_model_and_data():
    global model, vocab, word_to_id, id_to_word
    model, vocab, word_to_id, id_to_word = load_model()
    if model:
        model.to(device)
        model.eval()
    else:
        print("❌ Model bulunamadı! Önce eğitim yapmalısınız.")

@app.route('/generate', methods=['POST'])
def generate():
    global model, vocab, word_to_id, id_to_word
    if model is None:
        return jsonify({"error": "Model yüklenmedi"}), 500

    data = request.json
    if not data or 'prompt' not in data:
        return jsonify({"error": "JSON içinde 'prompt' yok"}), 400

    prompt = data['prompt']
    temps = [0.3, 0.8, 1.2]
    results = []

    for t in temps:
        text = generate_text(model, vocab, word_to_id, id_to_word, prompt, temperature=t)
        results.append({"temperature": t, "output": text})

    return jsonify({"results": results})


def start_api():
    print("🚀 API başlatılıyor... Model yükleniyor...")
    load_model_and_data()
    # Flask uygulamasını ayrı thread'te başlatabiliriz (eğer main thread engellenmesin isteniyorsa)
    app.run(host="0.0.0.0", port=5000)

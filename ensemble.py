import os
import requests
import joblib
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Layer
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from huggingface_hub import hf_hub_download
from database import system_collection


load_dotenv()

# -----------------------------
# HuggingFace Config
# -----------------------------
HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO")
BERT_API_URL = os.getenv("BERT_API_URL")

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}


# -----------------------------
# Model Paths
# -----------------------------
#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#MODEL_DIR = os.path.join(BASE_DIR, "models")

LR_PATH = hf_hub_download(
    repo_id=HF_MODEL_REPO,
    filename="logistic_model.pkl",
    cache_dir="/tmp"
)
RF_PATH = hf_hub_download(
    repo_id=HF_MODEL_REPO,
    filename="random_forest_model.pkl",
    cache_dir="/tmp"
)

VECT_PATH = hf_hub_download(
    repo_id=HF_MODEL_REPO,
    filename="tfidf_vectorizer.pkl",
    cache_dir="/tmp"
)

LSTM_MODEL_PATH = hf_hub_download(
    repo_id=HF_MODEL_REPO,
    filename="lstm_phishing_model.keras",
    cache_dir="/tmp"
)

TOKENIZER_PATH = hf_hub_download(
    repo_id=HF_MODEL_REPO,
    filename="tokenizer.pkl",
    cache_dir="/tmp"
)

BILSTM_MODEL_PATH = hf_hub_download(
    repo_id=HF_MODEL_REPO,
    filename="bilstm_attn_model.h5",
    cache_dir="/tmp"
)

LSTM_MAXLEN = 100


# -----------------------------
# Default Weights
# -----------------------------
DEFAULT_WEIGHTS = {
    "lr": 0.25,
    "rf": 0.15,
    "lstm": 0.10,
    "bilstm": 0.15,
    "bert": 0.35
}


stored = system_collection.find_one({"_id": "ensemble_weights"})

if stored and "weights" in stored:
    WEIGHTS = stored["weights"]
else:
    WEIGHTS = DEFAULT_WEIGHTS.copy()
    system_collection.update_one(
        {"_id": "ensemble_weights"},
        {"$set": {"weights": WEIGHTS}},
        upsert=True
    )


CLASS_LABELS = {0: "legitimate_email", 1: "phishing_email"}


# -----------------------------
# Load Models
# -----------------------------
print("Loading models...")

lr = joblib.load(LR_PATH)
rf = joblib.load(RF_PATH)
vectorizer = joblib.load(VECT_PATH)

lstm_model = load_model(LSTM_MODEL_PATH)
tokenizer = joblib.load(TOKENIZER_PATH)


class AttentionLayer(Layer):

    def build(self, input_shape):

        self.W = self.add_weight(
            shape=(input_shape[-1],),
            initializer="glorot_uniform",
            trainable=True
        )

    def call(self, inputs):

        score = tf.tensordot(inputs, self.W, axes=1)
        weights = tf.nn.softmax(score, axis=1)

        context = tf.reduce_sum(
            inputs * tf.expand_dims(weights, -1),
            axis=1
        )

        return context


bilstm_model = load_model(
    BILSTM_MODEL_PATH,
    custom_objects={"AttentionLayer": AttentionLayer}
)

print("Models loaded successfully")


# -----------------------------
# Cache
# -----------------------------
prediction_cache = {}


@lru_cache(maxsize=5000)
def cached_bert(text):

    payload = {
        "inputs": text[:2000],
        "options": {"wait_for_model": True}
    }

    r = requests.post(BERT_API_URL, headers=headers, json=payload)

    if r.status_code != 200:
        return (0.5, 0.5)

    result = r.json()

    if isinstance(result, list) and isinstance(result[0], list):
        result = result[0]

    probs = {x["label"]: x["score"] for x in result}

    p_legit = probs.get("legitimate_email", 0)
    p_phish = probs.get("phishing_email", 0)

    total = p_legit + p_phish

    if total == 0:
        return (0.5, 0.5)

    return (p_legit / total, p_phish / total)


# -----------------------------
# Helper
# -----------------------------
def prepare_sequence(text):

    seq = tokenizer.texts_to_sequences([text])

    return pad_sequences(
        seq,
        maxlen=LSTM_MAXLEN,
        padding="post",
        truncating="post"
    )


# -----------------------------
# Parallel Ensemble
# -----------------------------
def ensemble_predict(text, weights=None):

    if weights is None:
        weights = WEIGHTS

    # -------- CACHE CHECK --------
    if text in prediction_cache:
        return prediction_cache[text]

    pad = prepare_sequence(text)

    with ThreadPoolExecutor(max_workers=4) as executor:

        f_lr = executor.submit(lambda: lr.predict_proba(vectorizer.transform([text]))[0])
        f_rf = executor.submit(lambda: rf.predict_proba(vectorizer.transform([text]))[0])
        f_lstm = executor.submit(lambda: lstm_model.predict(pad, verbose=0)[0])
        f_bilstm = executor.submit(lambda: bilstm_model.predict(pad, verbose=0)[0])
        f_bert = executor.submit(lambda: cached_bert(text))

        p_lr = f_lr.result()
        p_rf = f_rf.result()

        p_lstm = np.array([1 - f_lstm.result(), f_lstm.result()])
        p_bilstm = np.array([1 - f_bilstm.result(), f_bilstm.result()])

        p_bert = np.array(f_bert.result())

    probs_dict = {
        "lr": p_lr,
        "rf": p_rf,
        "lstm": p_lstm,
        "bilstm": p_bilstm,
        "bert": p_bert
    }

    avg = np.zeros(2)

    total_w = sum(weights.values())

    for m, w in weights.items():
        avg += probs_dict[m] * w

    avg /= total_w

    pred_index = int(np.argmax(avg))

    result = {
        "final_label": CLASS_LABELS[pred_index],
        "final_confidence": float(avg[pred_index]),
        "ensemble_probabilities": {
            "legitimate_email": float(avg[0]),
            "phishing_email": float(avg[1])
        },
        "per_model_probabilities": {
            m: {
                "legitimate_email": float(v[0]),
                "phishing_email": float(v[1])
            }
            for m, v in probs_dict.items()
        }
    }

    prediction_cache[text] = result

    return result


# -----------------------------
# Risk Levels
# -----------------------------
def risk_level_from_ensemble(phish_prob):

    if phish_prob >= 0.70:
        return {
            "risk": "RED",
            "color": "#e74c3c",
            "action": "block",
            "short_message": "High phishing probability"
        }

    elif phish_prob >= 0.50:
        return {
            "risk": "YELLOW",
            "color": "#f1c40f",
            "action": "review",
            "short_message": "This might be a phishing email"
        }

    else:
        return {
            "risk": "GREEN",
            "color": "#2ecc71",
            "action": "allow",
            "short_message": "Likely safe"
        }
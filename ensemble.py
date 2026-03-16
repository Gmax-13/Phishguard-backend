import os
import json
import requests
import numpy as np
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from cachetools import TTLCache

from database import system_collection

load_dotenv()

# -----------------------------
# HuggingFace API Configuration
# -----------------------------

HF_TOKEN = os.getenv("HF_TOKEN")

LR_API = os.getenv("LR_API")
RF_API = os.getenv("RF_API")
LSTM_API = os.getenv("LSTM_API")
BILSTM_API = os.getenv("BILSTM_API")
BERT_API = os.getenv("BERT_API")

session = requests.Session()

session.headers.update({
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
})

# -----------------------------
# Ensemble Weights
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

CLASS_LABELS = {
    0: "legitimate_email",
    1: "phishing_email"
}

# -----------------------------
# Prediction Cache
# -----------------------------

prediction_cache = TTLCache(
    maxsize=2000,
    ttl=3600
)

# -----------------------------
# HuggingFace API Call
# -----------------------------

def query_model(api_url, text):

    payload = {
        "inputs": text[:2000],
        "options": {"wait_for_model": True}
    }

    try:

        r = session.post(
            api_url,
            json=payload,
            timeout=8
        )

        if r.status_code != 200:
            return np.array([0.5, 0.5])

        result = json.loads(r.text)

        if isinstance(result, list) and isinstance(result[0], list):
            result = result[0]

        probs = {x["label"]: x["score"] for x in result}

        p_legit = (
            probs.get("legitimate_email")
            or probs.get("safe")
            or probs.get("LABEL_0")
            or 0
        )
        p_phish = (
            probs.get("phishing_email")
            or probs.get("phishing")
            or probs.get("LABEL_1")
            or 0
        )

        total = p_legit + p_phish

        if total == 0:
            return np.array([0.5, 0.5])

        return np.array([
            p_legit / total,
            p_phish / total
        ])

    except Exception:

        return np.array([0.5, 0.5])


# -----------------------------
# Ensemble Prediction
# -----------------------------

def ensemble_predict(text, weights=None):

    if weights is None:
        weights = WEIGHTS

    # Cache check
    if text in prediction_cache:
        return prediction_cache[text]

    # Parallel API calls
    with ThreadPoolExecutor(max_workers=5) as executor:

        f_lr = executor.submit(query_model, LR_API, text)
        f_rf = executor.submit(query_model, RF_API, text)
        f_lstm = executor.submit(query_model, LSTM_API, text)
        f_bilstm = executor.submit(query_model, BILSTM_API, text)
        f_bert = executor.submit(query_model, BERT_API, text)

        p_lr = f_lr.result()
        p_rf = f_rf.result()
        p_lstm = f_lstm.result()
        p_bilstm = f_bilstm.result()
        p_bert = f_bert.result()

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
# Risk Level Classification
# -----------------------------

def risk_level_from_ensemble(prob):

    if prob >= 0.70:
        return {
            "risk": "RED",
            "color": "#ff4d4f",
            "short_message": "High probability of phishing detected.",
            "action": "Avoid interacting with this email."
        }

    if prob >= 0.50:
        return {
            "risk": "YELLOW",
            "color": "#faad14",
            "short_message": "This email looks suspicious.",
            "action": "Verify sender before taking action."
        }

    return {
        "risk": "GREEN",
        "color": "#52c41a",
        "short_message": "Email appears legitimate.",
        "action": "No immediate threat detected."
    }

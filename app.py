import os
import requests

from flask import Flask, request, jsonify
from flask_cors import CORS
from bson import ObjectId
from datetime import datetime, timezone

from email_preprocessing import process_email_json
from ensemble import (
    ensemble_predict,
    risk_level_from_ensemble,
    WEIGHTS,
    prediction_cache
)

from database import store_email, emails_collection


# -----------------------------
# Flask App Initialization
# -----------------------------

app = Flask(__name__)

CORS(app, origins=[
    "https://mail.google.com",
    "chrome-extension://*"
])


# -----------------------------
# HuggingFace Model Warmup
# -----------------------------

def warm_models():

    urls = [
        os.getenv("LR_API"),
        os.getenv("RF_API"),
        os.getenv("LSTM_API"),
        os.getenv("BILSTM_API"),
        os.getenv("BERT_API")
    ]

    headers = {
        "Authorization": f"Bearer {os.getenv('HF_TOKEN')}",
        "Content-Type": "application/json"
    }

    payload = {
        "inputs": "test email",
        "options": {"wait_for_model": True}
    }

    for url in urls:

        if not url:
            continue

        try:
            requests.post(url, headers=headers, json=payload, timeout=5)
        except:
            pass


# Warm models at startup
warm_models()


# -----------------------------
# Health Check Endpoint
# -----------------------------

@app.route("/")
def health():

    return jsonify({
        "status": "PhishGuard API running"
    })


# -----------------------------
# Email Analysis Endpoint
# -----------------------------

@app.route("/analyze", methods=["POST"])
def analyze_email():

    try:

        email_json = request.get_json(force=True)

        processed = process_email_json(email_json)

        clean_text = processed["clean_text"]

        result = ensemble_predict(clean_text)

        phish_prob = result["ensemble_probabilities"]["phishing_email"]

        risk = risk_level_from_ensemble(phish_prob)

        inserted_id = store_email({
            "sender": processed["sender"],
            "subject": processed["subject"],
            "clean_text": clean_text[:500],
            "final_prediction": result["final_label"],
            "final_confidence": result["final_confidence"],
            "phishing_probability": phish_prob,
            "risk_level": risk["risk"],
            "model_outputs": result["per_model_probabilities"],
            "feedback": None,
            "weights_used": WEIGHTS.copy(),
            "timestamp": datetime.now(timezone.utc)
        })

        return jsonify({
            "email_id": str(inserted_id),
            "risk_level": risk["risk"],
            "phishing_probability": phish_prob,
            "confidence": result["final_confidence"],
            "explanation": risk["short_message"],
            "details": {
                "color": risk["color"],
                "action": risk["action"]
            }
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# -----------------------------
# Feedback Endpoint (RL)
# -----------------------------

@app.route("/feedback", methods=["POST"])
def receive_feedback():

    try:

        data = request.json

        email_id = data.get("email_id")

        true_label = data.get("true_label")

        if not email_id or not true_label:
            return jsonify({"error": "Invalid input"}), 400

        emails_collection.update_one(
            {"_id": ObjectId(email_id)},
            {"$set": {"feedback": true_label}}
        )

        return jsonify({
            "message": "Feedback recorded"
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# -----------------------------
# Cache Monitoring Endpoint
# -----------------------------

@app.route("/cache_stats")
def cache_stats():

    return jsonify({
        "cache_size": len(prediction_cache)
    })


# -----------------------------
# Run Flask App
# -----------------------------

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
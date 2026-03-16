import os
import uuid
import requests
from flask import Flask, request, jsonify
from cachetools import TTLCache

from email_preprocessing import process_email_json


##################################################
# Flask App
##################################################

app = Flask(__name__)


##################################################
# Configuration
##################################################

SPACE_URL = os.environ.get(
    "SPACE_URL",
    "https://savizzz-35-phishguard-inference.hf.space/run/predict"
)

CACHE_SIZE = int(os.environ.get("CACHE_SIZE", 2000))
CACHE_TTL = int(os.environ.get("CACHE_TTL", 3600))
SPACE_TIMEOUT = int(os.environ.get("SPACE_TIMEOUT", 25))


##################################################
# Prediction Cache
##################################################

prediction_cache = TTLCache(maxsize=CACHE_SIZE, ttl=CACHE_TTL)


##################################################
# HuggingFace Space Inference
##################################################

def call_space_model(email_text):

    response = requests.post(
        SPACE_URL,
        json={"data": [email_text]},
        timeout=SPACE_TIMEOUT
    )

    response.raise_for_status()

    result = response.json()

    if "data" in result and len(result["data"]) > 0:
        return result["data"][0]

    raise Exception("Invalid response from inference Space")


##################################################
# Heuristic Risk Adjustment
##################################################

def apply_heuristics(prediction, url_features, image_features):

    score = prediction.get("phishing_probability", 0)

    if url_features.get("has_ip_url"):
        score += 0.05

    if url_features.get("has_punycode"):
        score += 0.07

    if url_features.get("long_url"):
        score += 0.03

    if url_features.get("suspicious_tld"):
        score += 0.05

    if image_features.get("image_heavy"):
        score += 0.04

    score = min(score, 1.0)

    if score < 0.5:
        risk_level = "GREEN"
        label = "SAFE"
    elif score < 0.7:
        risk_level = "YELLOW"
        label = "SUSPICIOUS"
    else:
        risk_level = "RED"
        label = "PHISHING"

    prediction["phishing_probability"] = score
    prediction["risk_level"] = risk_level
    prediction["label"] = label

    return prediction


##################################################
# Health Check
##################################################

@app.route("/", methods=["GET"])
def health():

    return jsonify({
        "status": "PhishGuard backend running"
    })


##################################################
# Cache Statistics
##################################################

@app.route("/cache_stats", methods=["GET"])
def cache_stats():

    return jsonify({
        "cache_size": len(prediction_cache),
        "cache_limit": CACHE_SIZE,
        "ttl_seconds": CACHE_TTL
    })


##################################################
# Main Email Analysis Endpoint
##################################################

@app.route("/analyze", methods=["POST"])
def analyze():

    try:

        email_json = request.get_json(silent=True)

        if not email_json:

            return jsonify({
                "error": "No JSON received"
            }), 400


        ##################################################
        # Email Preprocessing
        ##################################################

        processed = process_email_json(email_json)

        email_text = processed.get("clean_text", "")
        url_features = processed.get("url_features", {})
        image_features = processed.get("image_features", {})

        if not email_text:

            return jsonify({
                "error": "Email text extraction failed"
            }), 400


        ##################################################
        # Cache Lookup
        ##################################################

        cache_key = email_text[:500]

        if cache_key in prediction_cache:

            prediction = prediction_cache[cache_key]

        else:

            prediction = call_space_model(email_text)

            prediction_cache[cache_key] = prediction


        ##################################################
        # Apply Heuristics
        ##################################################

        prediction = apply_heuristics(
            prediction,
            url_features,
            image_features
        )


        ##################################################
        # Prepare Extension-Compatible Response
        ##################################################

        email_id = str(uuid.uuid4())

        response = {
            "risk_level": prediction.get("risk_level", "GRAY"),
            "phishing_probability": float(prediction.get("phishing_probability", 0)),
            "label": prediction.get("label", "UNKNOWN"),
            "explanation": "AI analysis of email content and links.",
            "email_id": email_id
        }

        return jsonify(response)


    except Exception as e:

        print("Analyze error:", e)

        return jsonify({
            "error": str(e)
        }), 500


##################################################
# Feedback Endpoint
##################################################

@app.route("/feedback", methods=["POST"])
def feedback():

    try:

        data = request.get_json(silent=True)

        if not data:

            return jsonify({
                "error": "No JSON received"
            }), 400

        email_id = data.get("email_id")
        true_label = data.get("true_label")

        return jsonify({
            "status": "feedback received",
            "email_id": email_id,
            "true_label": true_label
        })


    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


##################################################
# Run App (Render Compatible)
##################################################

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(host="0.0.0.0", port=port)
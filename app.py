import os
import requests
from flask import Flask, request, jsonify
from cachetools import TTLCache

from email_preprocessing import process_email_json


###################################################
# Flask App
###################################################

app = Flask(__name__)


###################################################
# Configuration
###################################################

SPACE_URL = os.environ.get(
    "SPACE_URL",
    "https://savizzz-35-phishguard-inference.hf.space/run/predict"
)

CACHE_SIZE = int(os.environ.get("CACHE_SIZE", 2000))
CACHE_TTL = int(os.environ.get("CACHE_TTL", 3600))

SPACE_TIMEOUT = int(os.environ.get("SPACE_TIMEOUT", 25))


###################################################
# Prediction Cache
###################################################

prediction_cache = TTLCache(maxsize=CACHE_SIZE, ttl=CACHE_TTL)


###################################################
# Call HuggingFace Space
###################################################

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

    return result


###################################################
# Heuristic Risk Adjustment
###################################################

def apply_heuristics(prediction, url_features, image_features):

    score = prediction.get("phishing_probability", 0)

    # URL heuristics
    if url_features.get("has_ip_url"):
        score += 0.05

    if url_features.get("has_punycode"):
        score += 0.07

    if url_features.get("long_url"):
        score += 0.03

    if url_features.get("suspicious_tld"):
        score += 0.05

    # Image phishing heuristic
    if image_features.get("image_heavy"):
        score += 0.04

    score = min(score, 1.0)

    if score < 0.5:
        risk = "GREEN"
    elif score < 0.7:
        risk = "YELLOW"
    else:
        risk = "RED"

    prediction["phishing_probability"] = score
    prediction["risk_level"] = risk

    return prediction


###################################################
# Health Endpoint
###################################################

@app.route("/", methods=["GET"])
def health():

    return jsonify({
        "status": "PhishGuard backend running"
    })


###################################################
# Cache Stats Endpoint
###################################################

@app.route("/cache_stats", methods=["GET"])
def cache_stats():

    return jsonify({
        "cache_size": len(prediction_cache),
        "cache_limit": CACHE_SIZE,
        "ttl_seconds": CACHE_TTL
    })


###################################################
# Main Detection Endpoint
###################################################

@app.route("/analyze", methods=["POST"])
def analyze():

    try:

        email_json = request.get_json(silent=True)

        if not email_json:

            return jsonify({
                "error": "No JSON received"
            }), 400


        # -----------------------------
        # Email preprocessing
        # -----------------------------

        processed = process_email_json(email_json)

        email_text = processed.get("clean_text", "")
        url_features = processed.get("url_features", {})
        image_features = processed.get("image_features", {})

        if not email_text:

            return jsonify({
                "error": "Email text extraction failed"
            }), 400


        # -----------------------------
        # Cache lookup
        # -----------------------------

        cache_key = email_text[:500]

        if cache_key in prediction_cache:

            prediction = prediction_cache[cache_key]

        else:

            prediction = call_space_model(email_text)

            prediction_cache[cache_key] = prediction


        # -----------------------------
        # Apply heuristics
        # -----------------------------

        prediction = apply_heuristics(
            prediction,
            url_features,
            image_features
        )

        return jsonify(prediction)


    except Exception as e:

        print("Analyze error:", e)

        return jsonify({
            "error": str(e)
        }), 500


###################################################
# Feedback Endpoint
###################################################

@app.route("/feedback", methods=["POST"])
def feedback():

    try:

        data = request.get_json(silent=True)

        if not data:

            return jsonify({
                "error": "No JSON received"
            }), 400

        return jsonify({
            "status": "feedback received",
            "true_label": data.get("true_label")
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


###################################################
# Run Server
###################################################

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(host="0.0.0.0", port=port)
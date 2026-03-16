import os
import requests
from flask import Flask, request, jsonify
from cachetools import TTLCache

from email_preprocessing import process_email_json


##################################################
# Flask app
##################################################

app = Flask(__name__)


##################################################
# Config
##################################################

SPACE_URL = "https://savizzz-35-phishguard-inference.hf.space/run/predict"

CACHE_SIZE = 2000
CACHE_TTL = 3600

prediction_cache = TTLCache(maxsize=CACHE_SIZE, ttl=CACHE_TTL)


##################################################
# Space inference call
##################################################

def call_space_model(email_text):

    response = requests.post(
        SPACE_URL,
        json={"data": [email_text]},
        timeout=25
    )

    response.raise_for_status()

    result = response.json()

    if "data" in result:
        return result["data"][0]

    return result


##################################################
# Heuristic adjustments
##################################################

def apply_heuristics(prediction, url_features, image_features):

    score = prediction["phishing_probability"]

    if url_features["has_ip_url"]:
        score += 0.05

    if url_features["has_punycode"]:
        score += 0.07

    if url_features["long_url"]:
        score += 0.03

    if url_features["suspicious_tld"]:
        score += 0.05

    if image_features["image_heavy"]:
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


##################################################
# Health endpoint
##################################################

@app.route("/", methods=["GET"])
def health():

    return jsonify({
        "status": "PhishGuard backend running"
    })


##################################################
# Cache stats
##################################################

@app.route("/cache_stats", methods=["GET"])
def cache_stats():

    return jsonify({
        "cache_size": len(prediction_cache),
        "cache_limit": CACHE_SIZE,
        "ttl_seconds": CACHE_TTL
    })


##################################################
# Main detection endpoint
##################################################

@app.route("/analyze", methods=["POST"])
def analyze():

    try:

        email_json = request.json

        processed = process_email_json(email_json)

        email_text = processed["clean_text"]
        url_features = processed["url_features"]
        image_features = processed["image_features"]

        cache_key = email_text[:500]

        if cache_key in prediction_cache:

            prediction = prediction_cache[cache_key]

        else:

            prediction = call_space_model(email_text)

            prediction_cache[cache_key] = prediction

        prediction = apply_heuristics(
            prediction,
            url_features,
            image_features
        )

        return jsonify(prediction)

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


##################################################
# Feedback endpoint
##################################################

@app.route("/feedback", methods=["POST"])
def feedback():

    try:

        data = request.json

        return jsonify({
            "status": "feedback stored",
            "true_label": data.get("true_label")
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


##################################################
# Local run
##################################################

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(host="0.0.0.0", port=port)
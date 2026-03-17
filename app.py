import os
import uuid
import requests
from flask import Flask, request, jsonify
from cachetools import TTLCache

from email_preprocessing import process_email_json
from database import store_email


##################################################
# Flask App
##################################################

app = Flask(__name__)


##################################################
# Configuration
##################################################

# Gradio 5.x uses a two-step event protocol under /gradio_api/call/
SPACE_BASE = os.environ.get(
    "SPACE_BASE",
    "https://savizzz-35-phishguard-inference.hf.space"
)

CACHE_SIZE    = int(os.environ.get("CACHE_SIZE", 2000))
CACHE_TTL     = int(os.environ.get("CACHE_TTL", 3600))
SPACE_TIMEOUT = int(os.environ.get("SPACE_TIMEOUT", 60))


##################################################
# Prediction Cache
##################################################

prediction_cache = TTLCache(maxsize=CACHE_SIZE, ttl=CACHE_TTL)


##################################################
# HuggingFace Gradio Space Inference
#
# The Space's run_detection() returns a plain dict:
#   {
#     "phishing_probability": float,
#     "risk_level": "GREEN" | "YELLOW" | "RED",
#     "model_outputs": { "lr": float, ... }
#   }
#
# Gradio wraps that in: {"data": [ <your dict> ], ...}
# So we must unwrap result["data"][0].
#
# If the Space returns {"error": ...} we surface it cleanly.
##################################################

def call_space_model(email_text):
    """
    Gradio 5.x two-step event protocol:
      Step 1 — POST /gradio_api/call/run_detection  →  {"event_id": "..."}
      Step 2 — GET  /gradio_api/call/run_detection/{event_id}  →  SSE stream
                    last event line: data: {"data": [<result>]}
    """

    call_url   = f"{SPACE_BASE}/gradio_api/call/run_detection"
    headers    = {"Content-Type": "application/json"}

    # --- Step 1: submit the job ---
    r1 = requests.post(
        call_url,
        json={"data": [email_text]},
        headers=headers,
        timeout=SPACE_TIMEOUT
    )
    r1.raise_for_status()

    event_id = r1.json().get("event_id")
    if not event_id:
        raise Exception(f"No event_id in Gradio response: {r1.text}")

    # --- Step 2: poll the SSE result stream ---
    result_url = f"{call_url}/{event_id}"
    r2 = requests.get(result_url, stream=True, timeout=SPACE_TIMEOUT)
    r2.raise_for_status()

    last_data_line = None
    for raw_line in r2.iter_lines():
        if not raw_line:
            continue
        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
        if line.startswith("data:"):
            last_data_line = line[len("data:"):].strip()

    if not last_data_line:
        raise Exception("No data received from Gradio SSE stream")

    import json as _json
    envelope = _json.loads(last_data_line)

    # Gradio SSE data line is either:
    #   a list directly: [{...}]
    #   or a dict:       {"data": [{...}]}
    if isinstance(envelope, list) and envelope:
        result = envelope[0]
    elif isinstance(envelope, dict) and "data" in envelope and envelope["data"]:
        result = envelope["data"][0]
    else:
        raise Exception(f"Unexpected Gradio SSE envelope: {envelope}")

    if isinstance(result, dict) and "error" in result:
        raise Exception(f"Space inference error: {result['error']}")

    if not isinstance(result, dict) or "phishing_probability" not in result:
        raise Exception(f"Missing phishing_probability in Space response: {result}")

    return result


##################################################
# Heuristic Risk Adjustment
#
# Nudges the Space's probability score upward based
# on structural URL/image signals, then re-derives
# risk_level and label so they stay consistent.
##################################################

def apply_heuristics(prediction, url_features, image_features):

    score = float(prediction.get("phishing_probability", 0))

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
        label     = "SAFE"
    elif score < 0.7:
        risk_level = "YELLOW"
        label     = "SUSPICIOUS"
    else:
        risk_level = "RED"
        label     = "PHISHING"

    prediction["phishing_probability"] = score
    prediction["risk_level"]           = risk_level
    prediction["label"]                = label

    return prediction


##################################################
# Health Check
##################################################

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "PhishGuard backend running"})


##################################################
# Cache Statistics
##################################################

@app.route("/cache_stats", methods=["GET"])
def cache_stats():
    return jsonify({
        "cache_size":  len(prediction_cache),
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
            return jsonify({"error": "No JSON received"}), 400

        # --------------------------------------------------
        # Preprocessing
        # --------------------------------------------------
        processed     = process_email_json(email_json)
        email_text    = processed.get("clean_text", "")
        url_features  = processed.get("url_features", {})
        image_features = processed.get("image_features", {})

        if not email_text:
            return jsonify({"error": "Email text extraction failed"}), 400

        # --------------------------------------------------
        # Cache lookup
        # --------------------------------------------------
        cache_key = email_text[:500]

        if cache_key in prediction_cache:
            prediction = prediction_cache[cache_key]
        else:
            prediction = call_space_model(email_text)
            prediction_cache[cache_key] = prediction

        # --------------------------------------------------
        # Heuristic adjustment
        # --------------------------------------------------
        prediction = apply_heuristics(prediction, url_features, image_features)

        # --------------------------------------------------
        # Build extension-compatible response
        # --------------------------------------------------
        email_id = str(uuid.uuid4())

        response_body = {
            "risk_level":           prediction.get("risk_level", "GRAY"),
            "phishing_probability": float(prediction.get("phishing_probability", 0)),
            "label":                prediction.get("label", "UNKNOWN"),
            "explanation":          "AI analysis of email content and links.",
            "email_id":             email_id,
            "model_outputs":        prediction.get("model_outputs", {})
        }

        # --------------------------------------------------
        # Persist to MongoDB
        # --------------------------------------------------
        store_email({
            "type":                 "analysis",
            "email_id":             email_id,
            "sender":               processed.get("sender", ""),
            "subject":              processed.get("subject", ""),
            "risk_level":           response_body["risk_level"],
            "label":                response_body["label"],
            "phishing_probability": response_body["phishing_probability"],
            "model_outputs":        response_body["model_outputs"],
            "url_features":         url_features,
            "image_features":       image_features,
        })

        return jsonify(response_body)

    except Exception as e:
        print("Analyze error:", e)
        return jsonify({"error": str(e)}), 500


##################################################
# Feedback Endpoint
##################################################

@app.route("/feedback", methods=["POST"])
def feedback():

    try:

        data = request.get_json(silent=True)

        if not data:
            return jsonify({"error": "No JSON received"}), 400

        email_id   = data.get("email_id")
        true_label = data.get("true_label")

        store_email({
            "type":       "feedback",
            "email_id":   email_id,
            "true_label": true_label
        })

        return jsonify({
            "status":     "feedback received",
            "email_id":   email_id,
            "true_label": true_label
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


##################################################
# Run App (Render compatible)
##################################################

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
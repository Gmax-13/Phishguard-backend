import os
import json
import uuid
import requests
from flask import Flask, request, jsonify
from cachetools import TTLCache

from email_preprocessing import process_email_json
from database import store_email
from rl_weights import get_weights, apply_feedback


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
SPACE_TIMEOUT = int(os.environ.get("SPACE_TIMEOUT", 120))


##################################################
# Prediction Cache
##################################################

prediction_cache = TTLCache(maxsize=CACHE_SIZE, ttl=CACHE_TTL)


##################################################
# HuggingFace Gradio Space Inference
##################################################

def call_space_model(email_text):
    """
    Gradio 5.x two-step event protocol:
      Step 1 — POST /gradio_api/call/run_detection  ->  {"event_id": "..."}
      Step 2 — GET  /gradio_api/call/run_detection/{event_id}  ->  SSE stream
    """

    call_url = f"{SPACE_BASE}/gradio_api/call/run_detection"
    headers  = {"Content-Type": "application/json"}

    # Step 1: submit the job
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

    # Step 2: stream the result
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

    envelope = json.loads(last_data_line)

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
# RL-Weighted Ensemble Score
#
# The HF Space computes per-model probabilities using
# its own fixed internal weights. We re-combine those
# per-model probabilities here using the RL-trained
# weights so the ensemble score reflects learned policy.
##################################################

def apply_rl_weights(model_outputs: dict) -> float:
    """
    Re-combine per-model phishing probabilities using current RL weights.
    Returns a float in [0, 1].
    """
    weights = get_weights()
    total_w = sum(weights.values())
    score   = sum(
        weights.get(m, 0) * model_outputs.get(m, 0.5)
        for m in weights
    )
    return score / total_w


##################################################
# Heuristic Risk Adjustment
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
        label      = "SAFE"
    elif score < 0.7:
        risk_level = "YELLOW"
        label      = "SUSPICIOUS"
    else:
        risk_level = "RED"
        label      = "PHISHING"

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
# Warmup Endpoint
##################################################

@app.route("/warmup", methods=["GET"])
def warmup():
    try:
        r = requests.get(
            f"{SPACE_BASE}/gradio_api/info",
            timeout=30
        )
        space_status = "awake" if r.status_code == 200 else f"status {r.status_code}"
    except Exception as e:
        space_status = f"unreachable: {str(e)}"

    return jsonify({
        "backend":  "awake",
        "hf_space": space_status
    })


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
# RL Weight Inspection
# GET /weights to see the current live weights
##################################################

@app.route("/weights", methods=["GET"])
def weights_endpoint():
    return jsonify({
        "weights": get_weights()
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

        # Preprocessing
        processed      = process_email_json(email_json)
        email_text     = processed.get("clean_text", "")
        url_features   = processed.get("url_features", {})
        image_features = processed.get("image_features", {})

        if not email_text:
            return jsonify({"error": "Email text extraction failed"}), 400

        # Cache lookup
        cache_key = email_text[:500]

        if cache_key in prediction_cache:
            prediction = prediction_cache[cache_key]
        else:
            prediction = call_space_model(email_text)
            prediction_cache[cache_key] = prediction

        # Re-score using RL weights
        # The HF Space returns raw per-model probabilities.
        # We override phishing_probability with the RL-weighted combo.
        model_outputs = prediction.get("model_outputs", {})

        if model_outputs:
            rl_score = apply_rl_weights(model_outputs)
            prediction["phishing_probability"] = rl_score

        # Heuristic adjustment
        prediction = apply_heuristics(prediction, url_features, image_features)

        # Build response
        email_id = str(uuid.uuid4())

        response_body = {
            "risk_level":           prediction.get("risk_level", "GRAY"),
            "phishing_probability": float(prediction.get("phishing_probability", 0)),
            "label":                prediction.get("label", "UNKNOWN"),
            "explanation":          "AI analysis of email content and links.",
            "email_id":             email_id,
            "model_outputs":        model_outputs,
            "weights_used":         get_weights(),
        }

        # Persist to MongoDB
        store_email({
            "type":                 "analysis",
            "email_id":             email_id,
            "sender":               processed.get("sender", ""),
            "subject":              processed.get("subject", ""),
            "risk_level":           response_body["risk_level"],
            "label":                response_body["label"],
            "phishing_probability": response_body["phishing_probability"],
            "model_outputs":        model_outputs,
            "weights_used":         response_body["weights_used"],
            "url_features":         url_features,
            "image_features":       image_features,
        })

        return jsonify(response_body)

    except Exception as e:
        print("Analyze error:", e)
        return jsonify({"error": str(e)}), 500


##################################################
# Feedback Endpoint
# Stores feedback AND triggers RL weight update
##################################################

@app.route("/feedback", methods=["POST"])
def feedback():

    try:

        data = request.get_json(silent=True)

        if not data:
            return jsonify({"error": "No JSON received"}), 400

        email_id   = data.get("email_id")
        true_label = data.get("true_label")

        if not true_label:
            return jsonify({"error": "true_label is required"}), 400

        # Look up the original analysis record to get
        # model_outputs and predicted_label for the RL update
        from database import emails_collection
        analysis_doc = None

        if email_id:
            analysis_doc = emails_collection.find_one(
                {"email_id": email_id, "type": "analysis"}
            )

        model_outputs   = analysis_doc.get("model_outputs", {})  if analysis_doc else {}
        predicted_label = analysis_doc.get("label", "UNKNOWN")   if analysis_doc else "UNKNOWN"

        # RL weight update
        rl_meta = None

        if model_outputs:
            rl_meta = apply_feedback(
                model_outputs   = model_outputs,
                true_label      = true_label,
                predicted_label = predicted_label,
                email_id        = email_id,
            )
        else:
            print(f"Feedback {email_id}: no model_outputs found, skipping RL update")

        # Store feedback document
        store_email({
            "type":            "feedback",
            "email_id":        email_id,
            "true_label":      true_label,
            "predicted_label": predicted_label,
            "rl_update_done":  rl_meta is not None,
        })

        response = {
            "status":         "feedback received",
            "email_id":       email_id,
            "true_label":     true_label,
            "rl_update_done": rl_meta is not None,
            "new_weights":    get_weights(),
        }

        if rl_meta:
            response["old_weights"] = rl_meta.get("old_weights")

        return jsonify(response)

    except Exception as e:
        print("Feedback error:", e)
        return jsonify({"error": str(e)}), 500


##################################################
# Run App (Render compatible)
##################################################

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
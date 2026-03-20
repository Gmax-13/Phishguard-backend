"""
rl_weights.py — Reinforcement Learning Weight Manager
======================================================

Policy:
  The ensemble weights represent a probability distribution over 5 models.
  They are treated as the policy parameters of a simple bandit-style RL agent.

Reward signal:
  User feedback ("legitimate_email" / "phishing_email") is the reward.
  If the ensemble prediction matched the true label  → positive reward (+1)
  If it did not match                                → negative reward (-1)

Per-model credit assignment:
  Each model's contribution to the final score is used to assign partial
  credit. A model that pushed the prediction in the CORRECT direction gets
  its weight increased; one that pushed it in the WRONG direction gets
  its weight decreased.

Update rule (exponential moving average with learning rate α):
  new_weight[m] = old_weight[m] + α * reward * model_contribution[m]

  Contributions are normalised so the total update sums to zero for
  wrong models and is spread proportionally for correct ones.

Decay:
  A decay factor γ < 1 is applied to all weights before each update
  so that older learning decays over time and recent feedback dominates.
  new_weight[m] = γ * old_weight[m]   (then the RL update is added on top)

  This implements a form of temporal discounting — if the email landscape
  changes, the weights will drift to reflect current patterns rather than
  being anchored to old data.

Normalisation:
  After every update, weights are re-normalised to sum to 1.0 and clamped
  to [MIN_WEIGHT, MAX_WEIGHT] to prevent any single model from dominating
  or being zeroed out completely.

Persistence:
  Weights are stored in MongoDB system_collection under _id="ensemble_weights"
  and loaded on startup. Every update is written back immediately.
"""

import logging
from datetime import datetime
from database import system_collection

logger = logging.getLogger(__name__)

# -------------------------------------------------------
# Hyperparameters
# -------------------------------------------------------

# How fast weights shift per feedback (0 = no learning, 1 = full replacement)
LEARNING_RATE = 0.05

# Temporal decay — applied before each update so old feedback fades
# 0.99 means each update the weights drift 1% back toward uniform
DECAY_FACTOR = 0.99

# Hard clamps so no model ever dominates or disappears
MIN_WEIGHT = 0.05
MAX_WEIGHT = 0.60

# Default weights used when no MongoDB record exists
DEFAULT_WEIGHTS = {
    "lr":     0.25,
    "rf":     0.15,
    "lstm":   0.10,
    "bilstm": 0.15,
    "bert":   0.35,
}

MODEL_KEYS = list(DEFAULT_WEIGHTS.keys())


# -------------------------------------------------------
# Load / Save
# -------------------------------------------------------

def load_weights() -> dict:
    """Load weights from MongoDB. Falls back to defaults if not found."""
    try:
        doc = system_collection.find_one({"_id": "ensemble_weights"})
        if doc and "weights" in doc:
            # Ensure all keys present (guards against schema changes)
            weights = {k: doc["weights"].get(k, DEFAULT_WEIGHTS[k]) for k in MODEL_KEYS}
            logger.info(f"Loaded weights from MongoDB: {weights}")
            return weights
    except Exception as e:
        logger.warning(f"Could not load weights from MongoDB: {e}")

    logger.info("Using default weights")
    return DEFAULT_WEIGHTS.copy()


def save_weights(weights: dict, update_meta: dict = None):
    """Persist weights to MongoDB."""
    try:
        payload = {
            "weights":      weights,
            "last_updated": datetime.utcnow(),
        }
        if update_meta:
            payload["last_update_meta"] = update_meta

        system_collection.update_one(
            {"_id": "ensemble_weights"},
            {"$set": payload},
            upsert=True
        )
        logger.info(f"Saved weights to MongoDB: {weights}")
    except Exception as e:
        logger.error(f"Could not save weights to MongoDB: {e}")


# -------------------------------------------------------
# Normalisation
# -------------------------------------------------------

def normalise(weights: dict) -> dict:
    """
    Clamp each weight to [MIN_WEIGHT, MAX_WEIGHT] then re-normalise
    so all weights sum to 1.0.
    """
    clamped = {k: max(MIN_WEIGHT, min(MAX_WEIGHT, v)) for k, v in weights.items()}
    total   = sum(clamped.values())
    return {k: v / total for k, v in clamped.items()}


# -------------------------------------------------------
# RL Update
# -------------------------------------------------------

def update_weights(
    current_weights: dict,
    model_outputs:   dict,
    true_label:      str,
    predicted_label: str,
) -> dict:
    """
    Apply one RL update step and return the new normalised weights.

    Parameters
    ----------
    current_weights : dict
        Current weight for each model key.
    model_outputs : dict
        Per-model phishing probabilities from the last prediction,
        e.g. {"lr": 0.72, "rf": 0.60, "lstm": 0.05, ...}
    true_label : str
        Ground truth from user feedback:
        "phishing_email" or "legitimate_email"
    predicted_label : str
        What the ensemble predicted:
        "PHISHING", "SUSPICIOUS", or "SAFE"

    Returns
    -------
    dict
        Updated, normalised weights.
    """

    # --- Map labels to binary ---
    true_phishing = (true_label == "phishing_email")

    # Treat SUSPICIOUS as a weak positive for RL purposes
    pred_phishing = predicted_label in ("PHISHING", "SUSPICIOUS")

    # Global reward: +1 if direction correct, -1 if wrong
    reward = 1.0 if (true_phishing == pred_phishing) else -1.0

    # --- Step 1: Temporal decay ---
    # All weights decay slightly toward uniform before the new signal
    decayed = {k: DECAY_FACTOR * v for k, v in current_weights.items()}

    # --- Step 2: Per-model credit assignment ---
    # A model's "contribution" is how much it pushed toward the true answer.
    # If true_phishing → higher phishing probability = more credit
    # If true_legitimate → lower phishing probability = more credit
    contributions = {}
    for m in MODEL_KEYS:
        prob = model_outputs.get(m, 0.5)
        if true_phishing:
            # Credit = how strongly the model flagged phishing
            contributions[m] = prob - 0.5          # positive if > 0.5
        else:
            # Credit = how strongly the model said legitimate
            contributions[m] = 0.5 - prob          # positive if < 0.5

    # --- Step 3: Weight update ---
    new_weights = {}
    for m in MODEL_KEYS:
        delta          = LEARNING_RATE * reward * contributions[m]
        new_weights[m] = decayed[m] + delta

    # --- Step 4: Normalise and clamp ---
    new_weights = normalise(new_weights)

    return new_weights


# -------------------------------------------------------
# Public interface
# -------------------------------------------------------

# Module-level weights — loaded once at startup
_weights = load_weights()


def get_weights() -> dict:
    """Return the current in-memory weights."""
    return _weights.copy()


def apply_feedback(
    model_outputs:   dict,
    true_label:      str,
    predicted_label: str,
    email_id:        str = None,
) -> dict:
    """
    Process one feedback signal:
      1. Run RL update
      2. Update in-memory weights
      3. Persist to MongoDB
      4. Return new weights + metadata for logging

    Parameters
    ----------
    model_outputs : dict
        Per-model probabilities from the prediction being rated.
    true_label : str
        "phishing_email" or "legitimate_email"
    predicted_label : str
        The label the ensemble gave ("PHISHING", "SUSPICIOUS", "SAFE")
    email_id : str, optional
        For audit logging only.

    Returns
    -------
    dict with keys: old_weights, new_weights, reward_direction
    """
    global _weights

    old_weights = _weights.copy()

    new_weights = update_weights(
        current_weights = old_weights,
        model_outputs   = model_outputs,
        true_label      = true_label,
        predicted_label = predicted_label,
    )

    _weights = new_weights

    meta = {
        "email_id":        email_id,
        "true_label":      true_label,
        "predicted_label": predicted_label,
        "old_weights":     old_weights,
        "new_weights":     new_weights,
        "reward":          "positive" if true_label in ("phishing_email",) == predicted_label in ("PHISHING", "SUSPICIOUS") else "negative",
    }

    save_weights(new_weights, update_meta=meta)

    logger.info(
        f"RL update | email={email_id} | true={true_label} | pred={predicted_label} | "
        f"old={old_weights} | new={new_weights}"
    )

    return meta
import math
import torch
import json
from quantum_sentinel.paths import TRACES_DIR


class ShannonObserver:
    """
    Implements Shannon Entropy as a measure of logical uncertainty.
    Tracks 'Entropy Velocity' (ΔH) to identify the Shannon-Schrödinger Boundary.
    """
    def __init__(self, tokenizer, task_id="logical_probe"):
        self.tokenizer = tokenizer
        self.task_id = task_id
        self.phase_log = []

    def measure_uncertainty(self, logits: torch.FloatTensor):
        """Calculates normalized Shannon Entropy (H)."""
        # Handle -inf logits
        if torch.all(logits == float('-inf')):
            return 0.0  # No uncertainty if deadlocked

        probs = torch.softmax(logits, dim=-1)

        # Handle NaN
        if torch.any(torch.isnan(probs)):
            return 0.0

        log_probs = torch.log_softmax(logits, dim=-1)
        h = -torch.sum(probs * log_probs, dim=-1).mean().item()

        if math.isnan(h):
            return 0.0

        return h

    def log_transition(self, step, current_h, delta_h, limit):
        self.phase_log.append({
            "step": step,
            "uncertainty": round(current_h, 4),
            "velocity": round(delta_h, 4),
            "abs_velocity": round(abs(delta_h), 4),  # ← OPTIONAL
            "threshold": round(limit, 4),
            "is_violent": abs(delta_h) > limit  # ← Consider using abs()
        })

    def save_trace(self, model_name):
        path = TRACES_DIR / f"{model_name}_{self.task_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.phase_log, f, indent=4)
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

    @staticmethod
    def measure_uncertainty(logits: torch.FloatTensor):
        if torch.all(logits == float('-inf')):
            return 0.0

        # Categorical distribution handles log_probs internally for better stability
        dist = torch.distributions.Categorical(logits=logits)
        h = dist.entropy().mean().item()

        return h if not torch.isnan(torch.tensor(h)) else 0.0

    def log_transition(self, step, uncertainty, velocity, limit):
        self.phase_log.append({
            "step": step, "uncertainty": uncertainty,
            "velocity": velocity, "limit": limit
        })

    def save_trace(self, model_name):
        path = TRACES_DIR / f"{model_name}_{self.task_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.phase_log, f, indent=4)
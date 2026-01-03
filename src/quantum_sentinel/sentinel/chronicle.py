import torch
import logging
import json
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

class SentinelAligner:
    """Monitors model coherence using normalized Shannon Entropy."""
    def __init__(self, tokenizer, task_type="logical_deduction"):
        self.tokenizer = tokenizer
        self.task_type = task_type
        self.chronicle = [[]]
        self.failed_thoughts = []

    def calculate_alignment_distance(self, logits: torch.FloatTensor):
        """Calculates Shannon Entropy. Normalized version used in Weaver."""
        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        return torch.mean(entropy).item()

    def surgical_chronicle_slice(self, thread_idx: int, steps: int = 1):
        """Moves rejected tokens from active chronicle to failed_thoughts archive."""
        if thread_idx < len(self.chronicle):
            for _ in range(steps):
                if self.chronicle[thread_idx]:
                    bad_thought = self.chronicle[thread_idx].pop()
                    self.failed_thoughts.append(bad_thought)
            logger.info("✂️ Chronicle: Divergent thought moved to archive.")

    def archive_step(self, thread_idx: int, token_id: int, distance: float, is_candidate=False):
        """Records a token. Set is_candidate=True for rejected tokens."""
        entry = {
            "token": self.tokenizer.decode([token_id]),
            "token_id": token_id,
            "dist_normalized": round(distance, 4)
        }
        if is_candidate:
            self.failed_thoughts.append(entry)
        else:
            self.chronicle[thread_idx].append(entry)

    def export_research_log(self, path="results/alignment_session.json"):
        """Seals the session into a JSON record."""
        Path("results").mkdir(exist_ok=True)
        data = {
            "final_thread": self.chronicle,
            "failed_attempts": self.failed_thoughts,
            "metrics": {"backtracks": len(self.failed_thoughts), "task": self.task_type}
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"\n💾 [ARCHIVE SEALED] Data saved to: {path}")
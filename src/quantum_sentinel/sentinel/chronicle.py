import torch
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class SentinelAligner:
    """Monitors model coherence using normalized Shannon Entropy."""
    def __init__(self, tokenizer, task_type="logical_deduction"):
        self.tokenizer = tokenizer
        self.task_type = task_type
        self.chronicle = [[]]
        self.failed_thoughts = []
        # Initialize phase log for experimental tracking
        self.phase_log = []

    def calculate_alignment_distance(self, logits: torch.FloatTensor):
        """Calculates Shannon Entropy with numerical stability using log_softmax."""
        # Grounding the distribution to prevent log(0) instability
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)

        # Entropy as a measure of Entanglement
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return torch.mean(entropy).item()

    def record_multiverse_step(self, step, realities, is_rejected=False):
        if not hasattr(self, 'multiverse_log'): self.multiverse_log = []
        self.multiverse_log.append({
            "step": step,
            "realities": realities,
            "rejected_event": is_rejected
        })

    def print_multiverse_report(self):
        print("\n🌌 [MULTIVERSE TRACE] Evidence of Deterministic Branching:")
        for log in self.multiverse_log:
            marker = "❌ REJECTED" if log['rejected_event'] else "✅ RESOLVED"
            primary = log['realities'][0]
            ghosts = ", ".join([f"{r['token']} ({r['prob']:.2f})" for r in log['realities'][1:3]])
            print(f"Step {log['step']} {marker}: '{primary['token']}' | Ghosts: {ghosts}")

    def record_phase_transition(self, step: int, current_h: float, future_h: float, delta_h: float, limit: float):
        """Logs the entropy velocity while filtering for numerical singularities."""
        # 🌀 Numerical Grounding: If future_h failed, we treat it as maximum decoherence
        # rather than a NaN to preserve the Peak Delta H signature.
        safe_future_h = future_h if not torch.isnan(torch.tensor(future_h)) else 1.0
        safe_delta_h = safe_future_h - current_h

        entry = {
            "step": step,
            "current_entropy": round(current_h, 4),
            "future_entropy": round(safe_future_h, 4),
            "velocity": round(safe_delta_h, 4),
            "threshold": round(limit, 4),
            "is_divergent": safe_delta_h > limit
        }
        self.phase_log.append(entry)

    def surgical_chronicle_slice(self, thread_idx: int, steps: int = 1):
        """
        Surgically removes the last N steps from the active logical thread.
        Used when the Sentinel detects a delayed decoherence event.
        """
        if thread_idx < len(self.chronicle):
            for _ in range(steps):
                if self.chronicle[thread_idx]:
                    # Move 'Decohered' thought to the failed archive for research analysis
                    self.failed_thoughts.append(self.chronicle[thread_idx].pop())
            logger.info(f"✂️ Removed {steps} step(s) from Thread {thread_idx} due to Decoherence.")

    def archive_step(self, thread_idx: int, token_id: int, distance: float, is_rejected=False):
        """
        Records a token into the chronicle.
        If is_rejected=True, it bypasses the active thread and goes to failed_thoughts.
        """
        entry = {
            "token": self.tokenizer.decode([token_id]),
            "token_id": token_id,
            "h_normalized": round(distance, 4)  # Renamed to reflect entropy (H)
        }

        if is_rejected:
            self.failed_thoughts.append(entry)
        else:
            # Ensure the thread exists before appending
            while thread_idx >= len(self.chronicle):
                self.chronicle.append([])
            self.chronicle[thread_idx].append(entry)

    def export_research_log(self, path="results/alignment_session.json"):
        """Seals the session into a JSON record."""
        Path("results").mkdir(exist_ok=True)
        data = {
            "final_thread": self.chronicle,
            "failed_attempts": self.failed_thoughts,
            "phase_transition_data": self.phase_log,
            "metrics": {
                "backtracks": len(self.failed_thoughts),
                "task": self.task_type
            }
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"\n💾 [ARCHIVE SEALED] Data saved to: {path}")
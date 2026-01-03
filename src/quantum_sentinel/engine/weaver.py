import torch
import logging
from typing import Optional, List
from quantum_sentinel.sentinel.chronicle import SentinelAligner

logger = logging.getLogger(__name__)

class QuantumLogitProcessor:
    def __init__(self, temperature=0.7):
        self.temperature = temperature

    def __call__(self, logits: torch.FloatTensor, forbidden_ids: List[int]):
        scores = logits.clone() / self.temperature
        if forbidden_ids:
            scores[..., forbidden_ids] = -float('inf')
        return scores


class BacktrackingWeaver:
    def __init__(self, model, tokenizer, sentinel: SentinelAligner, processor: QuantumLogitProcessor):
        self.model = model
        self.tokenizer = tokenizer
        self.sentinel = sentinel
        self.processor = processor
        self.reference_distribution = None

    def _get_future_entropy(self, current_ids, candidate_token_id, past_key_values, steps=2):
        """
        Quantum Look-Ahead: Peeks into the future to see if a token
        leads to logical decoherence later on.
        """
        temp_ids = torch.cat([current_ids, torch.tensor([[candidate_token_id]]).to(current_ids.device)], dim=-1)
        temp_kv = past_key_values
        total_future_h = 0

        with torch.no_grad():
            for _ in range(steps):
                outputs = self.model(temp_ids[:, -1:], past_key_values=temp_kv, use_cache=True)
                logits = outputs.logits[:, -1, :]
                temp_kv = outputs.past_key_values

                # Calculate normalized entropy of this potential future
                probs = torch.softmax(logits, dim=-1)
                h = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean().item()
                total_future_h += h

                # Greedy step for the simulation
                next_tok = torch.argmax(logits, dim=-1, keepdim=True)
                temp_ids = torch.cat([temp_ids, next_tok], dim=-1)

        return total_future_h / steps

    def generate_with_alignment(self, input_ids, max_tokens=50, entropy_threshold=0.35, kl_threshold=15.0):
        current_ids = input_ids.clone()
        past_key_values = None
        backtrack_count = 0
        max_h = torch.log(torch.tensor(self.model.config.vocab_size)).item()

        for i in range(max_tokens):
            outputs = self.model(
                current_ids if past_key_values is None else current_ids[:, -1:],
                past_key_values=past_key_values,
                use_cache=True
            )
            logits = outputs.logits[:, -1, :]

            # --- Capture Initial Mode ---
            if i == 0:
                self.reference_distribution = torch.softmax(logits, dim=-1).detach()

            # --- Candidate Selection with Look-Ahead ---
            forbidden_ids = []
            while len(forbidden_ids) < 5:
                # Sample a candidate
                processed_logits = self.processor(logits, forbidden_ids)
                candidate_token = torch.multinomial(torch.softmax(processed_logits, dim=-1), 1).item()

                # 1. Check current entropy
                current_h = self.sentinel.calculate_alignment_distance(logits) / max_h

                # 2. Check future entropy (The Look-Ahead)
                # This is the "Quantum Probe"
                future_h = self._get_future_entropy(current_ids, candidate_token, outputs.past_key_values) / max_h

                # 3. Decision: If the future looks like a mess, reject the current token
                # Even if the model is confident (low current_h),
                # a high future_h suggests a logical dead-end.
                if current_h > entropy_threshold or future_h > (entropy_threshold * 1.5):
                    logger.info(
                        f"🌀 [INTERFERENCE] Rejecting '{self.tokenizer.decode([candidate_token])}' (Future H: {future_h:.3f})")
                    forbidden_ids.append(candidate_token)
                    backtrack_count += 1
                else:
                    next_token = torch.tensor([[candidate_token]]).to(current_ids.device)
                    break

            # Update state
            current_ids = torch.cat([current_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values

            # --- Global Mode-Shift Stop ---
            curr_prob = torch.softmax(logits, dim=-1)
            kl_div = torch.sum(self.reference_distribution * (
                    torch.log(self.reference_distribution + 1e-10) - torch.log(curr_prob + 1e-10)
            ), dim=-1).mean().item()

            if kl_div > kl_threshold:
                logger.info(f"🛑 [MODE SHIFT] Exit detected. Stopping.")
                break

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        return {"sequences": current_ids, "backtracks": backtrack_count}
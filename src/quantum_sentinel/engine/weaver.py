import torch
import logging
from typing import List

logger = logging.getLogger(__name__)


class QuantumLogitProcessor:
    def __init__(self, temperature=0.1, repetition_penalty=1.5):
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty

    def __call__(self, logits: torch.FloatTensor, forbidden_ids: List[int], current_ids: torch.LongTensor):
        # 1. Scaling the distribution (Thermal grounding)
        scores = logits.clone() / self.temperature

        # 2. Apply Repetition Penalty (Schrödinger Damping)
        # Prevents local minima traps like the 'can's's' loop
        if current_ids is not None:
            for token_id in set(current_ids[0].tolist()):
                if scores[..., token_id] < 0:
                    scores[..., token_id] *= self.repetition_penalty
                else:
                    scores[..., token_id] /= self.repetition_penalty

        # 3. Apply Forbidden Token Mask (The Sentinel's Intervention)
        if forbidden_ids:
            scores[..., forbidden_ids] = -float('inf')
        return scores


class BacktrackingWeaver:
    def __init__(self, model, tokenizer, sentinel, processor):
        self.model = model
        self.tokenizer = tokenizer
        self.sentinel = sentinel
        self.processor = processor
        self.reference_distribution = None
        self.baseline_h_flow = []

    def _get_future_entropy(self, current_ids, candidate_token_id, past_key_values, steps=2):
        """Probes the future to detect potential logical decoherence."""
        temp_ids = torch.cat([current_ids, torch.tensor([[candidate_token_id]]).to(current_ids.device)], dim=-1)
        temp_kv = past_key_values
        total_future_h = 0

        with torch.no_grad():
            for _ in range(steps):
                outputs = self.model(temp_ids[:, -1:], past_key_values=temp_kv, use_cache=True)
                logits = outputs.logits[:, -1, :]
                temp_kv = outputs.past_key_values

                # STABLE LOGIC: Use log_softmax to prevent NaN
                probs = torch.softmax(logits, dim=-1)
                log_probs = torch.log_softmax(logits, dim=-1)
                h = -torch.sum(probs * log_probs, dim=-1).mean().item()

                total_future_h += h

                next_tok = torch.argmax(logits, dim=-1, keepdim=True)
                temp_ids = torch.cat([temp_ids, next_tok], dim=-1)

        return total_future_h / steps

    def _get_multiverse_density(self, logits, k=5):
        """Captures the probability mass of the top K 'Parallel Realities'."""
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, k=k)

        return [{
            "token": self.tokenizer.decode([top_indices[0, i].item()]),
            "prob": round(top_probs[0, i].item(), 4)
        } for i in range(k)]

    def generate_with_alignment(self, input_ids, max_tokens=50, kl_threshold=15.0):
        current_ids = input_ids.clone()
        past_key_values = None
        backtrack_count = 0
        # Numerical grounding for entropy normalization
        max_h = torch.log(torch.tensor(self.model.config.vocab_size)).item()

        self.baseline_h_flow = []
        ghost_ids = []  # Tracks suppressed 'Truth' candidates

        for i in range(max_tokens):
            outputs = self.model(
                current_ids if past_key_values is None else current_ids[:, -1:],
                past_key_values=past_key_values,
                use_cache=True
            )
            logits = outputs.logits[:, -1, :]
            multiverse_state = self._get_multiverse_density(logits)

            if i == 0:
                self.reference_distribution = torch.softmax(logits, dim=-1).detach()

            # Fix: Epsilon grounding to prevent 'nan' during phase transitions
            current_h = self.sentinel.calculate_alignment_distance(logits) / max_h

            # Unified Threshold Calibration
            if i < 3:
                self.baseline_h_flow.append(current_h) # Record early stability
                divergence_limit = 0.15
            else:
                diffs = [abs(self.baseline_h_flow[j] - self.baseline_h_flow[j - 1])
                         for j in range(1, len(self.baseline_h_flow))]
                avg_jitter = sum(diffs) / len(diffs) if diffs else 0.05
                divergence_limit = max(0.05, avg_jitter * 2.0)

            forbidden_ids = []
            next_token = None

            while len(forbidden_ids) < 5:
                processed_logits = self.processor(logits, forbidden_ids, current_ids)

                # 🌀 ADAPTIVE SCALING: Adjust 'Injection Energy' based on current Entanglement (H)
                # High noise (current_h > 0.3) triggers a stronger nudge.
                injection_power = 3.5 if current_h > 0.3 else 2.0

                # 🌀 GROUND STATE INJECTION
                if len(ghost_ids) > 0 and past_key_values is None:
                    processed_logits[..., ghost_ids] += injection_power
                    logger.info(
                        f"✨ [INJECTION] Power: {injection_power} | Ghost IDs: {self.tokenizer.decode(ghost_ids)}")

                candidate_token = torch.multinomial(torch.softmax(processed_logits, dim=-1), 1).item()

                # Quantum Probe for future entropy velocity
                future_h = self._get_future_entropy(current_ids, candidate_token, outputs.past_key_values) / max_h
                delta_h = future_h - current_h

                if delta_h > divergence_limit:
                    # Capture Top-3 'Ghost' paths before rejection
                    _, top_indices = torch.topk(torch.softmax(logits, dim=-1), k=3)
                    ghost_ids.extend(top_indices[0].tolist())

                    logger.info(
                        f"🌀 [DECOHERENCE] Rejecting '{self.tokenizer.decode([candidate_token])}' | ΔH: {delta_h:.3f}")

                    self.sentinel.record_phase_transition(i, current_h, future_h, delta_h, divergence_limit)
                    self.sentinel.record_multiverse_step(i, multiverse_state, is_rejected=True)

                    forbidden_ids.append(candidate_token)
                    backtrack_count += 1

                    if len(forbidden_ids) >= 5:
                        logger.warning("🔥 [TOTAL DECOHERENCE] Performing Multiverse Flush + Injection.")
                        past_key_values = None  # Unitary Reset
                        self.processor.temperature += 0.05  # Thermal Damping
                        torch.cuda.empty_cache()
                        break

                    torch.cuda.empty_cache()
                    logits = logits * 1.1
                else:
                    # Success: Path resolution
                    ghost_ids = []
                    self.sentinel.record_phase_transition(i, current_h, future_h, delta_h, divergence_limit)
                    self.sentinel.record_multiverse_step(i, multiverse_state, is_rejected=False)
                    self.sentinel.archive_step(0, candidate_token, current_h)
                    self.baseline_h_flow.append(current_h)
                    next_token = torch.tensor([[candidate_token]]).to(current_ids.device)
                    break

            if next_token is None:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            current_ids = torch.cat([current_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values

            # Logical Stability Stop
            curr_prob = torch.softmax(logits, dim=-1)
            kl_div = torch.sum(self.reference_distribution * (
                    torch.log(self.reference_distribution + 1e-10) - torch.log(curr_prob + 1e-10)
            ), dim=-1).mean().item()

            if kl_div > kl_threshold or next_token.item() == self.tokenizer.eos_token_id:
                break

        return {"sequences": current_ids, "backtracks": backtrack_count}


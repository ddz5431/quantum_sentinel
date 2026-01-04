import torch


class UnitaryWeaver:
    """
    Evolves the sequence state via Wavefunction Projection.
    When decoherence is detected, it collapses the 'Bad Branch'
    rather than resetting time.
    """

    def __init__(self, model, tokenizer, observer, processor=None):
        self.model = model
        self.tokenizer = tokenizer
        self.observer = observer
        self.processor = processor
        self.max_h = torch.log(torch.tensor(self.model.config.vocab_size)).item()

    def evolve(self, input_ids, max_tokens=20, verbose=False):
        current_ids = input_ids.clone()
        h_history = []
        backtracks = 0
        forbidden_tokens = set()

        for i in range(max_tokens):
            with torch.no_grad():
                outputs = self.model(input_ids=current_ids, use_cache=False)

            logits = outputs.logits[:, -1, :].clone()  # Clone to avoid in-place issues

            # Apply Forbidden Mask (Pruning the Multiverse)
            for token_id in forbidden_tokens:
                logits[:, token_id] = float('-inf')

            # SAFETY: Check if all tokens are forbidden
            if torch.all(logits == float('-inf')):
                if verbose:
                    print(f"⚠️ [DEADLOCK] Step {i}: Clearing forbidden tokens")
                forbidden_tokens.clear()
                logits = outputs.logits[:, -1, :].clone()

            # 1. Observe Uncertainty (Shannon Entropy)
            h = self.observer.measure_uncertainty(logits) / self.max_h
            h_history.append(h)

            # 2. Detect Decoherence (Schrödinger Observation)
            delta_h = h - h_history[-2] if len(h_history) > 1 else 0.0
            limit = 0.20 if i < 3 else (sum([abs(h_history[j] - h_history[j - 1])
                                             for j in range(1, len(h_history))]) / len(h_history)) * 2.5

            self.observer.log_transition(i, h, delta_h, limit)

            # 3. Decision: Branch Collapse
            if abs(delta_h) > limit:
                backtracks += 1
                bad_token = torch.argmax(logits, dim=-1).item()
                forbidden_tokens.add(bad_token)
                logits[:, bad_token] = float('-inf')

                # Check again after forbidding
                if torch.all(logits == float('-inf')):
                    print(f"⚠️ [DEADLOCK] All tokens forbidden at step {i}, clearing forbidden set")
                    forbidden_tokens.clear()
                    # Re-run forward pass without forbidden tokens
                    with torch.no_grad():
                        outputs = self.model(input_ids=current_ids, use_cache=False)
                    logits = outputs.logits[:, -1, :]

                if verbose:
                    print(f"🔥 [COLLAPSE] Step {i}: Pruning '{self.tokenizer.decode(bad_token)}'")

            # 4. Final Projection
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            current_ids = torch.cat([current_ids, next_token], dim=-1)

            # Clear forbidden tokens after stable step
            if abs(delta_h) <= limit:
                forbidden_tokens.clear()

        return {"sequences": current_ids, "backtracks": backtracks}

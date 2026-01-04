import torch


class UnitaryWeaver:
    """
    Evolves the sequence state deterministically.
    Forces a 'Wavefunction Collapse' (KV-Cache Reset) when decoherence is detected.
    """

    def __init__(self, model, tokenizer, observer, processor=None):
        self.model = model
        self.tokenizer = tokenizer
        self.observer = observer
        self.processor = processor
        self.max_h = torch.log(torch.tensor(self.model.config.vocab_size)).item()

    def evolve(self, input_ids, max_tokens=15):
        current_ids = input_ids.clone()
        past_key_values = None
        backtracks = 0
        h_history = []

        for i in range(max_tokens):
            outputs = self.model(
                current_ids if past_key_values is None else current_ids[:, -1:],
                past_key_values=past_key_values,
                use_cache=True
            )
            logits = outputs.logits[:, -1, :]

            # 1. Observe Uncertainty (Shannon)
            h = self.observer.measure_uncertainty(logits) / self.max_h
            h_history.append(h)

            # 2. Calculate Phase Limit (τ)
            delta_h = h - h_history[-2] if len(h_history) > 1 else 0.0
            limit = 0.20 if i < 3 else (sum([abs(h_history[j] - h_history[j - 1]) for j in
                                             range(1, len(h_history))]) / len(h_history)) * 2.5

            self.observer.log_transition(i, h, delta_h, limit)

            # 3. Decision: Unitary Reset or Evolution
            if delta_h > limit:
                past_key_values = None
                backtracks += 1
                logits = logits * 1.2

            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            current_ids = torch.cat([current_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values

        return {"sequences": current_ids, "backtracks": backtracks}
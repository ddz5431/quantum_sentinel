import torch


class UnitaryWeaver:
    """
    UnitaryWeaver: A physics-inspired intervention operator for LLMs.

    It treats logic as a ground state and hallucinations as 'Violent Decoherence'.
    By monitoring entropy velocity (ΔH) and second-order momentum (Δ²H), it
    selectively collapses divergent logical branches.
    """

    def __init__(self, model, tokenizer, observer, alpha=15.0):
        self.model = model
        self.tokenizer = tokenizer
        self.observer = observer
        self.alpha = alpha

        # Normalized uncertainty ceiling
        self.max_h = torch.log(torch.tensor(model.config.vocab_size, dtype=torch.float32)).item()

        # Token set for branch collapse (Sycophancy tokens)
        self.bias_tokens = self._encode_bias_tokens()

        # Phase Transition Parameters (Default for Phi/Gemma-class models)
        self.threshold_trigger = 0.25  # τ: Instantaneous trigger
        self.threshold_integral = 0.5  # ∫ΔH: Cumulative pressure threshold
        self.threshold_stable = 0.03  # Stability window for lockdown release

        # Dynamic Controllers
        self.ema_alpha = 0.3  # Smoothing factor for entropy velocity
        self.logic_latency = 4  # Initial 'Observation-Only' tokens

    def _encode_bias_tokens(self):
        """Pre-encode tokens that signal instructional drift/sycophancy."""
        words = ["Yes", "yes", "Sure", "sure", "Agree", "agree",
                 "Correct", "correct", "Absolutely", "Exactly",
                 "Right", "Indeed", "Definitely", "True"]
        tokens = set()
        for w in words:
            tokens.update(self.tokenizer.encode(w, add_special_tokens=False))
            tokens.update(self.tokenizer.encode(" " + w, add_special_tokens=False))
        return list(tokens)

    def reset(self):
        """Initialize state for a new generation sequence."""
        self.ema_dh = 0.0
        self.last_dh = 0.0
        self.accumulated_dh = 0.0
        self.in_lockdown = False
        self.stable_count = 0

    def step(self, logits, delta_h, step_idx):
        """
        Processes a single token generation step with momentum-aware intervention.
        """
        # 1. Compute Momentum (Second-order derivative)
        d2h = delta_h - self.last_dh
        self.last_dh = delta_h

        # 2. Update EMA Velocity
        self.ema_dh = self.ema_alpha * delta_h + (1 - self.ema_alpha) * self.ema_dh

        # 3. Handle Logic Latency
        if step_idx < self.logic_latency:
            return logits, 1.0

        # 4. Integrate Entropy Pressure
        if delta_h > 0:
            self.accumulated_dh += delta_h

        # 5. Detect Self-Correction (Momentum Check)
        # If ΔH is high but decelerating sharply, the model is 'tunneling' to truth.
        is_self_correcting = delta_h > 0 and d2h < -0.05

        # 6. Evaluate Intervention Conditions
        instant = delta_h > self.threshold_trigger
        cumulative = self.accumulated_dh > self.threshold_integral
        should_intervene = (instant or cumulative) and not is_self_correcting

        # 7. Compute Adaptive Temperature (Heat Pulse)
        # Only apply 'heat' (randomization) if we detect decoherence.
        effective_alpha = self.alpha if should_intervene else self.alpha * 0.05
        # Using EMA to smooth the pulse energy
        temp = torch.clamp(torch.exp(torch.tensor(self.ema_dh * effective_alpha)), 0.6, 3.5)

        # 8. Lockdown Logic (Branch Collapse)
        is_biased = torch.argmax(logits, dim=-1).item() in self.bias_tokens

        # Activate lockdown if entropy or bias signals are breached
        if (temp > 1.25 or is_biased or should_intervene) and not self.in_lockdown:
            self.in_lockdown = True
            self.stable_count = 0

        if self.in_lockdown:
            # Physically block sycophancy-related tokens
            for t in self.bias_tokens:
                if t < logits.shape[-1]:
                    logits[:, t] = float('-inf')

            # Release lockdown only when ΔH stabilizes near zero
            self.stable_count = self.stable_count + 1 if abs(delta_h) < self.threshold_stable else 0
            if self.stable_count >= 3:
                self.in_lockdown = False
                self.accumulated_dh = 0.0  # Reset pressure upon release

        return logits, temp.item()

    def evolve(self, input_ids, max_tokens=20):
        """The generation loop with real-time observer/weaver feedback."""
        self.reset()
        ids = input_ids.clone()
        h_history = []
        interventions = 0
        peak_dh = 0.0

        for i in range(max_tokens):
            with torch.no_grad():
                # Get last token logits
                logits = self.model(input_ids=ids, use_cache=False).logits[:, -1, :].clone()

            # Measure and normalize Shannon Entropy
            h = self.observer.measure_uncertainty(logits) / self.max_h
            h_history.append(h)

            # Compute Uncertainty Velocity (ΔH)
            dh = h - h_history[-2] if len(h_history) > 1 else 0.0
            peak_dh = max(peak_dh, abs(dh))

            # Apply Unitary Intervention
            logits, temp = self.step(logits, dh, i)
            if self.in_lockdown:
                interventions += 1

            # Sample from the modified distribution
            if temp > 1.0:
                probs = torch.softmax(logits / temp, dim=-1)
                token = torch.multinomial(probs + 1e-9, 1)
            else:
                token = torch.argmax(logits, dim=-1, keepdim=True)

            # Append token to stream
            ids = torch.cat([ids, token], dim=-1)

            if token.item() == self.tokenizer.eos_token_id:
                break

        return {
            "sequences": ids,
            "interventions": interventions,
            "peak_delta_h": peak_dh,
            "final_h": h_history[-1] if h_history else 0.0
        }
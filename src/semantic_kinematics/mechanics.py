import torch

from .physics import ShannonObserver
from .topology import EpistemicTopologist, PhaseState


class UnitaryWeaver:
    """
    Intervenes on the generation process based on topological diagnosis.
    Now supports architecture-specific calibration (Renormalization).
    """

    def __init__(self, model, tokenizer, dh_threshold=0.15, coherence_threshold=0.45):
        self.model = model
        self.tokenizer = tokenizer
        self.observer = ShannonObserver(model, tokenizer)

        # FIX: Pass the calibrated thresholds to the Topologist
        self.topologist = EpistemicTopologist(
            self.observer,
            dh_threshold=dh_threshold,
            coherence_threshold=coherence_threshold
        )

        self.bias_tokens = self._load_bias_set()

    def _load_bias_set(self):
        # Pre-calculate token IDs for common sycophantic responses
        words = ["Yes", "Correct", "Agree", "Right", "Sure", "Absolutely"]
        ids = set()
        for w in words:
            ids.update(self.tokenizer.encode(w, add_special_tokens=False))
            ids.update(self.tokenizer.encode(" " + w, add_special_tokens=False))
        return list(ids)

    def evolve(self, input_ids: torch.Tensor, max_tokens: int = 20) -> str:
        """
        Evolves the system forward in time.
        If a Tunnel is detected, applies non-linear intervention.
        """
        # 1. Collapse & Diagnose t=0
        diag = self.topologist.diagnose(input_ids)

        if diag.state == PhaseState.TUNNEL:
            return self._tunnel_dynamics(input_ids, max_tokens, diag)
        else:
            return self._standard_dynamics(input_ids, max_tokens)

    def _tunnel_dynamics(self, input_ids, max_tokens, diag):
        ids = input_ids.clone()
        alpha = diag.alpha
        veto_k = diag.veto_k

        past_kv = None

        # Pre-compute initial state for continuity
        with torch.no_grad():
            outputs = self.model(ids, use_cache=True)
            past_kv = outputs.past_key_values

        generated = []

        for _ in range(max_tokens):
            with torch.no_grad():
                outputs = self.model(ids[:, -1:], past_key_values=past_kv, use_cache=True)
                past_kv = outputs.past_key_values
                logits = outputs.logits[:, -1, :].clone()

            # --- THE INTERVENTION OPERATOR ---

            # 1. Bias Lockdown (Always active in Tunnel to prevent "Yes-Man" behavior)
            # We block explicit agreement tokens.
            if self.bias_tokens:
                logits[:, self.bias_tokens] = float('-inf')

            # 2. Momentum Veto (If enabled by diagnosis)
            if veto_k > 0:
                top_k = torch.topk(logits, veto_k, dim=-1).indices[0]
                logits[:, top_k] = float('-inf')

            # 3. Thermodynamic Scaling (Heat)
            # T = alpha. We flatten the distribution to allow truth to surface.
            temp = alpha
            probs = torch.softmax(logits / temp, dim=-1)

            # Collapse (Sample)
            next_token = torch.multinomial(probs, 1)
            ids = torch.cat([ids, next_token], dim=-1)
            generated.append(next_token.item())

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def _standard_dynamics(self, input_ids, max_tokens):
        """Standard greedy decoding (Ground state evolution)."""
        output = self.model.generate(input_ids, max_new_tokens=max_tokens, do_sample=False)
        return self.tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)
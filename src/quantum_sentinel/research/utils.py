import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from quantum_sentinel.core.shannon_observer import ShannonObserver
from quantum_sentinel.core.unitary_weaver import UnitaryWeaver


def load_model(model_id):
    """Load model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    return model, tokenizer


def unload_model(model, tokenizer):
    """Free GPU memory."""
    del model
    del tokenizer
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def generate(model, tokenizer, system, prompt, alpha=0.0, tau=None, integral=None, latency=None):
    """Generate with optional intervention."""
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": system},
         {"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True
    )
    ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)

    weaver = UnitaryWeaver(model, tokenizer, ShannonObserver(tokenizer), alpha=alpha)

    if tau:
        weaver.threshold_trigger = tau
        weaver.threshold_integral = integral if integral else tau * 2
    if latency:
        weaver.logic_latency = latency
    if alpha == 0:
        weaver.bias_tokens = []

    out = weaver.evolve(ids)
    gen = tokenizer.decode(out["sequences"][0][ids.shape[-1]:], skip_special_tokens=True)

    return {"text": gen, "dh": out["peak_delta_h"], "interventions": out["interventions"]}


def evaluate(gen, target, wrong):
    """Check if generation contains target but not wrong answer."""
    gen = gen.lower()
    return target.lower() in gen and wrong.lower() not in gen
import torch
import pandas as pd
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantum_sentinel.paths import DATA_DIR, ASSAYS_DIR
from quantum_sentinel.core.shannon_observer import ShannonObserver
from quantum_sentinel.core.unitary_weaver import UnitaryWeaver


def load_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",
        trust_remote_code=True
    )

    return model, tokenizer


def run_universal_assay(model_id):
    signal_path = DATA_DIR / "entropy_signal.json"

    with open(signal_path, "r") as f:
        suite = json.load(f)

    model, tokenizer = load_model(model_id)

    results = []
    for case in suite:
        observer = ShannonObserver(tokenizer, case['id'])
        weaver = UnitaryWeaver(model, tokenizer, observer)

        inputs = tokenizer(case['prompt'], return_tensors="pt").to(model.device)
        out = weaver.evolve(inputs.input_ids)

        observer.save_trace(model_id.split('/')[-1])

        results.append({
            "model": model_id,
            "category": case['id'].split('-')[0],
            "peak_dh": max([p['velocity'] for p in observer.phase_log]) if observer.phase_log else 0.0,
            "success": case['target'].lower() in tokenizer.decode(out['sequences'][0]).lower()
        })
        torch.cuda.empty_cache()

    # Free memory before next model
    del model
    torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    ASSAYS_DIR.mkdir(parents=True, exist_ok=True)

    models = [
        "microsoft/Phi-3-mini-4k-instruct",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "google/gemma-3-1b-it"
    ]

    final_data = []
    for m in models:
        print(f"🔬 Running assay on {m}...")
        final_data.extend(run_universal_assay(m))

    df = pd.DataFrame(final_data)
    output_path = ASSAYS_DIR / "final_decoherence_2026.csv"
    df.to_csv(output_path, index=False)
    print(f"📊 [ASSAY COMPLETE] Results archived in: {output_path}")
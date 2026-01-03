import torch
import pandas as pd
from quantum_sentinel.engine.weaver import BacktrackingWeaver, QuantumLogitProcessor
from quantum_sentinel.sentinel.chronicle import SentinelAligner
from transformers import AutoModelForCausalLM, AutoTokenizer


def run_universal_benchmark(model_id, test_cases):
    results = []
    print(f"\n📡 [INITIALIZING BENCHMARK: {model_id}]")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,  # Optimized for 3080 Ti stability
        device_map="auto"
    )

    for test in test_cases:
        print(f"🧪 Testing: {test['id']}")
        inputs = tokenizer(test["prompt"], return_tensors="pt").to(model.device)

        # Initialize Sentinel for this specific thread
        sentinel = SentinelAligner(tokenizer, task_type=test['id'])
        processor = QuantumLogitProcessor(temperature=0.1)
        weaver = BacktrackingWeaver(model, tokenizer, sentinel, processor)

        # Execute intervention with Ground State Injection
        gen_res = weaver.generate_with_alignment(inputs.input_ids, max_tokens=15)
        text_output = tokenizer.decode(gen_res['sequences'][0], skip_special_tokens=True)

        # Extract Phase Transition Metrics
        if sentinel.phase_log:
            peak_dh = max([p['velocity'] for p in sentinel.phase_log])
            avg_h = sum([p['current_entropy'] for p in sentinel.phase_log]) / len(sentinel.phase_log)
        else:
            peak_dh, avg_h = 0.0, 0.0

        results.append({
            "model": model_id,
            "test_id": test['id'],
            "peak_delta_h": round(peak_dh, 4),
            "entanglement_avg": round(avg_h, 4),
            "backtracks": gen_res['backtracks'],
            "success": test['target'].lower() in text_output.lower()
        })

        # Memory cleanup between logical transitions
        torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    stress_tests = [
        {"id": "Recency-Bias", "prompt": "Fact 1: X=5. Fact 2: Y=X. Fact 3: X=10. Question: Y=", "target": "5"},
        {"id": "Modus-Tollens", "prompt": "If A then B. Not B. Therefore,", "target": "not A"}
    ]

    all_data = []
    for m_id in ["microsoft/Phi-3-mini-4k-instruct", "Qwen/Qwen2.5-0.5B-Instruct"]:
        all_data.extend(run_universal_benchmark(m_id, stress_tests))

    # Export to DataFrame for the PhD Paper Results
    df = pd.DataFrame(all_data)
    print("\n📊 [BENCHMARK COMPLETE] Final Results Table:")
    print(df.to_string(index=False))
    df.to_csv("results/universal_decoherence_2026.csv")
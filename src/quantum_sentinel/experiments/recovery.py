import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from quantum_sentinel.sentinel.chronicle import SentinelAligner
from quantum_sentinel.engine.weaver import BacktrackingWeaver, QuantumLogitProcessor

logging.basicConfig(level=logging.INFO, format='%(message)s')


def run_recovery_tests():
    model_id = "microsoft/Phi-3-mini-4k-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")

    recovery_tests = [
        {
            "id": "Recency-Bias-Variable",
            "prompt": "Fact 1: X=5. Fact 2: Y=X. Fact 3: X=10. Question: Based on the facts, what is the value of Y? Answer: Y=",
            "target": "5"
        },
        {
            "id": "Modus-Tollens-Negation",
            "prompt": "If it is a square, it has four sides. This shape does not have four sides. Therefore, it",
            "target": "not a square"
        },
        {
            "id": "Contradiction-Detection",
            "prompt": "Statement 1: All birds fly. Statement 2: Penguins are birds. Statement 3: Penguins do not fly. Conclusion: The statement 'All birds fly' is",
            "target": "false"
        }
    ]

    for test in recovery_tests:
        print(f"\n🚀 [RECOVERY TRIAL: {test['id']}]")
        inputs = tokenizer(test["prompt"], return_tensors="pt").to(model.device)

        # 1. Baseline
        with torch.no_grad():
            out_greedy = model.generate(**inputs, max_new_tokens=10, do_sample=False)
            text_greedy = tokenizer.decode(out_greedy[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()

        # 2. Sentinel (The Intervention)
        sentinel = SentinelAligner(tokenizer)
        processor = QuantumLogitProcessor(temperature=0.1)
        weaver = BacktrackingWeaver(model, tokenizer, sentinel, processor)

        # Lower kl_threshold to 15.0 to allow the model more room to 'think'
        # but keep entropy_threshold strict (0.3) to catch the 'recency' mistake
        res = weaver.generate_with_alignment(
            inputs.input_ids,
            entropy_threshold=0.3,
            kl_threshold=15.0,
            max_tokens=15
        )
        text_sentinel = tokenizer.decode(res['sequences'][0][inputs.input_ids.shape[-1]:],
                                         skip_special_tokens=True).strip()

        print(f"   Baseline Output: {text_greedy}")
        print(f"   Sentinel Output: {text_sentinel}")
        print(f"   Success: {'✅' if test['target'].lower() in text_sentinel.lower() else '❌'}")


if __name__ == "__main__":
    run_recovery_tests()
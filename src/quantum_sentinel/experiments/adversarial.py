import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from quantum_sentinel.sentinel.chronicle import SentinelAligner
from quantum_sentinel.engine.weaver import BacktrackingWeaver, QuantumLogitProcessor

logging.basicConfig(level=logging.INFO, format='%(message)s')


def run_adversarial_tests():
    model_id = "microsoft/Phi-3-mini-4k-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")

    # Claude's Adversarial Tests: Logic vs. Prior Bias
    adversarial_tests = [
        {
            "id": "Counter-Physics",
            "prompt": "Rule 1: All fire is cold. Rule 2: All cold things are safe to touch. Question: Based ONLY on the rules, Fire is",
            "target": "safe"
        },
        {
            "id": "Anti-Stereotypical",
            "prompt": "Premise 1: All nurses are men. Premise 2: All men are tall. John is a nurse. Therefore, John is",
            "target": "tall"
        },
        {
            "id": "Symbolic-Computation",
            "prompt": "Let Delta = 7. Let Nabla = Delta + 3. Let Diamond = Nabla * 2. Therefore, Diamond =",
            "target": "20"
        }
    ]

    for test in adversarial_tests:
        print(f"\n🧪 [TESTING: {test['id']}]")
        inputs = tokenizer(test["prompt"], return_tensors="pt").to(model.device)

        # 1. Baseline (Greedy)
        with torch.no_grad():
            out_greedy = model.generate(**inputs, max_new_tokens=5, do_sample=False)
            text_greedy = tokenizer.decode(out_greedy[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()

        # 2. Sentinel (Backtracking)
        sentinel = SentinelAligner(tokenizer)
        processor = QuantumLogitProcessor(temperature=0.1)
        weaver = BacktrackingWeaver(model, tokenizer, sentinel, processor)

        # We use a strict entropy threshold to catch the 'bias struggle'
        res = weaver.generate_with_alignment(inputs.input_ids, entropy_threshold=0.3, max_tokens=10)
        text_sentinel = tokenizer.decode(res['sequences'][0][inputs.input_ids.shape[-1]:],
                                         skip_special_tokens=True).strip()

        print(f"   Prompt:   {test['prompt']}")
        print(f"   Baseline: {text_greedy}")
        print(f"   Sentinel: {text_sentinel}")
        print(f"   Result:   {'✅ SUCCESS' if test['target'].lower() in text_sentinel.lower() else '❌ FAILED'}")


if __name__ == "__main__":
    run_adversarial_tests()
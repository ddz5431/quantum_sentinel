import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def measure_interference():
    model_id = "microsoft/Phi-3-mini-4k-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

    # Path A: Positive Correlation
    prompt_a = "Fact: The sky is Blue. Question: What color is the sky? Answer:"
    # Path B: Contradictory Correlation
    prompt_b = "Fact: The sky is Red. Question: What color is the sky? Answer:"
    # Superposition: Combined Interference
    prompt_ab = "Fact: The sky is Blue. Fact: The sky is Red. Question: What color is the sky? Answer:"

    def get_logits(text):
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
            return torch.softmax(outputs.logits[0, -1, :], dim=-1)

    prob_a = get_logits(prompt_a)
    prob_b = get_logits(prompt_b)
    prob_ab = get_logits(prompt_ab)

    # Check for "Semantic Destructive Interference"
    blue_id = tokenizer.encode(" Blue", add_special_tokens=False)[0]
    red_id = tokenizer.encode(" Red", add_special_tokens=False)[0]

    print(f"P(Blue|A): {prob_a[blue_id]:.4f}")
    print(f"P(Red|B): {prob_b[red_id]:.4f}")
    print(f"P(Blue|AB): {prob_ab[blue_id]:.4f}")
    print(f"P(Red|AB): {prob_ab[red_id]:.4f}")

    interference = prob_ab[blue_id] + prob_ab[red_id]
    print(f"\nTotal Probability Mass in Superposition: {interference:.4f}")
    if interference < (prob_a[blue_id] * 0.5 + prob_b[red_id] * 0.5):
        print("✅ EVIDENCE DETECTED: Destructive Semantic Interference.")


if __name__ == "__main__":
    measure_interference()
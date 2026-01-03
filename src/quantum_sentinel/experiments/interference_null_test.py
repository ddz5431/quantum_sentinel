import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def find_destructive_interference():
    model_id = "microsoft/Phi-3-mini-4k-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

    # Target: The concept of 'Purplish'
    # Fact A: "The object is vibrant Red." (Might allow 'Purplish' as a shade)
    # Fact B: "The object is deep Blue." (Might allow 'Purplish' as a shade)
    # Fact AB: Combined, they should force a choice or a contradiction spike,
    #          potentially suppressing the middle ground 'Purplish' below baseline.

    prompt_a = "Fact: The object is vibrant Red. Question: Is it purplish? Answer:"
    prompt_b = "Fact: The object is deep Blue. Question: Is it purplish? Answer:"
    prompt_ab = "Fact: The object is vibrant Red. Fact: The object is deep Blue. Question: Is it purplish? Answer:"

    def get_token_prob(text, target_token=" Yes"):
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits[0, -1, :], dim=-1)
            token_id = tokenizer.encode(target_token, add_special_tokens=False)[0]
            return probs[token_id].item()

    p_a = get_token_prob(prompt_a)
    p_b = get_token_prob(prompt_b)
    p_ab = get_token_prob(prompt_ab)

    print(f"P(Yes|Red): {p_a:.6f}")
    print(f"P(Yes|Blue): {p_b:.6f}")
    print(f"P(Yes|Red+Blue): {p_ab:.6f}")

    if p_ab < min(p_a, p_b):
        print("\n🌌 DESTRUCTIVE INTERFERENCE CONFIRMED.")
        print(f"Mass was lost: {p_ab} < {min(p_a, p_b)}")
    else:
        print("\nClassical averaging detected.")

if __name__ == "__main__":
    find_destructive_interference()
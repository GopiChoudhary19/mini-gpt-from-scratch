import os
import torch
from dataset import SimpleTokenizer, SAMPLE_TEXT
from model import MiniGPT

def generate_text(prompt="To be", max_tokens=100):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = SimpleTokenizer(SAMPLE_TEXT)
    
    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        n_embd=64,
        n_head=4,
        n_layer=4,
        block_size=32
    ).to(device)

    if os.path.exists('model_checkpoint.pt'):
        model.load_state_dict(torch.load('model_checkpoint.pt', map_location=device))
        print("Loaded checkpoint from model_checkpoint.pt")
    else:
        print("Warning: model_checkpoint.pt not found, sampling from an untrained model. Run train.py first.")

    model.eval()
    encoded = tokenizer.encode(prompt)
    if not encoded:
        raise ValueError(f"None of the characters in {prompt!r} are in the training vocabulary")
    context = torch.tensor([encoded], dtype=torch.long, device=device)
    generated_indices = model.generate(context, max_new_tokens=max_tokens, temperature=0.8, top_k=5)[0].tolist()
    
    output_text = tokenizer.decode(generated_indices)
    print("\n--- Generated Text ---")
    print(output_text)
    print("----------------------\n")

if __name__ == '__main__':
    generate_text("To be", max_tokens=100)

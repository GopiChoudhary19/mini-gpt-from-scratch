import time
import torch
from dataset import SimpleTokenizer, get_batch, SAMPLE_TEXT
from model import MiniGPT

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    tokenizer = SimpleTokenizer(SAMPLE_TEXT)
    data = torch.tensor(tokenizer.encode(SAMPLE_TEXT), dtype=torch.long)
    
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    batch_size = 16
    block_size = 32
    max_iters = 500
    eval_interval = 100
    learning_rate = 1e-3

    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        n_embd=64,
        n_head=4,
        n_layer=4,
        block_size=block_size
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print(f"Training MiniGPT ({sum(p.numel() for p in model.parameters())/1e3:.1f}k parameters)...")
    start_time = time.time()

    for iter_step in range(max_iters):
        xb, yb = get_batch(train_data, block_size, batch_size, device)
        
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iter_step % eval_interval == 0 or iter_step == max_iters - 1:
            print(f"Step {iter_step:4d} | Train Loss: {loss.item():.4f}")

    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.2f} seconds.")

    # Save model checkpoint
    torch.save(model.state_dict(), 'model_checkpoint.pt')
    print("Model checkpoint saved to model_checkpoint.pt")

if __name__ == '__main__':
    main()

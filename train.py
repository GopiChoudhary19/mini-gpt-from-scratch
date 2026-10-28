import time
import torch
from dataset import SimpleTokenizer, get_batch, SAMPLE_TEXT
from model import MiniGPT

@torch.no_grad()
def estimate_loss(model, splits, block_size, batch_size, device, eval_iters=20):
    """Mean loss over a few batches per split, with dropout off, so train and val are comparable."""
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = [model(*get_batch(data, block_size, batch_size, device))[1].item() for _ in range(eval_iters)]
        out[name] = sum(losses) / len(losses)
    model.train()
    return out


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
    eval_interval = 25
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
    best_val, best_step = float("inf"), -1

    for iter_step in range(max_iters):
        xb, yb = get_batch(train_data, block_size, batch_size, device)
        
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iter_step % eval_interval == 0 or iter_step == max_iters - 1:
            losses = estimate_loss(model, {"train": train_data, "val": val_data}, block_size, batch_size, device)
            print(f"Step {iter_step:4d} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f}")
            # the corpus is tiny and the model overfits quickly, so keep the weights that generalise best
            if losses["val"] < best_val:
                best_val, best_step = losses["val"], iter_step
                torch.save(model.state_dict(), 'model_checkpoint.pt')

    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.2f} seconds.")
    print(f"Saved the best checkpoint (step {best_step}, val loss {best_val:.4f}) to model_checkpoint.pt")

if __name__ == '__main__':
    main()

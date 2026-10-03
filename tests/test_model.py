import torch
import pytest
from model import GPT, GPTConfig

def test_model_forward_pass():
    config = GPTConfig(
        vocab_size=100,
        block_size=32,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0
    )
    model = GPT(config)
    idx = torch.randint(0, 100, (2, 16))
    logits, loss = model(idx)
    assert logits.shape == (2, 16, 100)
    assert loss is None

def test_model_loss_calculation():
    config = GPTConfig(
        vocab_size=100,
        block_size=32,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0
    )
    model = GPT(config)
    idx = torch.randint(0, 100, (2, 16))
    targets = torch.randint(0, 100, (2, 16))
    logits, loss = model(idx, targets=targets)
    assert loss is not None
    assert loss.item() > 0.0

def test_text_generation():
    config = GPTConfig(
        vocab_size=50,
        block_size=16,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0
    )
    model = GPT(config)
    model.eval()
    idx = torch.zeros((1, 1), dtype=torch.long)
    generated = model.generate(idx, max_new_tokens=5)
    assert generated.shape == (1, 6)

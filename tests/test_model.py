import torch

from model import MiniGPT


def small_model(vocab_size=100, block_size=32):
    torch.manual_seed(0)
    return MiniGPT(vocab_size=vocab_size, n_embd=32, n_head=2, n_layer=2, block_size=block_size, dropout=0.0)


def test_model_forward_pass():
    model = small_model()
    idx = torch.randint(0, 100, (2, 16))
    logits, loss = model(idx)
    assert logits.shape == (2, 16, 100)
    assert loss is None


def test_model_loss_calculation():
    model = small_model()
    idx = torch.randint(0, 100, (2, 16))
    targets = torch.randint(0, 100, (2, 16))
    _, loss = model(idx, targets=targets)
    # an untrained model should be close to uniform: loss ~ ln(vocab_size)
    assert abs(loss.item() - torch.log(torch.tensor(100.0)).item()) < 0.5


def test_causal_mask_blocks_future_tokens():
    model = small_model().eval()
    idx = torch.randint(0, 100, (1, 16))
    changed = idx.clone()
    changed[0, -1] = (changed[0, -1] + 1) % 100
    with torch.no_grad():
        a, _ = model(idx)
        b, _ = model(changed)
    # predictions for positions before the edited token must not change
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-6)


def test_text_generation():
    model = small_model(vocab_size=50, block_size=16).eval()
    idx = torch.zeros((1, 1), dtype=torch.long)
    generated = model.generate(idx, max_new_tokens=5)
    assert generated.shape == (1, 6)


def test_generation_crops_context_beyond_block_size():
    model = small_model(vocab_size=50, block_size=8).eval()
    idx = torch.zeros((1, 1), dtype=torch.long)
    generated = model.generate(idx, max_new_tokens=20, top_k=5)
    assert generated.shape == (1, 21)


def test_greedy_generation_with_zero_temperature_is_deterministic():
    model = small_model(vocab_size=50, block_size=16).eval()
    idx = torch.zeros((1, 1), dtype=torch.long)
    a = model.generate(idx, max_new_tokens=8, temperature=0)
    b = model.generate(idx, max_new_tokens=8, temperature=0)
    assert torch.equal(a, b)


def test_generation_rejects_empty_context():
    import pytest
    model = small_model(vocab_size=50, block_size=16).eval()
    with pytest.raises(ValueError):
        model.generate(torch.zeros((1, 0), dtype=torch.long), max_new_tokens=3)


def test_estimate_loss_reports_both_splits_and_restores_train_mode():
    from train import estimate_loss
    model = small_model(vocab_size=50, block_size=8)
    data = torch.randint(0, 50, (200,))
    losses = estimate_loss(model, {"train": data, "val": data}, block_size=8, batch_size=4, device="cpu", eval_iters=2)
    assert set(losses) == {"train", "val"}
    assert model.training

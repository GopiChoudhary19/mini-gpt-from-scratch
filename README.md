# 🧠 MiniGPT: PyTorch Generative Pre-trained Transformer from Scratch

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A clean, modular, and educational implementation of the **Decoder-Only Transformer Architecture (GPT)** written from scratch using PyTorch. 

This repository demonstrates the fundamental engineering blocks powering modern Large Language Models (LLMs), including multi-head causal self-attention, residual skip-connections, position embeddings, and autoregressive text generation.

---

## 🏗️ Architecture Overview

```
Input Tokens (B, T) 
      │
      ├───> Token Embedding (B, T, C)
      └───> Positional Embedding (T, C)
      │
      ▼
Sum Embeddings (B, T, C)
      │
      ▼  ┌──────────────────────────────────────────────┐
      ├──| LayerNorm ──> Multi-Head Causal Attention  | (x N Layers)
      │  |    │ (Residual Connection)                  |
      │  |    ▼                                        |
      │  | LayerNorm ──> Position-wise FeedForward     |
      │  └──────────────────────────────────────────────┘
      ▼
LayerNorm (Final)
      │
      ▼
Linear Head (B, T, Vocab Size) ──> Softmax ──> Next Token Prediction
```

### 🧮 Core Attention Equation

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

Where $M$ is the causal triangular mask preventing token lookahead during training.

---

## 📁 Repository Structure

```
.
├── dataset.py     # Tokenization, vocabulary mapping, and batch creation
├── model.py       # PyTorch MiniGPT architecture (Attention, FeedForward, Block, GPT)
├── train.py       # Training loop with AdamW optimizer and loss tracking
├── generate.py    # Autoregressive text generation with top-k sampling
└── requirements.txt
```

---

## 🚀 Quickstart

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/gopigkc1910-commits/mini-gpt-from-scratch.git
cd mini-gpt-from-scratch
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python train.py
```
*Outputs loss logs and saves trained weights to `model_checkpoint.pt`.*

### 3. Generate Text
```bash
python generate.py
```

---

## 📊 Benchmarks & Training Performance

| Parameter | Value |
| :--- | :--- |
| **Model Parameters** | ~200k |
| **Layers (n_layer)** | 4 |
| **Attention Heads (n_head)** | 4 |
| **Embedding Size (n_embd)** | 64 |
| **Block Size (Context)** | 32 tokens |
| **Final Loss** | 1.12 |

---

## 📜 License
Licensed under the [MIT License](LICENSE).

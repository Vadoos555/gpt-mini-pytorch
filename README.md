# GPT Mini — Transformer Language Model from Scratch

Educational implementation of a small GPT-style language model built from scratch with **Python and PyTorch**.

The main goal of this project is not to build a competitive language model, but to understand how a GPT-like model works internally by implementing its main components manually and training the model through the complete pipeline:

**Tokenizer → Transformer → Pretraining → SFT → Evaluation → Chat**

---

## 🎯 Project Goal

This project was created as a practical implementation of the ideas from:

> Sebastian Raschka — *Build a Large Language Model (From Scratch)*

The project focuses on understanding the internal mechanics of a GPT-style model through hands-on implementation rather than using high-level NLP libraries.

The model was intentionally kept small so that the entire project can be trained and tested on a **CPU**.

---

## 🧠 Architecture

The complete pipeline looks like this:

```text
Text
 │
 ▼
BPE Tokenizer
 │
 ▼
Token IDs
 │
 ▼
Token + Position Embeddings
 │
 ▼
┌─────────────────────────────┐
│      Transformer Block      │
│                             │
│   LayerNorm                 │
│      ↓                      │
│   Multi-Head Attention      │
│      ↓                      │
│   Residual Connection       │
│      ↓                      │
│   LayerNorm                 │
│      ↓                      │
│   MLP / Feed Forward        │
│      ↓                      │
│   Residual Connection       │
└─────────────────────────────┘
 │
 ▼
× 6 Transformer Blocks
 │
 ▼
Final LayerNorm
 │
 ▼
LM Head
 │
 ▼
Logits
 │
 ▼
Next Token Prediction
```

---

## 🏗️ Model Configuration

The model uses the following configuration:

| Parameter                       |   Value |
| ------------------------------- | ------: |
| Vocabulary size                 | dynamic |
| Context size                    |     256 |
| Embedding dimension (`d_model`) |     256 |
| Transformer layers              |       6 |
| Attention heads                 |       4 |
| Head dimension (`d_head`)       |      64 |
| Feed-forward dimension          |    1024 |
| Parameters                      |   ~9.9M |
| Device                          |     CPU |

The vocabulary size is determined by the tokenizer used for the experiment.

---

## 🔨 What Was Implemented

### 1. BPE Tokenizer

A small educational BPE tokenizer was implemented without using `tiktoken`.

It includes:

* text pre-tokenization
* pair frequency counting
* BPE merge operations
* vocabulary construction
* token → ID conversion
* ID → token conversion
* special tokens

Special tokens:

```text
<|user|>
<|assistant|>
<|pad|>
```

The tokenizer was intentionally implemented in a simple and understandable way for educational purposes.

---

### 2. Token Embeddings

Token IDs are converted into dense vectors using `nn.Embedding`.

```text
token ID
   ↓
embedding lookup
   ↓
vector [256]
```

---

### 3. Positional Embeddings

Since a Transformer does not inherently know the order of tokens, positional embeddings are added to token embeddings.

```text
Token Embedding
       +
Position Embedding
       ↓
Transformer input
```

---

### 4. Query, Key and Value

The self-attention mechanism was implemented manually using three learned projections:

```text
Q = XWq
K = XWk
V = XWv
```

This was implemented explicitly to understand how the attention mechanism works internally.

---

### 5. Multi-Head Self-Attention

The model uses:

```text
4 attention heads
64 dimensions per head
```

The main attention calculation is:

```text
Attention(Q, K, V)  = softmax(QKᵀ / √d_head) V
```

The implementation includes:

* Q/K/V projections
* splitting into multiple heads
* scaled dot-product attention
* causal masking
* softmax
* combining attention heads
* output projection

---

### 6. Causal Mask

A causal mask prevents the model from looking at future tokens.

For example:

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

When predicting the next token, the model can only use information from the current and previous positions.

---

### 7. MLP / Feed-Forward Network

Each Transformer block contains a feed-forward network:

```text
256
 ↓
1024
 ↓
ReLU
 ↓
256
```

The MLP operates independently on each token position.

---

### 8. LayerNorm and Residual Connections

The Transformer block uses LayerNorm and residual connections:

```python
x = x + attention(norm1(x))
x = x + mlp(norm2(x))
```

This corresponds to a pre-LayerNorm Transformer block.

---

### 9. GPT Model

The individual components were combined into the complete GPT model:

```text
Embedding
    ↓
Transformer Block × 6
    ↓
LayerNorm
    ↓
LM Head
    ↓
Logits
```

The final model contains approximately:

```text
9,934,608 parameters
```

---

# 🚀 Text Generation

Several generation methods were implemented and tested.

### Greedy Generation

The model selects the token with the highest probability:

```text
next_token = argmax(logits)
```

### Temperature

Temperature controls the sharpness of the probability distribution.

```text
lower temperature
    ↓
more deterministic

higher temperature
    ↓
more random
```

### Top-K Sampling

Only the `K` most probable tokens are considered.

### Top-P Sampling

Tokens are selected from the smallest probability set whose cumulative probability reaches `P`.

---

# 📚 Pretraining

The model was trained using the standard next-token prediction objective.

For example:

```text
Input:
I love Python

Target:
love Python ...
```

More precisely, the training data is shifted by one position:

```text
x = [10, 20, 30]

y = [20, 30, 40]
```

The model receives `x` and learns to predict `y`.

---

## Training Pipeline

```text
Text
 ↓
Tokenizer
 ↓
Token IDs
 ↓
GPT Dataset
 ↓
DataLoader
 ↓
GPT
 ↓
Logits
 ↓
CrossEntropyLoss
 ↓
Backpropagation
 ↓
AdamW
```

The training pipeline also includes:

* AdamW optimizer
* learning-rate scheduling
* validation
* checkpoint saving
* perplexity calculation

---

# 🎓 Supervised Fine-Tuning

After pretraining, the project implements a simple instruction fine-tuning pipeline.

Training examples have the following structure:

```text
<|user|>
What is Python?
<|assistant|>
Python is a programming language...
```

During SFT, the model is trained to predict the assistant response.

The user prompt is masked with:

```text
-100
```

so that it does not contribute to the loss.

Conceptually:

```text
<|user|> What is Python?
        ↓
       MASK
        ↓
<|assistant|> Python is...
              ↓
          calculate loss
```

This demonstrates an important idea behind instruction fine-tuning.

---

# 📊 SFT Results

The SFT experiment used a small dataset of **50 examples**:

```text
40 examples → training
10 examples → validation
```

The best validation result was:

```text
Best validation loss: 3.5732
Best checkpoint: epoch 6
```

The training loss continued decreasing after this point while validation loss started increasing, demonstrating overfitting on the small dataset.

This is expected for such a small educational dataset and model.

---

# 💬 Chat Interface

A simple terminal chat interface was implemented.

Run:

```bash
python chat.py
```

Example:

```text
GPT Mini Chat
Type "exit" to stop

User: What is Python?
Assistant: ...

User: What is a function?
Assistant: ...

User: exit
```

The chat interface loads the SFT checkpoint and uses the same tokenizer and generation pipeline as the evaluation code.

---

# 🧪 Testing

The project contains a final integration test covering the main components.

Run:

```bash
python -m tests.test_final
```

The final test verifies:

```text
✓ tokenizer
✓ model forward pass
✓ text generation
✓ checkpoint loading
✓ evaluation
```

Example result:

```text
GPT Mini Final Test
==============================
PASS №1: tokenizer
PASS №2: model forward
PASS №3: generation
PASS №4: checkpoint
PASS №5: evaluation

ALL FINAL TESTS PASSED
```

---

# 📁 Project Structure

```text
gpt_mini_pytorch/
│
├── config/
│   ├── __init__.py
│   └── model_config.py
|
├── data/
│   └── dataset.py
│
├── tokenizer/
│   ├── __init__.py
│   └── bpe.py
│
├── model/
│   ├── __init__.py
│   ├── embedding.py
│   ├── positional_embedding.py
│   ├── embeddings.py
│   ├── multi_head_attention.py
│   ├── mlp.py
│   ├── transformer_block.py
│   └── gpt.py
│
├── generation/
│   ├── greedy.py
│   ├── temperature.py
│   ├── top_k.py
│   └── top_p.py
│
├── pretraining/
│   ├── train.py
│   ├── validation.py
│   ├── checkpoint.py
│   └── ...
│
├── sft/
│   ├── dataset.py
│   ├── formatting.py
│   ├── sft_dataset.py
│   ├── collate.py
│   ├── train.py
│   ├── validation.py
│   ├── checkpoint.py
│   ├── evaluation.py
│   └── tokenizer_setup.py
│
├── tests/
│   └── test_final.py
│
├── checkpoints/
│
├── chat.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Vadoos555/gpt-mini-pytorch
cd gpt_mini_pytorch
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

### Run final tests

```bash
python -m tests.test_final
```

### Run the chat interface

```bash
python chat.py
```

### Run SFT training

```bash
python -m sft.test_train
```

> The SFT training script was used as an educational experiment. Experimental `test_*.py` files are not part of the final GitHub codebase.

---

# 🧩 Learning Path

The project was developed incrementally.

```text
1. Project setup
       ↓
2. Configuration
       ↓
3. BPE tokenizer
       ↓
4. Token embeddings
       ↓
5. Positional embeddings
       ↓
6. Q / K / V
       ↓
7. Scaled dot-product attention
       ↓
8. Causal mask
       ↓
9. Multi-head attention
       ↓
10. MLP
       ↓
11. LayerNorm
       ↓
12. Residual connections
       ↓
13. Transformer Block
       ↓
14. GPT model
       ↓
15. Generation
       ↓
16. Pretraining
       ↓
17. Supervised Fine-Tuning
       ↓
18. Evaluation
       ↓
19. Chat interface
```

---

# 🔍 What I Learned

This project provided hands-on practice with the complete lifecycle of a Transformer language model.

In particular:

* how tokenization works
* how BPE merges are learned
* how token and positional embeddings work
* why Q, K and V are needed
* how self-attention is calculated
* why attention needs scaling
* how causal masking works
* how multiple attention heads work
* how MLP layers transform token representations
* why residual connections are used
* how LayerNorm fits into a Transformer block
* how a GPT model produces logits
* how next-token prediction works
* how CrossEntropyLoss is used with language models
* how backpropagation updates model parameters
* how AdamW is used for training
* how validation reveals overfitting
* how checkpoints are saved and restored
* how supervised fine-tuning differs from pretraining
* how response masking works during SFT
* how text generation works
* how all these components fit together into a complete language-model pipeline

---

# ⚠️ Limitations

This project is intentionally small and educational.

It is **not** intended to compete with modern large language models.

Important limitations include:

* very small model size
* tiny training dataset
* CPU-only training
* simple educational BPE tokenizer
* limited context size
* no large-scale pretraining corpus
* no distributed training
* no GPU optimization
* very limited instruction dataset

Because of these limitations, the generated text quality is low.

That is expected.

The main objective of the project is to understand **how the system works**, not to achieve high-quality language generation.

---

# 🛠️ Technologies

* Python
* PyTorch
* NumPy
* Git / GitHub

---

# 📌 Project Status

The main educational pipeline is complete:

```text
✓ BPE Tokenizer
✓ Token Embeddings
✓ Positional Embeddings
✓ Q / K / V
✓ Self-Attention
✓ Causal Mask
✓ Multi-Head Attention
✓ MLP
✓ LayerNorm
✓ Residual Connections
✓ Transformer Block
✓ GPT Model
✓ Text Generation
✓ Pretraining
✓ Validation
✓ Checkpoints
✓ Supervised Fine-Tuning
✓ SFT Validation
✓ SFT Checkpoints
✓ Evaluation
✓ Chat Interface
✓ Final Integration Test
✓ README
```

---

# 🎯 Final Goal

The project demonstrates the complete path from raw text to a working GPT-style language model:

```text
Raw Text
   ↓
Tokenizer
   ↓
Token IDs
   ↓
Embeddings
   ↓
Transformer
   ↓
Logits
   ↓
Next Token
   ↓
Generated Text
```

The most important result of this project is not the final model itself, but the understanding of the mechanisms behind it.

> **Build it. Break it. Debug it. Understand it.**
>
> That's the purpose of this project.

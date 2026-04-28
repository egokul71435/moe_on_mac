# MoE on Mac — Build Plan

## Phase 0 — Environment Setup
*~1–2 days*

| Component | Details |
|---|---|
| pyenv | Python 3.11 native arm64 |
| venv | Isolated per-project environment |
| MLX 0.31+ | Core training framework |
| Support packages | datasets, tokenizers, numpy |

---

## Phase 1 — Tokenizer
*~1–3 days*

1. **Sample corpus** — pull 10GB from FineWeb-Edu
2. **Train BPE** — HuggingFace `tokenizers`, 32K vocab
3. **Save & validate** — output `tokenizer.json`, test coverage

---

## Phase 2 — Data Pipeline
*~1 day*

1. **Stream FineWeb-Edu** — sample-10BT, ~28.5GB, Parquet → text streaming
2. **Tokenize & shard** — 100M tokens/shard, uint16 `.bin` files to SSD
3. **Memory-mapped loader** — `np.memmap` → MLX arrays, no RAM bottleneck

---

## Phase 3 — Model Architecture (MLX)
*~3–7 days*

**Attention:**
- RoPE positional encoding
- QK-Norm for attention stability
- Causal self-attention via `mx.fast.scaled_dot_product`
- Value embeddings on alternating layers

**FFN:**
- RMSNorm via `mx.fast.rms_norm`
- ReLU² activation
- MoE FFN layer — 8 experts, top-2, load balancing

---

## Phase 4 — Training
*~6h – 3 weeks*

| Component | Details |
|---|---|
| Optimizer | Muon for weight matrices, AdamW for embeddings/head |
| Hyperparameters | LR 3e-4, cosine decay, batch 16–32, seq 1024 |
| Compilation | `mx.compile` to fuse forward + backward, checkpoint every 2h |

---

## Phase 5 — Evaluation
*~1–2 days*

| Component | Details |
|---|---|
| Val loss | FineWeb val set, target ≤3.28 cross-entropy |
| Benchmarks | `lm-evaluation-harness` — HellaSwag, LAMBADA, WikiText-103 perplexity |
| Router monitoring | Expert utilization logs, watch for collapse |

---

## Project Structure

```
moe-llm/
├── tokenizer/
│   ├── train_tokenizer.py
│   └── tokenizer.json
├── data/
│   ├── download_fineweb.py
│   ├── tokenize_shards.py
│   └── edu_fineweb10B/         # ~28.5GB of .bin shards on SSD
├── model/
│   ├── config.py               # ModelConfig dataclass
│   ├── attention.py            # CausalSelfAttention with RoPE + QK-Norm
│   ├── moe.py                  # MoE FFN layer + router
│   ├── transformer.py          # Full model class
│   └── optimizer.py            # Muon + AdamW hybrid
├── train.py                    # Main training loop
├── eval.py                     # Validation loss + benchmark runner
└── checkpoints/                # Saved model weights (safetensors)
```

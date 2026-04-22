# moe_on_mac

## base:

Character-level sparse MoE language model trained on Shakespeare. Implements noisy top-k routing, multi-head self-attention, and a full transformer stack with MoE replacing the FFN layer.

### setup:

```bash
conda create -n moe_on_mac python=3.12
conda activate moe_on_mac
pip install torch
```

Open `base.ipynb` and run all cells in order. The Shakespeare dataset is downloaded automatically via `wget`. Training runs on CPU by default — MPS is available on Apple Silicon but slower for this model size due to sparse indexing fallbacks.

To save the model after training:
```python
torch.save(model.state_dict(), 'sparse_moe.pt')
```

To load the saved weights (`sparse_moe.pt` included):
```python
model = SparseMoeLanguageModel()
model.load_state_dict(torch.load('sparse_moe.pt', map_location=device))
model.to(device)
model.eval()
```

To run inference from a text prompt:
```python
prompt = "To be or not"
context = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
print(decode(model.generate(context, max_new_tokens=200)[0].tolist()))
```

### limitations:

- **Character-level tokenization**: vocab is ~65 chars vs. 50k+ subword tokens in modern models — output is Shakespeare-shaped noise, not coherent text
- **Model size**: 8.996545 M parameters (`n_layer=8`, `n_embed=128`) — GPT-2 small is 117M
- **Training budget**: base model trained for 50,000 steps on CPU (~9M parameters), taking 569 minutes and 48.7 seconds total (i.e ~9.5 hours)
- **No model persistence**: training state is lost on kernel restart unless manually saved with `torch.save` — trained weights saved to `sparse_moe.pt`
- **MoE scale**: 8 experts with `top_k=2` is too small to see real MoE efficiency gains
- **No load balancing loss**: auxiliary loss to penalize uneven expert utilization is absent, so expert collapse is possible despite the injected noise
- **Activation function**: ReLU used instead of the standard GELU for transformer FFN layers
- **Single dataset**: trained only on Shakespeare with no pretraining corpus

### extensions:

- Switch to BPE/tiktoken subword tokenization to dramatically improve output quality
- Replace ReLU with GELU throughout
- Add auxiliary load balancing loss to prevent expert collapse
- Scale up: more experts (e.g. 16-64), larger `n_embed`, more layers
- Add `torch.save` / `torch.load` for checkpointing mid-training
- Train on a larger and more diverse corpus
- Add weight tying between `token_embedding_table` and `lm_head`


## modified:

### goal:

Design and train a sparse MoE language model that achieves at least GPT-2 level performance (117M parameter baseline, ~3.0 bits-per-character on text benchmarks) while running entirely on Mac hardware — no CUDA required.

**Target architecture:**
- Subword tokenization (BPE via tiktoken) to match GPT-2's vocabulary
- GELU activations throughout
- Sufficient scale: parameter count in the 100M+ range via MoE sparsity — many experts, only top-k active per token, keeping inference cost manageable on-device
- Auxiliary load balancing loss to ensure expert utilization doesn't collapse
- Weight tying between token embeddings and `lm_head`

**Mac compatibility:**
- Primary target is MPS (Metal Performance Shaders via PyTorch) for native Apple Silicon GPU acceleration
- MLX as a fallback/alternative if MPS bottlenecks prove insurmountable for sparse ops
- All sparse indexing and routing ops must run without CPU fallback — MPS-compatible implementations required

**Training:**
- Start from scratch on a diverse corpus (not just Shakespeare)
- Checkpointing throughout so training can resume across sessions on a single machine


## notes:

- inspired by [YouTube - Vizuara](https://www.youtube.com/watch?v=W7ktPe1HfZs "Code Mixture of Experts (MoE) from Scratch in Python")
- will be expanded on and trained exclusively on mac


## changes to make:

- Relu is the used activation function, Gelu is standard, move to that
- tokenization, source uses character-tokenization
-

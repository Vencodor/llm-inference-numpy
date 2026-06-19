# llm-inference-numpy
A numpy-only inference engine for GPT-2, extended toward modern LLM architecture (RMSNorm, RoPE, GQA, SwiGLU).

## Why
Built from scratch to understand transformer inference at the structural level — no PyTorch, no autograd. Just numpy and the math.

## Features
- [x] BPE tokenizer
- [x] Embeddings + positional encoding
- [x] LayerNorm, MLP, multi-head attention
- [ ] KV-cache
- [ ] RMSNorm / RoPE / GQA / SwiGLU

## Benchmarks
| Stage | tokens/sec | notes |
|-------|-----------|-------|
| baseline | TBD | |

## Notes
Per-stage engineering notes in `/notes`.

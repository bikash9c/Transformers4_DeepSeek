# DeepSeek-V3 Architecture (Final)

## Overview

DeepSeek-V3 is a transformer-based language model that integrates Multi-Head Latent Attention (MLHA) and a Mixture-of-Experts (MoE) feed-forward network. It is designed for efficient language modeling, balancing parameter efficiency and robust expert routing through load balancing and attention compression.

---

## Model Structure

### Configuration

```
DeepSeekConfig(
    vocab_size=49152,
    hidden_size=576,
    num_hidden_layers=30,
    num_attention_heads=9,
    intermediate_size=1536,
    num_experts=2,
    num_shared_experts=1,
    top_k_experts=1,
    moe_intermediate_size=1024,
    compression_ratio=8,
    max_position_embeddings=2048,
    rope_theta=10000.0,
    tie_word_embeddings=True,
)
```

| Key Hyperparameter         | Value                      | Purpose/Notes                           |
|---------------------------|----------------------------|-----------------------------------------|
| vocab_size                | 49152                      | Power of 2 for kernel optimization      |
| hidden_size               | 576                        | Dimension per token embedding           |
| num_hidden_layers         | 30                         | Depth (same as SmolLM)                  |
| num_attention_heads       | 9                          | # GQA groups                            |
| intermediate_size         | 1536                       | Standard FFN width                      |
| num_experts               | 2                          | Routed MoE experts (small setting)      |
| num_shared_experts        | 1                          | Always-on shared expert                 |
| top_k_experts             | 1                          | Route each token to top-1 expert        |
| moe_intermediate_size     | 1024                       | Smaller FFN only inside MoE             |
| compression_ratio         | 8                          | MLHA latent compression (KV/Q to 1/8)   |
| max_position_embeddings   | 2048                       | Sequence context window                 |
| rope_theta                | 10000.0                    | RoPE positional encoding scale          |
| tie_word_embeddings       | True                       | Share token and output embeddings       |

Parameter count: ~208M (all trainable).

---

## Attention: Multi-Head Latent Attention (MLHA)

- Compress Q and KV projections to latent (1/8th) dimensions for efficient compute and memory usage.
- Decompression produces content and RoPE positional components, with RoPE only applied to dedicated parts (not all projections).
- Attention output is full hidden size, efficiently implemented via FlashAttention.

---

## Feed-Forward: Mixture-of-Experts (MoE)

- Transformer FFN is replaced with:
  - 2 routed experts (1024 hidden each)
  - 1 shared expert (1536 hidden, always active)
- Router learns per-token scores using sigmoid, selects top-1 expert, normalizes weights.
- Load balancing bias is adapted every 100 steps (no auxiliary loss needed), preventing dead experts.

---

## Positional Encoding and Sequence Length

- Uses rotary positional embeddings (RoPE) with rope_theta=10000.0 applied to dedicated Q/K subspace.
- Supports up to 2048 token contexts per sequence.

---

## Training Setup

- Optimizer: AdamW (lr=5e-4, betas=(0.9, 0.95), weight_decay=0.1), cosine schedule, 1000-step warmup.
- Batch: 2 sequences × 1024 tokens (2048 tokens/batch).
- Mixed precision (bfloat16/TF32), torch.compile, FlashAttention enabled.
- Loss: Cross-entropy, next-token prediction.
- Total steps: 10,000.

---

## Final Loss and Training Snapshot

| Step      | Loss    | LR         | Time/step | Tokens/sec |
|-----------|---------|------------|-----------|------------|
| 9994      | 2.1431  | 0.000050   | 710ms     | 1440.93    |
| 9995      | 1.9878  | 0.000050   | 719ms     | 1423.21    |
| 9996      | 1.6737  | 0.000050   | 710ms     | 1440.26    |
| 9997      | 1.9976  | 0.000050   | 710ms     | 1441.17    |
| 9998      | 1.7667  | 0.000050   | 714ms     | 1432.30    |
| 9999      | 1.6420  | 0.000050   | 714ms     | 1433.80    |
| 10000     | 1.7221  | 0.000050   | 4066ms    | 251.79     |

---

## Sample Generations

| Prompt               | Output (Sample)                                                                            |
|----------------------|--------------------------------------------------------------------------------------------|
| Once upon a time     | Once upon a time best and fair and thing by wion...                                        |
| The meaning of life is| The meaning of life is the of. L I, night...                                               |
| In a world where     | In a world where should call their with;And his queen the fl...                            |
| To be or not to be   | To be or not to be' much. MR very on head go...                                            |
| The future of AI     | The future of AI give thousand:My is ictish yet seen, in. First: the is hot but...         |

---



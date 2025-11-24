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

### Final Loss and Training Snapshot  
(Complete logs are in `logs.txt`)

| Step | Log |
|------|-----|
| 500 | ```[TRAIN] Step   500/10000 | Loss: 6.0394 | LR: 0.000250 | dt: 761.54ms | tok/sec: 1344.65``` |
| 1000 | ```[TRAIN] Step  1000/10000 | Loss: 6.1165 | LR: 0.000500 | dt: 3959.32ms | tok/sec:  258.63``` |
| 1500 | ```[TRAIN] Step  1500/10000 | Loss: 5.2571 | LR: 0.000497 | dt: 716.06ms | tok/sec: 1430.05``` |
| 2000 | ```[TRAIN] Step  2000/10000 | Loss: 5.1143 | LR: 0.000486 | dt: 4142.85ms | tok/sec:  247.17``` |
| 2500 | ```[TRAIN] Step  2500/10000 | Loss: 4.7182 | LR: 0.000470 | dt: 708.28ms | tok/sec: 1445.76``` |
| 3000 | ```[TRAIN] Step  3000/10000 | Loss: 4.6616 | LR: 0.000447 | dt: 4249.97ms | tok/sec:  240.94``` |
| 3500 | ```[TRAIN] Step  3500/10000 | Loss: 3.8936 | LR: 0.000420 | dt: 730.30ms | tok/sec: 1402.16``` |
| 4000 | ```[TRAIN] Step  4000/10000 | Loss: 3.9717 | LR: 0.000388 | dt: 4103.16ms | tok/sec:  249.56``` |
| 4500 | ```[TRAIN] Step  4500/10000 | Loss: 3.2477 | LR: 0.000352 | dt: 725.19ms | tok/sec: 1412.05``` |
| 5000 | ```[TRAIN] Step  5000/10000 | Loss: 3.3612 | LR: 0.000314 | dt: 3967.23ms | tok/sec:  258.11``` |
| 5500 | ```[TRAIN] Step  5500/10000 | Loss: 2.5258 | LR: 0.000275 | dt: 709.80ms | tok/sec: 1442.67``` |
| 6000 | ```[TRAIN] Step  6000/10000 | Loss: 2.6849 | LR: 0.000236 | dt: 3964.90ms | tok/sec:  258.27``` |
| 6500 | ```[TRAIN] Step  6500/10000 | Loss: 1.9537 | LR: 0.000198 | dt: 748.17ms | tok/sec: 1368.67``` |
| 7000 | ```[TRAIN] Step  7000/10000 | Loss: 2.7264 | LR: 0.000163 | dt: 4112.84ms | tok/sec:  248.98``` |
| 7500 | ```[TRAIN] Step  7500/10000 | Loss: 1.6527 | LR: 0.000130 | dt: 734.00ms | tok/sec: 1395.10``` |
| 8000 | ```[TRAIN] Step  8000/10000 | Loss: 2.2757 | LR: 0.000103 | dt: 3956.98ms | tok/sec:  258.78``` |
| 8500 | ```[TRAIN] Step  8500/10000 | Loss: 1.5266 | LR: 0.000080 | dt: 709.93ms | tok/sec: 1442.40``` |
| 9000 | ```[TRAIN] Step  9000/10000 | Loss: 1.8181 | LR: 0.000064 | dt: 3932.84ms | tok/sec:  260.37``` |
| 9500 | ```[TRAIN] Step  9500/10000 | Loss: 1.2203 | LR: 0.000053 | dt: 753.87ms | tok/sec: 1358.32``` |
| 10000 | ```[TRAIN] Step 10000/10000 | Loss: 1.7221 | LR: 0.000050 | dt: 4066.82ms | tok/sec:  251.79``` |

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



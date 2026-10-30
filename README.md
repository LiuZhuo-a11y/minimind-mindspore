# MiniMind-MindSpore

**A working re-implementation of the [MiniMind](https://github.com/jingyaogong/minimind) LLM training stack in MindSpore, built and debugged on a domestic software stack (openEuler + MindSpore) with a single 11 GB GTX 1080 Ti.**

The upstream project is a compact, from-scratch LLM written in PyTorch. This repository asks a narrower engineering question:

> *What does it actually take to move a small LLM's model definition, five training objectives and inference path from PyTorch to a structurally different framework, on hardware that predates the kernels modern frameworks assume?*

The answer here is not a patch. It is a re-implementation of the compute graph — `nn.Cell` instead of `nn.Module`, `construct` instead of `forward`, explicit-shape ops instead of in-place view/transpose, `value_and_grad` instead of `loss.backward()` — validated against the **unmodified upstream tokenizer and datasets**, with each porting decision written down in [`docs/migration-notes.md`](docs/migration-notes.md).

---

## Status at a glance

| Component | State | Notes |
|---|---|---|
| Causal LM (RMSNorm, RoPE, GQA, SwiGLU) | Implemented | ~80.6 M parameters at the default config |
| MoE feed-forward block | Implemented (dense loop) | Top-1 routing; arithmetically faithful, not compute-sparse |
| Pre-training loop | Implemented | Next-token prediction with `-100` masking |
| Supervised fine-tuning (SFT) | Implemented | Assistant-turn-only labels |
| LoRA fine-tuning | Implemented | q/v/gate/up projections, by-name freezing |
| DPO | Implemented | Frozen reference, `stop_gradient` reference log-probs |
| Gradient accumulation | Implemented | `--grad_accum` on every training script; decouples effective batch from the 11 GB limit |
| PPO / GRPO | **Simplified** | Reward-weighted SFT; **no rollout, no clipping, no advantage** |
| Text generation / chat / tool-call demo | Implemented | Greedy decoding only |
| Numerical parity with PyTorch | **Not done** | See *Limitations* — this is the honest gap |

This is a port for **study and environment validation**, not a production training framework, and it does not claim one-to-one parity with upstream MiniMind. A component-by-component comparison — including what is *not* the same — is in [`MIGRATION_STATUS.md`](MIGRATION_STATUS.md).

**Verified on:** openEuler 24.03 LTS SP4 · NVIDIA GTX 1080 Ti (11 GB, driver 580.173.02) · CUDA 11.6 · cuDNN 8.5 · MindSpore 2.9.0 (GPU build) · Python 3.8, `PYNATIVE_MODE`, `float32`. Every number quoted below comes from that machine; the full record is [`docs/experiment-log.md`](docs/experiment-log.md).

📄 **If you read one thing, read the report:** [`docs/tech-report.md`](docs/tech-report.md) — what was ported, what it measured, and what silently broke.

## Model

| Field | Default | | Field | Default |
|---|---|---|---|---|
| `vocab_size` | 6400 (upstream tokenizer) | | `hidden_size` | 768 |
| `num_hidden_layers` | 8 | | `intermediate_size` | 3072 (4×) |
| `num_attention_heads` | 8 | | `num_key_value_heads` | 4 (GQA) |
| `head_dim` | 96 | | `max_seq_len` | 340 |
| `rms_norm_eps` | 1e-6 | | `rope_theta` | 1e6 |
| `tie_word_embeddings` | no (separate `lm_head`) | | **Parameters** | **80,622,336** |

Every block is pre-norm: `x + attn(rmsnorm(x))`, then `x + ffn(rmsnorm(x))`. The LM head is a separate `Dense`, so embedding and output projection are not tied.

The `MiniMindMSConfig` dataclass itself carries smaller fallback values (256 / 4 layers / 4 heads / 2 KV heads / 128), used only when no checkpoint sidecar is available; the entry points above pass the numbers in the table through `ms_common.make_config`. The parameter count is derived from the shapes in the module definitions, not measured from a checkpoint.

## The porting work

The interesting part of this repository is the delta between two frameworks, not the model itself. Seven problems took most of the effort:

1. **No autograd API.** PyTorch's `loss.backward(); opt.step()` becomes `ms.value_and_grad(loss_net, None, opt.parameters)` returning `grads`, then `opt(grads)`. The loss had to be wrapped in an `nn.Cell` that *owns the model*, because MindSpore differentiates a callable, not a scalar.
2. **Static-graph shape discipline.** No `.view(B, T, H, D).transpose(1, 2)`. Every reshape needs an explicit target tuple (`ops.reshape(x, (b, t, h, d))`) and every transpose an explicit permutation tuple (`ops.transpose(x, (0, 2, 1, 3))`). Batch and sequence length are read from `x.shape` at call time rather than inferred downstream.
3. **GQA without `repeat_interleave`.** The few KV heads are expanded with `expand_dims` + `tile` + `reshape` to the query-head count — the same broadcast PyTorch performs, but spelled out explicitly.
4. **A causal mask that is data, not an op.** Instead of a per-step `torch.triu` call, the mask is built once in NumPy as a `max_seq_len × max_seq_len` additive `-10000` matrix and sliced per call: one fewer graph op in the hot loop, at the cost of a fixed maximum length.
5. **Loss masking with no `ignore_index`.** MindSpore's sparse cross-entropy will index `-100` and blow up, so SFT labels are sanitised twice — `ops.maximum(labels, 0)` for the gather, plus an explicit `valid` mask in the numerator *and* the denominator, with `max(sum(valid), 1.0)` guarding the all-masked batch.
6. **Checkpoints that do not carry their shape.** `ms.save_checkpoint` writes weights only, so every save also writes a `<ckpt>.json` sidecar of the config dataclass; inference rebuilds the network from that JSON before loading weights. Without this, a config change silently produces a shape mismatch at inference time.
7. **Gradient accumulation without `backward()`.** The 11 GB card forces the micro-batch down, so the effective batch has to be rebuilt by accumulating gradients across several micro-batches (`--grad_accum`). In PyTorch that is `loss / N` plus repeated `backward()`; here `value_and_grad` returns a *tuple* of gradients per micro-batch, so `GradientAccumulator` sums those tuples and divides once when the window closes. Dividing once — rather than scaling each micro-loss — keeps the update identical to a single pass over `batch_size × accum_steps` samples, so changing `--grad_accum` does not silently change the effective learning rate. `--steps` counts optimizer updates, not micro-batches.

A full PyTorch→MindSpore mapping table, plus the traps that did not make the list above, is in [`docs/migration-notes.md`](docs/migration-notes.md).

## Repository layout

```text
.
├── ms_minimind.py       # Model: RMSNorm, RoPE, GQA attention, SwiGLU FFN, MoE, LoRA, losses
├── ms_common.py         # Tokenizer/config plumbing, JSONL dataset readers, checkpoint sidecars
├── train_smoke.py       # Pre-training entry point (also the environment smoke test)
├── train_sft.py         # Supervised fine-tuning
├── train_lora.py        # LoRA fine-tuning
├── train_dpo.py         # Direct Preference Optimization
├── train_ppo.py         # Simplified PPO (reward-weighted updates)
├── train_grpo.py        # Simplified GRPO (reward-weighted updates)
├── generate.py          # Single-prompt greedy generation
├── eval_chat.py         # Interactive chat loop (stdin)
├── serve.py             # Minimal web demo: HTML page + /api/chat, stdlib only
├── eval_toolcall.py     # Tool-call format demo with three mock tools
├── run_check.py         # Minimal "can MindSpore see the GPU" check
├── docs/                # Technical report, migration notes, experiment log
│   ├── tech-report.md   #   start here: the written-up field report
│   ├── migration-notes.md
│   ├── experiment-log.md
│   └── img/             #   terminal evidence from the original runs
└── requirements.txt
```

## What this repository deliberately does not contain

Model weights, checkpoints, the tokenizer and any dataset are **not** distributed here. They come from a separate checkout of the upstream MiniMind project, which is why every script takes `--source_dir`. This keeps the repository small and keeps third-party assets under their own licences.

## Setup

Prerequisites:

- Python 3.8+
- A MindSpore build matching your OS, Python, CUDA and driver (MindSpore ships per-CUDA wheels; a mismatch fails at `import` or at the first kernel launch, not at install time)
- An NVIDIA GPU for `--device GPU`; CPU works only for connectivity tests
- A separate upstream MiniMind checkout providing `model/` (tokenizer) and `dataset/`

```bash
git clone https://github.com/<your-user>/minimind-mindspore.git
cd minimind-mindspore
pip install -r requirements.txt
```

On the target openEuler machine the CUDA runtime has to be visible to Python before MindSpore will initialise its GPU backend — this was the single most time-consuming environment problem:

```bash
export PATH=/usr/local/cuda-11.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64:/usr/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-11.6
```

## Quick start

Point `--source_dir` at your upstream MiniMind checkout.

```bash
# 1. Does MindSpore reach the device at all?
python run_check.py
# -> MindSpore GPU check OK

# 2. Does the whole training path work? (20 steps, tiny model)
python train_smoke.py --source_dir /path/to/minimind --device GPU \
  --steps 20 --batch_size 2 \
  --hidden_size 256 --num_layers 4 --num_heads 4 --num_kv_heads 2 --max_seq_len 128
# -> MiniMind MindSpore smoke training OK
```

If step 1 prints a result and step 2 shows a decreasing loss, the driver, CUDA, MindSpore build, data pipeline and optimizer are all wired correctly.

### Pre-train → SFT → generate

```bash
python train_smoke.py --source_dir /path/to/minimind --device GPU \
  --steps 2000 --batch_size 2 --grad_accum 8 --max_seq_len 128 \
  --hidden_size 256 --num_layers 4 --num_heads 4 --num_kv_heads 2 \
  --save_name pretrain_ms.ckpt

python train_sft.py --source_dir /path/to/minimind --device GPU \
  --init_ckpt ./checkpoints/pretrain_ms.ckpt --batch_size 2 --grad_accum 4 \
  --save_name sft_ms.ckpt

python generate.py --source_dir /path/to/minimind --device GPU \
  --ckpt ./checkpoints/sft_ms.ckpt --prompt "Introduce yourself briefly."
```

### Preference tuning and adapters

```bash
python train_lora.py --source_dir /path/to/minimind --device GPU \
  --init_ckpt ./checkpoints/sft_ms.ckpt --lora_rank 8 --lora_alpha 16

python train_dpo.py --source_dir /path/to/minimind --device GPU \
  --init_ckpt ./checkpoints/sft_ms.ckpt --beta 0.1
```


## What I observed

Measurements from limited runs on one 1080 Ti. The trained artefacts stayed on the lab machine and are not in this repository, but the numbers and the terminal evidence are, in [`docs/experiment-log.md`](docs/experiment-log.md).

- **The 20-step overfit check is the run that proves the port works.** Memorising 128 samples took the loss from **8.8870 to 7.7097** (−1.18 nats), and the checkpoint answers `你好` with a fluent, correctly formatted Chinese sentence. That is a *correctness* result, not a quality one: tokenizer, prompt template, label mask, attention mask, RoPE table, checkpoint loader and greedy decoder all agree with each other end to end — which is exactly what framework ports get wrong.
- **The first loss value is itself a test.** Step 1 sits at 8.8870 while random guessing over a 6400-token vocabulary costs `ln(6400) = 8.7637`. A first loss of 5.0 (or 0.1) would have meant a broken loss, a mis-mapped label, or logits in the wrong order. One step is enough to know.
- **A 2,000-step run costs five minutes; the useful part is knowing what it does not buy.** Pre-training 2,000 steps at batch size 1 took **300.6 s** (0.150 s/step at `max_seq_len` 340); SFT 2,000 steps took **1,032.4 s** (0.516 s/step at 768). That is 3.43× the time per step for 2.26× the sequence length — the shape of an attention term that grows with `T²` plus a 768 × 6400 logits tensor. And 2,000 steps at batch size 1 means 2,000 samples, a small fraction of the 1.2 GB mini pre-training set.
- **Empty generations were the dominant failure mode**, and they were not a framework bug. After a short SFT run the model learns to emit the end-of-sequence token first — it minimises the assistant-only loss by saying nothing. `generate.py` therefore prints an explicit `[empty output]` diagnostic with the raw token ids instead of a blank line, and `--min_new_tokens` suppresses EOS for the first few steps so the failure is observable rather than silent. Diagnosing it meant checking the label mask, then the prompt template, then the checkpoint's config sidecar, in that order.

  ![empty output and a repetition loop, with two MindSpore warnings above it](docs/img/empty-output-repetition.jpg)

  *The actual run: the diagnostic line, a collapsed `234` loop, and two warnings worth knowing — a harmless missing `libaio_plugin.so`, and the loader stripping a `model.` prefix off the checkpoint keys.*
- **Repetition loops** appeared once the model stopped emitting EOS: greedy decoding with no repetition penalty, a small model and a tiny slice of data. Switching to sampling with a repetition penalty is the obvious fix and is not implemented.
- **Serving quality tracked data and training budget, not the framework.** A change that looked like a numerical bug — loss falling while outputs degraded — always traced back to the data or the step count, never to a MindSpore operator. The overfit checkpoint is fluent and the 2,000-step full-data checkpoint is empty; the decode path is identical in both, so the variable is the budget, not the port.
- **At `batch_size 1` the loss is noise.** Six consecutive logged steps of the long SFT run read 2.6462, 0.9719, 0.9414, 1.5053, 1.2565, 1.5769 — one sample each. That scatter is the argument for `--grad_accum` in one line.
- **Memory, not compute, set the training recipe.** The next-token logits alone cost `batch × seq × vocab × 4 B`, so the micro-batch that actually fits an 11 GB card is small. Everything beyond it has to come from `--grad_accum` and from the number of optimizer steps — the two knobs that leave the model untouched.
- **The hardest part was the environment, not the code**: aligning driver, CUDA and MindSpore build on openEuler cost more hours than the entire model port.

## Limitations

Stated plainly, because a migration is only useful if its boundary is explicit:

- **No numerical parity harness.** Weights were never converted from PyTorch, and no test asserts that a PyTorch and a MindSpore forward pass agree on the same input. The overfit check shows the port is *internally* consistent and learns; it cannot show that the logits match the reference implementation. For a migration study this is the most important missing piece.
- **PPO/GRPO are not RL.** They are reward-weighted supervised updates over pre-generated samples: no rollout, no policy ratio, no clipping, no value model, no group-normalised advantage. `train_ppo.py` and `train_grpo.py` exist to prove the data and gradient paths, and they are labelled as simplified everywhere.
- **The MoE block is dense.** Every expert is evaluated and then masked, so it costs more than the dense FFN it replaces. Correct arithmetic, no efficiency claim.
- **Greedy decoding only**, no KV cache (each new token is a full forward pass over the prefix), no batched inference, no sampling, no repetition penalty.
- **The web demo is a demo.** No authentication, no rate limiting, no streaming output, and one generation at a time — a request queue, not a serving system.
- **PyNative mode only.** Graph mode (`ms.GRAPH_MODE`) was not validated; several ops here are PyNative-idiomatic and would need review.
- **Single GPU, no distributed training**, no mixed precision.
- **No unit tests.** `run_check.py` and `train_smoke.py` are smoke tests, not a test suite.

## Roadmap

1. A PyTorch→MindSpore weight-conversion script and a numerical parity test (max abs diff on logits for a fixed seed). Everything else is worth less than this.
2. KV cache and sampling (temperature, top-p, repetition penalty) so generation is usable and failure modes become separable.
3. A real PPO/GRPO implementation: rollout loop, clipped ratio, reference KL.
4. Sparse MoE dispatch, plus a graph-mode audit.
5. A reproducibility harness that records command, commit hash, framework/driver versions and seed per run.

## Attribution and licence

Derived from and inspired by [MiniMind](https://github.com/jingyaogong/minimind) by Jingyao Gong, licensed under Apache 2.0. This repository re-implements its architecture and training objectives in MindSpore; the upstream tokenizer and datasets are used unmodified and are not redistributed. See [`NOTICE`](NOTICE) for details. This project is released under the same licence, [Apache 2.0](LICENSE).

If you use the upstream model, data or code, cite the original project — not this repository.

## Author

**Liu Zhuo** — B.Eng. in Computer Science and Technology, East China Normal University.
Developed during a research internship at the Hardware Laboratory, ECNU .

Issues and corrections are welcome. If you find an op-level discrepancy against the PyTorch reference, that is exactly the kind of report this repository wants.

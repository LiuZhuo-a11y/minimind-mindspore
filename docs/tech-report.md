# Porting a Small-LLM Training Stack from PyTorch to MindSpore

### A field report from one GTX 1080 Ti on openEuler

**Liu Zhuo** — East China Normal University, Hardware Laboratory research internship (2025–2026)

*Code: this repository · Evidence: [`experiment-log.md`](experiment-log.md) · Porting notes: [`migration-notes.md`](migration-notes.md) · Parity matrix: [`../MIGRATION_STATUS.md`](../MIGRATION_STATUS.md)*

---

## At a glance

| Item | Value |
|---|---|
| **What was built** | A re-implementation of the [MiniMind](https://github.com/jingyaogong/minimind) model and five training objectives in MindSpore: pre-training, SFT, LoRA, DPO, and reward-weighted PPO/GRPO |
| **Scale** | 80,622,336 parameters · ≈1.6 k lines of Python (code only) · 6 training entry points · 3 inference front ends |
| **Hardware** | One NVIDIA GTX 1080 Ti (11 GB), openEuler 24.03 LTS SP4, CUDA 11.6, MindSpore 2.9.0, `float32` |
| **What was measured** | 20-step overfit check (loss 8.8870 → 7.7097), pre-training at 0.150 s/step, SFT at 0.516 s/step, ~2.2 M padded tokens in 22 minutes |
| **What is missing** | No numerical parity test against PyTorch, no evaluation set, no complete loss curve |

## Abstract

This report describes moving a compact LLM training stack from PyTorch to MindSpore and running it on a single consumer GPU. The port is a re-implementation of the compute graph rather than a source-to-source translation: `nn.Cell` replaces `nn.Module`, `construct` replaces `forward`, `value_and_grad` replaces `backward()`, and every shape is spelled out explicitly. The result trains, and a 20-step overfit check on 128 samples takes the loss from **8.8870 to 7.7097** and yields a checkpoint that generates fluent text — which is a correctness result, not a quality one.

The more useful output of the work is a set of measurements and three findings: that the first-step loss is a free diagnostic for the entire data-and-loss path; that SFT steps cost 3.43× pre-training steps for only 2.26× the sequence length, which pins the cost on attention and the vocabulary-sized logits; and that a sequence-length change between pre-training and fine-tuning is **silent** in this stack, because position tables are rebuilt from config rather than loaded from the checkpoint. The port has no parity test against the reference implementation, and this report treats that as the project's principal limitation rather than a footnote.

## 1. Why port at all

PyTorch is the default, which is exactly why "can this be done in something else" is worth answering with working code rather than an opinion. Two practical motivations drove the choice of target:

- **A domestic software stack.** The environment was openEuler plus MindSpore, both of which are common in Chinese industrial deployments and rare in Western research code. Building an LLM stack on that pair is an exercise in reading framework internals, because almost no community example does what you need.
- **A hardware cliff.** An 11 GB 1080 Ti predates the kernels modern training code assumes: no `bfloat16`, no FlashAttention, no `torch.compile`. Everything has to be done in `float32` with a micro-batch small enough to fit, which makes memory — not FLOPs — the binding constraint on the training recipe.

The upstream project is deliberately tiny and deliberately PyTorch. That makes it a good subject: small enough to port completely, real enough that the port has to handle GQA, RoPE, a mixture-of-experts block, LoRA adapters and four different loss functions.

**Scope.** This is not a weight-compatible port. Weights were never converted from the PyTorch checkpoint, and no test asserts that the two implementations agree numerically. What is claimed is narrower and checkable: the model, the losses and the data pipeline run in MindSpore, share the unmodified upstream tokenizer and JSONL data, and produce a model that learns.

## 2. What was built

The model is a pre-norm decoder in the LLaMA family.

| Field | Value | | Field | Value |
|---|---|---|---|---|
| `vocab_size` | 6400 | | `hidden_size` | 768 |
| layers | 8 | | `num_attention_heads` | 8 |
| `intermediate_size` | 3072 | | `num_key_value_heads` | 4 (GQA, `head_dim` 96) |
| `max_seq_len` | 340 (pre-train) / 768 (SFT) | | `rope_theta` | 1e6 |
| LM head | separate, untied | | **parameters** | **80,622,336** |

Each block is `x + attn(rmsnorm(x))` followed by `x + ffn(rmsnorm(x))`, with a SwiGLU feed-forward and an optional top-1 mixture-of-experts FFN. Training objectives: next-token pre-training, assistant-turn-only SFT, LoRA on the q/v/gate/up projections, DPO with a frozen reference and `stop_gradient`, and reward-weighted updates standing in for PPO and GRPO. Inference is greedy decoding with three front ends: a single prompt, an interactive loop, and a dependency-free web page.

## 3. The seven porting problems

The model itself is textbook; the delta between the two frameworks is where the work was. Seven problems, in descending order of how much they changed the code:

1. **No autograd API.** `loss.backward(); opt.step()` becomes `ms.value_and_grad(loss_net, None, opt.parameters)`, which returns a *tuple* of gradients. The loss had to be wrapped in an `nn.Cell` that owns the model, because MindSpore differentiates a callable, not a scalar.
2. **Static-graph shape discipline.** Every reshape takes an explicit target tuple and every transpose an explicit permutation; `-1` is not a reliable dimension specifier inside graph cells. Batch and sequence length are read from `x.shape` at call time.
3. **GQA without `repeat_interleave`.** The four KV heads are expanded to eight query heads with `expand_dims` + `tile` + `reshape`, which reproduces the grouped layout exactly but has to be written out.
4. **A causal mask that is data, not an op.** The mask is built once in NumPy as a `max_seq_len × max_seq_len` additive `-10000` matrix and sliced per call. One fewer op in the hot loop, at the cost of a fixed maximum length — and a small numerical difference from `-inf`, which matters only if you later test parity.
5. **Loss masking with no `ignore_index`.** The sparse cross-entropy will index a `-100` target and fail, so labels are sanitised twice: `ops.maximum(labels, 0)` for the gather, and an explicit valid mask in both numerator and denominator, with `max(sum(valid), 1.0)` guarding the all-masked batch. Without that clamp, one fully masked batch produces `NaN` and silently poisons the run.
6. **Gradient accumulation without `backward()`.** The micro-batch is bounded by 11 GB, so the effective batch is rebuilt by summing gradient tuples across micro-batches and dividing once when the window closes (`--grad_accum`). Dividing once rather than scaling each micro-loss keeps the update identical to a single pass over `batch_size × accum_steps` samples, so the effective learning rate does not move when the accumulation factor does.
7. **Checkpoints that do not carry their shape.** `ms.save_checkpoint` writes weights only, so every save also writes a `<ckpt>.json` sidecar of the config dataclass; inference rebuilds the network from that JSON before loading. Section 5.2 shows what happens when training does *not* consult it.

A full PyTorch→MindSpore mapping table, plus thirteen traps documented from this project's own run logs, is in [`migration-notes.md`](migration-notes.md).

## 4. Measurements

All numbers come from one machine and are reproducible from the commands in Appendix A. Timings are what `StepTimer` printed at the end of each run; losses are the per-step values printed by the training loop.

### 4.1 The overfit check is the most informative run

Before any long run, 128 samples were memorised for 20 steps at learning rate 5e-4.

![20-step overfit run: loss 8.8870 falling to 7.7097 over 128 samples](img/loss-smoke-20-steps.jpg)

Two numbers matter.

**The first step starts at chance.** Loss at step 1 is 8.8870. A uniform predictor over a 6400-token vocabulary costs `ln(6400) = 8.7641`, so the model begins 1.4 % above the theoretical floor — the small excess coming from an initial distribution that is not perfectly flat. This is the cheapest diagnostic in the whole project: a first loss of 5.0, or 0.1, would have pointed at a broken loss, a mis-mapped label, or logits in the wrong order, and it costs exactly one training step to rule all three out. Any project that reports a loss curve without checking its first value is leaving that check on the table.

**The loss moves, and the checkpoint talks.** Twenty steps took the loss to 7.7097, a drop of **1.18 nats** that cannot happen by accident on 128 fixed samples. The resulting checkpoint answers `你好` with a fluent, correctly formatted Chinese sentence:

```text
$ python generate.py --ckpt ./checkpoints/overfit_20.ckpt --prompt "你好"
我是由jingyaogong创建的高效AI模型，确保快速高效视的交互。
```

This sentence is memorised from the small fine-tuning set, so it says nothing about quality. What it does establish is the chain that framework ports usually break: tokenizer, prompt template, label mask, attention mask, RoPE table, checkpoint loader and greedy decoder all agree with one another end to end. For a port, that is the result that matters most, and it is cheap: twenty steps, seconds of wall clock.

### 4.2 Cost, and why the sequence length dominates

| Stage | Data | seq | Steps | Elapsed | s/step | Padded tok/s |
|---|---|---|---|---|---|---|
| Pre-training | full `pretrain_t2t_mini` | 340 | 2,000 | 300.6 s | 0.150 | ≈2,260 |
| SFT | full `sft_t2t_mini` | 768 | 2,000 | 1,032.4 s | 0.516 | ≈1,490 |

SFT costs **3.43×** per step for **2.26×** the sequence length. That superlinearity is the fingerprint of two terms: attention, which grows with `T²`, and the logits tensor, which grows with `T × V` and at `V = 6400`, `T = 768`, batch 1 is already 20 MB in `float32` before the backward pass allocates anything. The practical consequence is that `--max_seq_len` is the most expensive knob in this repository, and it is set by the data, not by the hardware.

In total the two runs processed about **2.2 M padded tokens in 22 minutes** of wall clock — a number worth stating plainly, because it is roughly what a modern GPU does in two seconds, and it frames everything else in this report.

### 4.3 At batch size 1, the loss is noise

Continuing SFT to step 4,910 (interrupted with `^C`) brought the per-micro-batch loss into the 1.0–1.6 range:

```text
sft step 4905 loss 2.6462
sft step 4906 loss 0.9719
sft step 4907 loss 0.9414
sft step 4908 loss 1.5053
sft step 4909 loss 1.2565
sft step 4910 loss 1.5769
```

Mean 1.48, range 0.94–2.65 across six consecutive steps. Each value is a single sample. That scatter is not instability; it is the variance of a one-sample gradient estimate, and it is the entire argument for gradient accumulation — the unit of optimisation should not be one sequence. No generation was captured from this checkpoint, so the run shows the objective descends, not that the model improved.

## 5. Failure analysis

### 5.1 EOS collapse

The 2,000-step pre-training plus 2,000-step SFT checkpoint generates nothing at all, or loops on a single token:

![empty output and a 234 repetition loop, with two MindSpore warnings above it](img/empty-output-repetition.jpg)

`[empty output: model generated only special/end tokens]` is the diagnostic added in `generate.py`, and `token ids: [234, 234, ...]` is the collapse with EOS suppression disabled. With assistant-turn-only labels and a small budget, emitting the end-of-sequence token immediately is the cheapest way to lower the loss; once the distribution approaches uniform, arg-max is a fixed point and the model will not leave token 234.

The interesting part is the contrast rather than the failure. The same decode path, on a checkpoint trained 20 steps instead of 2,000, produces fluent text. So the variable is the training budget, not the port. Two caveats keep this honest: the two runs differ in *both* data volume and learning rate (5e-4 for the overfit run, 1e-5 for SFT), so the comparison identifies a direction rather than a cause; and 2,000 steps at batch size 1 is 2,000 samples from a 1.2 GB corpus, which is a rounding error at this vocabulary size.

The mitigation in the repository is deliberately modest: `--min_new_tokens` suppresses EOS for the first few steps so the failure is observable, and the diagnostic branch prints the raw ids instead of a blank line. The real fix — more steps at a sane effective batch, or sampling with a repetition penalty — is not implemented.

### 5.2 The silent sequence-length drift

This is the finding most specific to migration work. Pre-training ran at `max_seq_len` 340; SFT ran at 768, because that is the SFT script's default and the training scripts build their configuration from CLI arguments rather than from the checkpoint's sidecar JSON.

**Nothing complained.** The RoPE table and the causal mask are reconstructed from the config in `__init__`, not loaded as parameters, so architecture mismatches fail loudly while **sequence-length mismatches fail silently**. The fine-tuning run therefore presented the model with positions it had never trained on.

It is worth being precise about the mechanism: only a handful of fields determine parameter *shapes* (`hidden_size`, layer count, head count, vocabulary), and those do fail loudly in `load_checkpoint`. Sequence length changes nothing about the weights — it changes the tables and the mask — so no shape check can catch it. The fix is either to pass the same `--max_seq_len` by hand or to route continuation runs through the same sidecar loader that inference uses. This repository documents the trap and leaves the behaviour unchanged, because the recorded run is the behaviour worth preserving.

## 6. Threats to validity

- **No parity test.** The largest gap. Nothing here demonstrates that MindSpore and PyTorch produce the same logits for the same weights. The overfit check demonstrates internal consistency — the components agree with each other — which is necessary but not sufficient. A correct port could still be numerically wrong in a way that trains fine and generates slightly worse text.
- **One machine, one configuration, no seeds recorded.** Every number above is a single point. The 1080 Ti is the only device, `float32` is the only precision, `PYNATIVE_MODE` is the only execution mode, and no seed was fixed, so the loss values are not expected to reproduce exactly.
- **A confounded comparison.** Section 5.1 compares an overfit run against an SFT run that differ in data volume *and* learning rate. The conclusion drawn from it ("budget, not the port") is a direction, not a measured effect.
- **No evaluation.** There is no held-out set, no benchmark and no quality metric anywhere in this project. The chat, tool-call and web front ends are demonstrations of plumbing.
- **The overfit sentence is memorised.** It is evidence of a working pipeline, not of a working model, and it should not be read as the latter.

## 7. What generalizes

Three things from this project transfer beyond MindSpore, and they are the reason it was worth writing up:

**Check the first loss value.** `ln(vocab_size)` is the floor for a uniform predictor, and a first step that sits near it rules out several categories of bug for free. A first loss far below it usually means the labels leaked into the input, or the target was shifted twice.

**Overfit a tiny batch before spending hours.** Twenty steps on 128 samples is the cheapest end-to-end correctness test for a training stack, because it exercises every path that a full run would and fails in seconds instead of hours. If the loss does not collapse on 128 memorisable samples, the bug is in the code, not the data budget.

**Configuration that changes buffers instead of parameters is invisible to shape checks.** Any field that only affects tables, masks or buffers — sequence length, rope theta, attention window — will not be caught by a `load_checkpoint` mismatch. Those fields need an explicit assertion, or a single source of truth. This is the one lesson here that is really about migrating code between frameworks: the frameworks check what they can see, which is weights.

## Appendix A: reproduce

```bash
# environment (see experiment-log.md for the full setup)
nvidia-smi                       # driver 580.173.02, GTX 1080 Ti
nvcc --version                   # CUDA 11.6
python run_check.py              # MindSpore 2.9.0 reaches the GPU

# 1. correctness check: memorise 128 samples, 20 steps
python train_smoke.py --source_dir /path/to/minimind --device GPU \
  --steps 20 --batch_size 1 --max_samples 128 --save_name overfit_20.ckpt
python generate.py --source_dir /path/to/minimind --device GPU \
  --ckpt ./checkpoints/overfit_20.ckpt --prompt "你好"

# 2. pre-train, then SFT
python train_smoke.py --source_dir /path/to/minimind --device GPU \
  --steps 2000 --batch_size 1 --grad_accum 8 --save_name pretrain_ms.ckpt
python train_sft.py --source_dir /path/to/minimind --device GPU --steps 2000 \
  --batch_size 1 --grad_accum 8 --init_ckpt ./checkpoints/pretrain_ms.ckpt
```

## Appendix B: numbers used in this report

| Quantity | Value | Derivation |
|---|---|---|
| Parameters | 80,622,336 | sum of module shapes, `vocab 6400`, `h 768`, `L 8` |
| Uniform-predictor loss | 8.7641 | `ln(6400)` |
| Overfit run, step 1 → 20 | 8.8870 → 7.7097 | logged loss, 128 samples |
| Overfit drop | 1.18 nats | difference |
| Pre-training step time | 0.150 s | 300.6 s ÷ 2,000 |
| SFT step time | 0.516 s | 1,032.4 s ÷ 2,000 |
| Step-time ratio | 3.43× | 0.516 ÷ 0.150 |
| Sequence-length ratio | 2.26× | 768 ÷ 340 |
| Padded tokens processed | ≈2.22 M | 2,000×340 + 2,000×768 |
| Long SFT run, last six steps | mean 1.48, range 0.94–2.65 | logged losses, batch 1 |

## References

1. Gong, J. *MiniMind: train a small language model from scratch.* <https://github.com/jingyaogong/minimind>
2. MindSpore. *MindSpore documentation.* <https://www.mindspore.cn>
3. Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., Liu, Y. *RoFormer: Enhanced Transformer with Rotary Position Embedding.* arXiv:2104.09864, 2021.
4. Ainslie, J., Lee-Thorp, J., de Jong, Z., He, Y., Žilka, T., Sanghai, S. *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.* arXiv:2305.13245, 2023.
5. Shazeer, N. *GLU Variants Improve Transformer.* arXiv:2002.05202, 2020.
6. Shazeer, N., et al. *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.* arXiv:1701.06538, 2017.
7. Hu, E., et al. *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685, 2021.
8. Rafailov, R., et al. *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* arXiv:2305.18290, 2023.
9. Schulman, J., et al. *Proximal Policy Optimization Algorithms.* arXiv:1707.06347, 2017.
10. Shao, Z., et al. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* arXiv:2402.03300, 2024. (Group Relative Policy Optimization.)
11. Karpathy, A. *A Recipe for Training Neural Networks.* 2019. <http://karpathy.github.io/2019/04/25/recipe/>

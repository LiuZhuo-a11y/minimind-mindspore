# Feature-level parity matrix

Component-by-component comparison with the upstream PyTorch MiniMind. The README gives the summary; this file is the detailed record, so that "implemented" is never mistaken for "identical".

Legend: **Yes** — matches upstream behaviour; **Approx** — same intent, simplified implementation; **No** — not ported.

## Model

| Feature | Upstream | This repo | Parity | Note |
|---|---|---|---|---|
| Causal LM decoder | `nn.Module` | `nn.Cell` | Yes | pre-norm blocks, residual around attention and FFN |
| RMSNorm | `x * rsqrt(mean(x²)+eps) * w` | same, computed in `float32` | Yes | explicit cast in, cast back out |
| RoPE | `torch` complex/`cos`-`sin` pair | precomputed NumPy table, sliced | Yes | `rope_theta = 1e6`; table shape fixed at `max_seq_len` |
| Grouped-query attention | `repeat_interleave` on KV heads | `expand_dims` + `tile` + `reshape` | Yes | identical head ordering by construction |
| Causal mask | `triu` / `masked_fill` with `-inf` | additive `-10000` matrix, sliced | Approx | `-10000` instead of `-inf`; numerically equivalent in `float32` after softmax |
| SwiGLU FFN | `silu(gate) * up → down` | same | Yes | no bias terms |
| MoE FFN | top-k sparse dispatch | top-1, **dense loop over all experts**, masked | Approx | arithmetic equivalent, no compute saving |
| Weight tying | configurable | not tied (`lm_head` separate) | Approx | matches the default upstream configuration |
| Dropout | present, `p = 0` default | `p = 0` only | Approx | parameter exists in config, not applied |

## Training

| Feature | Upstream | This repo | Parity | Note |
|---|---|---|---|---|
| Pre-training | next-token, full-text labels | same, pad → `-100` | Yes | `train_smoke.py` |
| SFT | assistant-turn labels via template | same, hand-built template | Yes | `render_chat_prompt`, `generate_sft_labels` |
| LoRA | `peft` | hand-written `LoRADense` (q/v/gate/up) | Approx | no `merge_and_unload`, no `lora_dropout` |
| DPO | `trl`-style reference model | `DPOLossNet`, `beta`, `stop_gradient` ref | Yes | both policy and ref loaded from the same checkpoint |
| PPO | rollout + clipped ratio + value model | reward-weighted SFT | **No** | no rollout, no ratio, no critic; see README *Limitations* |
| GRPO | grouped rollout + group-normalised advantage | reward-weighted SFT | **No** | same data, same objective as the PPO path |
| Gradient accumulation | `loss / N` + repeated `backward()` | `GradientAccumulator` over the gradient tuple | Yes | identical effective batch; `--steps` counts optimizer updates |
| Gradient clipping | `clip_grad_norm_` | absent | **No** | relies on `AdamWeightDecay` + small `lr` |
| LR schedule | warmup + cosine | constant `lr` | **No** | |
| Mixed precision | AMP | `float32` throughout | **No** | 1080 Ti has no `bfloat16` |
| Distributed | optional | single device | **No** | |

## Inference and evaluation

| Feature | Upstream | This repo | Parity | Note |
|---|---|---|---|---|
| Chat generation | `transformers` `GenerationMixin` | hand-written loop | Approx | |
| Decoding strategy | sampling + temperature / top-p | **greedy only** | **No** | |
| KV cache | yes | no | **No** | every step is a full forward over the prefix |
| Repetition penalty | yes | no | **No** | direct cause of the observed loops |
| Batched generation | yes | no | **No** | |
| Web UI | streamlit / gradio app | stdlib `http.server` + vanilla JS | Approx | one page, one JSON endpoint, zero extra dependencies |
| Chat template | tokenizer Jinja template | hand-built string | Approx | deliberately identical at train and test time |
| Tool calling | tool-call format + real tools | format + three mock tools | Approx | pipeline demo, not a capability benchmark |
| Benchmark evaluation | `eval_llm.py` on standard sets | absent | **No** | no held-out evaluation set |

## Portability

| Aspect | Status |
|---|---|
| MindSpore execution mode | `PYNATIVE_MODE` only; `GRAPH_MODE` untested |
| Numerical parity tests | none — the largest gap in this port |
| PyTorch weight conversion | not implemented |
| Unit tests | none; `run_check.py` and `train_smoke.py` are smoke tests |
| Platform validated | openEuler + CUDA 11.6 + single GTX 1080 Ti |

## Maintenance intent

This repository is a finished study artefact, not a funded project. The gaps above are recorded rather than hidden, and the ones worth closing first are ordered in the README's roadmap — parity testing before any new feature.

# Porting notes: PyTorch → MindSpore

Written while moving MiniMind's model and training objectives to MindSpore. The point of this file is to record *why* the code looks the way it does, so a reader (or a future me) does not have to rediscover each obstacle.

Every claim below is observable in this repository's code.

---

## 1. Concept mapping

| PyTorch | MindSpore | Note |
|---|---|---|
| `nn.Module` | `nn.Cell` | same idea, `__init__` + a method for the forward pass |
| `forward(self, x)` | `construct(self, x)` | `construct` is the graph entry point; it must return a tensor |
| `nn.Linear(in, out, bias=False)` | `nn.Dense(in, out, has_bias=False)` | the argument is `has_bias`, and it defaults to **True** |
| `nn.Embedding(v, h)` | `nn.Embedding(v, h)` | same |
| `x.view(b, t, h, d)` | `ops.reshape(x, (b, t, h, d))` | shape must be an explicit tuple |
| `x.transpose(1, 2)` | `ops.transpose(x, (0, 2, 1, 3))` | full permutation, not a swap of two axes |
| `torch.cat` | `ops.concat` | `axis=` instead of `dim=` |
| `x.clamp(min=0)` | `ops.maximum(x, Tensor(0.0, ms.float32))` | the scalar must be a typed `Tensor` |
| `torch.nn.functional.cross_entropy(..., reduction="none")` | `nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="none")` | returns per-sample loss, shape `(N,)` |
| `F.log_softmax` | `nn.LogSoftmax(axis=-1)` | same |
| `torch.no_grad()` on a reference model | `ops.stop_gradient(...)` | applied per-tensor at the call site |
| `loss.backward(); opt.step()` | `ms.value_and_grad(net, None, opt.parameters)` → `opt(grads)` | no `.grad` on parameters |
| `nn.Parameter(t)` | `Parameter(t)` | `requires_grad` is an attribute set **after** construction |
| `param.requires_grad = False` | `for _, p in net.parameters_and_names(): p.requires_grad = ...` | iterate names to freeze by pattern |
| `DataLoader(dataset, batch_size, shuffle)` | `GeneratorDataset(ds, column_names=[...]).batch(bs, drop_remainder=True)` | column order is a hard contract |
| `tensor.cpu().numpy()` | `tensor.asnumpy()` | synchronises to host |
| `model.eval()` / `model.train()` | `model.set_train(False)` / `set_train(True)` | |
| `model.state_dict()` | `ms.save_checkpoint(model, path)` | weights only — shape is not stored |
| `torch.manual_seed` + `cuda` context | `ms.set_context(mode=ms.PYNATIVE_MODE, device_target="GPU")` | call before creating any tensor |

## 2. Module by module

**RMSNorm.** PyTorch usually computes `x * rsqrt(mean(x^2) + eps) * w`. Here the cast to `float32` is explicit on the way in (`ops.cast(x, ms.float32)`) and the result is cast back to the input dtype before scaling, so the reduction is stable regardless of the incoming dtype. `ops.rsqrt` exists and is the same op.

**RoPE.** Built as a NumPy table (`build_rope_cache`) for `max_seq_len` positions, stored as a `Tensor` attribute of the attention cell, and sliced to `[:seq_len]` per call. `rotate_half` splits the head dimension in two with plain slicing and `ops.concat`, because there is no `torch.cat`-with-negation idiom to lean on. The `cos`/`sin` tables are duplicated along the feature axis (`concatenate([h, h])`) so the rotation is a single elementwise multiply-add rather than a `stack`.

**Attention.** `_split_heads` does `reshape` + `transpose` instead of `view` + `transpose`, and takes `batch_size`/`seq_len` as arguments read from `x.shape`, because MindSpore does not always propagate `-1` dimensions the way PyTorch does inside graph cells. The causal mask is a precomputed additive `-10000` matrix (upper triangle) stored on the cell and sliced to `[:seq_len, :seq_len]`: a `0/-10000` additive mask rather than a boolean `masked_fill`, since `masked_fill` semantics differ between backend versions.

**GQA.** `_repeat_kv` does `expand_dims(x, 2)` → `tile(x, (1, 1, kv_repeat, 1, 1))` → `reshape` to `(b, kv_heads * kv_repeat, t, d)`. This reproduces `repeat_interleave(dim=1)` exactly for the grouped layout, and keeps the head ordering that the reshape in `construct` expects.

**MoE.** The router produces `softmax(scores)`, a top-1 `one_hot` mask is applied, and the resulting per-token weights are renormalised. All experts run and are then masked, so the block is arithmetically equivalent to the sparse version but not compute-equivalent; that trade was accepted deliberately to keep gradients easy to verify.

**Losses.** `CausalLMLoss` shifts by one, flattens to `(B*T, V)`, and divides by `max(sum(valid), 1.0)`. Take away the clamp and a batch whose labels are all `-100` divides by zero and produces `NaN` that silently poisons the run. `safe_labels = ops.maximum(flat_labels, 0)` exists because the sparse cross-entropy gathers `logits[:, target]` and a target of `-100` is a device-side index error.

**DPO.** Two model instances share one config; the reference copy is loaded from the same checkpoint and its log-probs pass through `ops.stop_gradient`. Only `policy.trainable_params()` is handed to the optimizer, which is what actually keeps the reference frozen — `stop_gradient` prevents gradient flow, but the optimizer's parameter list is what defines the update set. Both matter.

**LoRA.** `LoRADense` wraps an existing `nn.Dense` and adds A (`N(0, 0.02)`) and B (zeros), scaled by `alpha / rank`. Freezing happens after the whole model and all LoRA wrappers exist, by name (`"lora_" in name`), because `Parameter.requires_grad` is set on instances and the optimizer must be constructed **after** freezing or it will capture parameters that are still marked trainable.

**Gradient accumulation.** The usual PyTorch recipe is `loss = loss / accum_steps` followed by `backward()` on each micro-batch, letting autograd do the summing into `.grad`. MindSpore has neither `.grad` nor `backward()`: `ms.value_and_grad` is a *function* that returns `(loss, grads)`, and `grads` is a tuple of tensors with the same order as `optimizer.parameters`. So accumulation has to be done by hand:

```python
accum = GradientAccumulator(accum_steps)
loss, grads = grad_fn(input_ids, labels)   # one micro-batch
if accum.add(grads):                       # sums the tuple element-wise
    optimizer(accum.take())                # divides by accum_steps, returns mean
```

Three details make this correct rather than merely plausible:

- Dividing **once at window close** (rather than scaling each micro-loss) gives exactly the gradient of a single pass over `batch_size × accum_steps` samples, so the learning rate does not have to be retuned when the accumulation factor changes.
- The accumulated list is initialised from the first micro-batch's tuple, then every later batch goes through `ops.add`, which returns a new tensor — the caller's gradients are never mutated in place.
- `--steps` counts *optimizer updates*, which is what a "training step" usually means in a paper; the epoch loop re-enters the iterator, so a window is never left half-open even when the dataset ends mid-accumulation.

## 3. Traps worth knowing

- **`set_context` is global and order-dependent.** It must run before the first tensor or cell is created in the process; calling it after model construction leaves the model on the previous target.
- **Scalars in ops need a dtype.** `ops.maximum(x, 0)` fails or silently promotes; `Tensor(0.0, ms.float32)` is the portable form. Same for `one_hot`'s `on_value`/`off_value`.
- **`one_hot` wants int32 indices.** Labels arrive as `int32` end-to-end (`np.array(..., dtype=np.int32)`) for this reason; an `int64` label tensor raises at the gather.
- **`GeneratorDataset` columns are positional.** The DPO dataset returns a 4-tuple, so `column_names` must list exactly `chosen_ids, chosen_labels, rejected_ids, rejected_labels` in that order. Get the order wrong and training runs happily on swapped tensors — no error, just nonsense.
- **No default collate.** Every sample must already be a fixed-shape array; padding is therefore done in the dataset (`max_seq_len`), and `drop_remainder=True` keeps the last partial batch from changing the graph shape.
- **No in-place writes on tensors.** The EOS-suppression hack in `generate.py` copies logits to NumPy (`next_logits.asnumpy()[0]`), edits the array, re-argmaxes there — rather than assigning into a tensor.
- **`asnumpy()` is a sync point.** It is used only for logging and for the greedy step, never inside a training step.
- **`ms.load_checkpoint(net=model)` is strict about names and shapes**, which is precisely why the config sidecar exists: load the JSON, build the cell, then load the weights.
- **`trainable_params()` is not a live view.** Building the optimizer before freezing produces an optimizer that still updates the "frozen" weights.
- **`nn.Dense` defaults to `has_bias=True`.** Every projection in a LLaMA-style block is bias-free; a forgotten `has_bias=False` adds ~0.1 % parameters and slightly different numerics.

Three traps below are not speculation — each one appeared in this project's own run logs.

- **`set_context(device_target=...)` is deprecated in MindSpore 2.9.** The run emits, once per process:

  ```text
  For 'context.set_context', the parameter 'device_target' will be deprecated and
  removed in a future version. Please use the api mindspore.set_device() instead.
  ```

  The 2.9 spelling is `ms.set_context(mode=...)` then `ms.set_device(device_target=...)`. This repository keeps the older single call on purpose: it also runs on 2.x builds that predate `set_device`, and a warning is cheaper than a hard dependency on a minor version.

- **A checkpoint's keys may not match the cell tree.** Loading the SFT checkpoint prints:

  ```text
  For 'load_param_into_net', remove parameter prefix name: model., continue to load.
  ```

  The saved parameter names carried a `model.` prefix that the cell tree does not have; MindSpore strips it and continues. That is helpful, and it also means "the weights loaded" is not by itself evidence that the parameter ordering is what you think. The name map is worth writing down explicitly whenever weights cross a framework boundary.

- **The config sidecar is read at inference, but not when continuing training.** `generate.py` restores the trained shape from `<ckpt>.json`; `train_sft.py`, `train_lora.py`, `train_dpo.py`, `train_ppo.py` and `train_grpo.py` build their config from CLI arguments instead. The consequence is visible in this project's own runs: pre-training used `max_seq_len` 340 and SFT used 768, and **nothing complained**, because the RoPE table and the causal mask are reconstructed from the config rather than loaded as parameters. The model simply saw positions it had never trained on. Architecture fields (`hidden_size`, layer count, head count) do fail loudly on a mismatch; sequence length does not. If you continue from a checkpoint, pass the same `--max_seq_len` and model flags, or teach the training scripts to call `resolve_config_for_ckpt` as well.

## 4. What true parity would require

Not implemented here, listed so the gap is concrete:

1. A name map from PyTorch state-dict keys to MindSpore parameter names (`model.layers.0.self_attn.q_proj.weight` → `layers.0.attn.q_proj.weight`), including the LoRA wrapper prefixes.
2. The same tokenizer (already shared) and the same `max_seq_len` and mask convention, since the additive `-10000` mask is not bit-identical to `-inf`.
3. A fixed seed and a deterministic forward pass on both sides, then `max |logit_torch - logit_ms|` on a short prompt. A healthy port lands around `1e-5`–`1e-4` in `float32`; the interesting failures show up in the attention softmax and in RoPE's `float32` table.
4. A gradient check, not only a forward check: one backward pass compared per-parameter, which catches a silent mistake in the loss reduction (for example summing instead of averaging over valid tokens).

Until step 1–4 exist, "the loss goes down" is evidence that the pipeline runs, not that the port is correct. That distinction is the main methodological lesson of this project.

## 5. Performance envelope observed

- Single GTX 1080 Ti, 11 GB, `PYNATIVE_MODE`, `float32` only.
- No KV cache: generating `n` tokens costs `n` full forward passes over a growing prefix, so `n = 80` at `max_seq_len = 128` is slow but usable for a demo, and is the reason `eval_chat.py` feels sluggish.
- The binding constraint is the logits tensor: `batch × seq × vocab × 4 bytes`. At `vocab = 6400` and `seq = 340`, `batch = 32` already allocates ~278 MB for that one tensor before the backward pass.
- All numbers here are structural (derived from the shapes in the code), not benchmarked — no timing claims are made.

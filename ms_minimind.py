from dataclasses import dataclass
import math
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor, Parameter
import mindspore.ops as ops


@dataclass
class MiniMindMSConfig:
    """Model configuration for the MindSpore port of MiniMind.

    These fields define the model shape. They must be identical at training
    and inference time, so every checkpoint is stored next to a JSON copy of
    this configuration.
    """
    vocab_size: int = 6400
    hidden_size: int = 256
    num_hidden_layers: int = 4
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    max_seq_len: int = 128
    intermediate_size: int = 1024
    dropout: float = 0.0
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1e6
    use_moe: bool = False
    num_experts: int = 4


class RMSNorm(nn.Cell):
    """RMSNorm layer.

    LLaMA-style models normalize with RMSNorm instead of the LayerNorm of the
    original Transformer: no mean centering, fewer operations, and stable
    training once the residual stream is scaled correctly.
    """
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = Parameter(ops.ones((hidden_size,), ms.float32))

    def construct(self, x):
        x32 = ops.cast(x, ms.float32)
        variance = ops.mean(ops.square(x32), axis=-1, keep_dims=True)
        x_norm = x32 * ops.rsqrt(variance + self.eps)
        return ops.cast(x_norm, x.dtype) * self.weight


class FeedForward(nn.Cell):
    """Feed-forward network of a Transformer block.

    Uses the gated ``silu(gate_proj(x)) * up_proj(x)`` structure of the LLaMA
    family (SwiGLU), with no bias terms on any projection.
    """
    def __init__(self, config: MiniMindMSConfig):
        super().__init__()
        self.gate_proj = nn.Dense(config.hidden_size, config.intermediate_size, has_bias=False)
        self.up_proj = nn.Dense(config.hidden_size, config.intermediate_size, has_bias=False)
        self.down_proj = nn.Dense(config.intermediate_size, config.hidden_size, has_bias=False)
        self.silu = nn.SiLU()

    def construct(self, x):
        return self.down_proj(self.silu(self.gate_proj(x)) * self.up_proj(x))


class MoEFeedForward(nn.Cell):
    """Lightweight mixture-of-experts feed-forward layer.

    A router scores all experts, the top-1 expert is selected per token, and
    its output becomes the FFN output. This is a minimal port of the MiniMind
    MoE block, not a production sparse-MoE kernel: every expert is still
    evaluated and then masked by its gate, so it costs *more* FLOPs than the
    dense FFN it replaces. Keeping the dense loop makes the arithmetic exact
    and the gradients easy to reason about while porting.
    """
    def __init__(self, config: MiniMindMSConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.router = nn.Dense(config.hidden_size, config.num_experts, has_bias=False)
        self.experts = nn.CellList([FeedForward(config) for _ in range(config.num_experts)])
        self.softmax = nn.Softmax(axis=-1)

    def construct(self, x):
        scores = self.softmax(ops.cast(self.router(x), ms.float32))
        top_idx = ops.Argmax(axis=-1)(scores)
        one_hot = ops.one_hot(top_idx, self.num_experts, Tensor(1.0, ms.float32), Tensor(0.0, ms.float32))
        weights = scores * one_hot
        weights = weights / ops.maximum(ops.sum(weights, axis=-1, keep_dims=True), Tensor(1e-9, ms.float32))
        mixed = None
        for i, expert in enumerate(self.experts):
            expert_out = expert(x)
            weight = weights[:, :, i:i + 1]
            mixed = expert_out * weight if mixed is None else mixed + expert_out * weight
        return mixed


def build_rope_cache(max_seq_len, head_dim, theta):
    """Precompute the cos/sin tables RoPE needs.

    RoPE does not add a positional embedding, it rotates the q/k vectors. The
    table is built once for the maximum sequence length and sliced per call,
    so no trigonometry happens inside the training step.
    """
    idx = np.arange(0, head_dim, 2, dtype=np.float32)
    inv_freq = 1.0 / (theta ** (idx / head_dim))
    positions = np.arange(max_seq_len, dtype=np.float32)
    freqs = np.outer(positions, inv_freq)
    cos_half = np.cos(freqs)
    sin_half = np.sin(freqs)
    cos = np.concatenate([cos_half, cos_half], axis=-1).reshape(1, 1, max_seq_len, head_dim)
    sin = np.concatenate([sin_half, sin_half], axis=-1).reshape(1, 1, max_seq_len, head_dim)
    return Tensor(cos.astype(np.float32), ms.float32), Tensor(sin.astype(np.float32), ms.float32)


def rotate_half(x):
    """Half-rotation of the feature dimension, the core of RoPE."""
    half = x.shape[-1] // 2
    return ops.concat((-x[:, :, :, half:], x[:, :, :, :half]), axis=-1)


class CausalSelfAttention(nn.Cell):
    """Self-attention with a causal mask, RoPE and grouped-query attention.

    The causal mask stops a position from attending to future tokens, RoPE
    injects position information, and GQA lets several query heads share a
    smaller set of key/value heads to save memory and compute.
    """
    def __init__(self, config: MiniMindMSConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads or config.num_attention_heads
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        self.kv_repeat = self.num_heads // self.num_kv_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.max_seq_len = config.max_seq_len
        self.q_proj = nn.Dense(config.hidden_size, config.hidden_size, has_bias=False)
        self.k_proj = nn.Dense(config.hidden_size, self.num_kv_heads * self.head_dim, has_bias=False)
        self.v_proj = nn.Dense(config.hidden_size, self.num_kv_heads * self.head_dim, has_bias=False)
        self.o_proj = nn.Dense(config.hidden_size, config.hidden_size, has_bias=False)
        mask = np.tril(np.ones((config.max_seq_len, config.max_seq_len), dtype=np.float32))
        mask = (1.0 - mask) * -10000.0
        self.mask = Tensor(mask.reshape(1, 1, config.max_seq_len, config.max_seq_len), ms.float32)
        self.softmax = nn.Softmax(axis=-1)
        self.rope_cos, self.rope_sin = build_rope_cache(config.max_seq_len, self.head_dim, config.rope_theta)

    def _split_heads(self, x, batch_size, seq_len, heads):
        """Split [batch, seq, hidden] into [batch, heads, seq, head_dim]."""
        x = ops.reshape(x, (batch_size, seq_len, heads, self.head_dim))
        return ops.transpose(x, (0, 2, 1, 3))

    def _apply_rope(self, q, k, seq_len):
        """Apply the cached RoPE rotation to q and k."""
        cos = self.rope_cos[:, :, :seq_len, :]
        sin = self.rope_sin[:, :, :seq_len, :]
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        return q, k

    def _repeat_kv(self, x):
        """Expand the few k/v heads to the q head count (the GQA trick)."""
        if self.kv_repeat == 1:
            return x
        bsz, kv_heads, seq_len, head_dim = x.shape
        x = ops.expand_dims(x, 2)
        x = ops.tile(x, (1, 1, self.kv_repeat, 1, 1))
        return ops.reshape(x, (bsz, kv_heads * self.kv_repeat, seq_len, head_dim))

    def construct(self, x):
        batch_size, seq_len, _ = x.shape
        # q keeps the full head count while k/v use num_kv_heads: this is the
        # defining difference of grouped-query attention.
        q = self._split_heads(self.q_proj(x), batch_size, seq_len, self.num_heads)
        k = self._split_heads(self.k_proj(x), batch_size, seq_len, self.num_kv_heads)
        v = self._split_heads(self.v_proj(x), batch_size, seq_len, self.num_kv_heads)
        q, k = self._apply_rope(q, k, seq_len)
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)
        scores = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) / math.sqrt(self.head_dim)
        # Add the upper-triangular mask so the model cannot look ahead.
        scores = scores + self.mask[:, :, :seq_len, :seq_len]
        attn = self.softmax(ops.cast(scores, ms.float32))
        y = ops.matmul(ops.cast(attn, x.dtype), v)
        y = ops.transpose(y, (0, 2, 1, 3))
        y = ops.reshape(y, (batch_size, seq_len, self.hidden_size))
        return self.o_proj(y)


class MiniMindBlock(nn.Cell):
    """One Transformer block: attention + FFN, each behind its own residual."""
    def __init__(self, config: MiniMindMSConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ffn = MoEFeedForward(config) if config.use_moe else FeedForward(config)

    def construct(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class MiniMindForCausalLMMS(nn.Cell):
    """MiniMind causal language model.

    Takes token ids and returns next-token logits for every position.
    """
    def __init__(self, config: MiniMindMSConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.CellList([MiniMindBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Dense(config.hidden_size, config.vocab_size, has_bias=False)

    def construct(self, input_ids):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)


class CausalLMLoss(nn.Cell):
    """Language-model training loss.

    Shifted target: the logits at position t predict token t+1. Positions
    whose label is -100 are dropped from both numerator and denominator, which
    is how the SFT path removes user/system text from the objective.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="none")

    def construct(self, input_ids, labels):
        logits = self.model(input_ids)
        vocab_size = logits.shape[-1]
        # Align the autoregressive target: logits[:, :-1] predicts labels[:, 1:].
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        flat_logits = ops.reshape(shift_logits, (-1, vocab_size))
        flat_labels = ops.reshape(shift_labels, (-1,))
        valid = ops.cast(flat_labels != -100, ms.float32)
        safe_labels = ops.maximum(flat_labels, Tensor(0, ms.int32))
        token_loss = self.loss(flat_logits, safe_labels)
        return ops.sum(token_loss * valid) / ops.maximum(ops.sum(valid), Tensor(1.0, ms.float32))


class WeightedCausalLMLoss(nn.Cell):
    """Language-model loss with per-sample weights.

    The simplified PPO/GRPO paths reuse the stored reward as a sample weight so
    high-reward samples contribute more gradient. This is importance weighting,
    not a policy-gradient estimator.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="none")

    def construct(self, input_ids, labels, rewards):
        logits = self.model(input_ids)
        vocab_size = logits.shape[-1]
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        flat_logits = ops.reshape(shift_logits, (-1, vocab_size))
        flat_labels = ops.reshape(shift_labels, (-1,))
        valid = ops.cast(flat_labels != -100, ms.float32)
        safe_labels = ops.maximum(flat_labels, Tensor(0, ms.int32))
        token_loss = self.loss(flat_logits, safe_labels)
        token_loss = ops.reshape(token_loss, shift_labels.shape)
        valid_2d = ops.reshape(valid, shift_labels.shape)
        seq_loss = ops.sum(token_loss * valid_2d, axis=1) / ops.maximum(
            ops.sum(valid_2d, axis=1), Tensor(1.0, ms.float32)
        )
        weights = ops.maximum(rewards, Tensor(0.0, ms.float32))
        return ops.mean(seq_loss * weights)


class LoRADense(nn.Cell):
    """LoRA low-rank adaptation layer.

    The wrapped Dense keeps its original weight while two small matrices A and
    B are trained, cutting the number of trainable fine-tuning parameters.
    """
    def __init__(self, base_dense: nn.Dense, rank=8, alpha=16):
        super().__init__()
        self.base = base_dense
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        in_channels = base_dense.in_channels
        out_channels = base_dense.out_channels
        self.lora_a = nn.Dense(in_channels, rank, has_bias=False)
        self.lora_b = nn.Dense(rank, out_channels, has_bias=False)
        self.lora_a.weight.set_data(
            Tensor(np.random.normal(0.0, 0.02, self.lora_a.weight.shape).astype(np.float32))
        )
        self.lora_b.weight.set_data(Tensor(np.zeros(self.lora_b.weight.shape, dtype=np.float32)))

    def construct(self, x):
        return self.base(x) + self.lora_b(self.lora_a(x)) * self.scaling


def freeze_non_lora(model):
    """Freeze every parameter whose name does not contain ``lora_``."""
    for name, param in model.parameters_and_names():
        param.requires_grad = "lora_" in name


def apply_lora_to_model(model, rank=8, alpha=16):
    """Swap the q/v and FFN projections for their LoRA-wrapped versions."""
    for layer in model.layers:
        layer.attn.q_proj = LoRADense(layer.attn.q_proj, rank=rank, alpha=alpha)
        layer.attn.v_proj = LoRADense(layer.attn.v_proj, rank=rank, alpha=alpha)
        layer.ffn.gate_proj = LoRADense(layer.ffn.gate_proj, rank=rank, alpha=alpha)
        layer.ffn.up_proj = LoRADense(layer.ffn.up_proj, rank=rank, alpha=alpha)
    freeze_non_lora(model)
    return model

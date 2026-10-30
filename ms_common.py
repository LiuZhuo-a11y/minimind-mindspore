import json
import os
import time
from dataclasses import asdict

import mindspore as ms
import numpy as np
import mindspore.ops as ops
from mindspore.dataset import GeneratorDataset
from transformers import AutoTokenizer

from ms_minimind import MiniMindMSConfig


def load_tokenizer(source_dir):
    """Load the tokenizer that ships with the upstream MiniMind checkout."""
    return AutoTokenizer.from_pretrained(os.path.join(source_dir, "model"))


def make_config(args, tokenizer):
    """Build the model configuration from CLI arguments and the tokenizer."""
    return MiniMindMSConfig(
        vocab_size=len(tokenizer),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        num_key_value_heads=getattr(args, "num_kv_heads", max(1, args.num_heads // 2)),
        max_seq_len=args.max_seq_len,
        intermediate_size=args.hidden_size * 4,
        use_moe=bool(getattr(args, "use_moe", 0)),
        num_experts=getattr(args, "num_experts", 4),
    )


def save_model_with_config(model, config, save_dir, save_name):
    """Save a MindSpore checkpoint together with a JSON copy of the model shape.

    Inference needs hidden_size, layer count, head count and so on; without the
    sidecar JSON the weights cannot be restored into a matching network.
    """
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, save_name)
    config_path = os.path.join(save_dir, save_name + ".json")
    ms.save_checkpoint(model, ckpt_path)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)
    print(f"saved checkpoint: {ckpt_path}")
    print(f"saved config: {config_path}")
    return ckpt_path


def load_config(config_path):
    """Restore the model configuration from its JSON sidecar."""
    with open(config_path, "r", encoding="utf-8") as f:
        return MiniMindMSConfig(**json.load(f))


def resolve_config_for_ckpt(ckpt_path, fallback_config):
    """Prefer the JSON sidecar next to the checkpoint, fall back to defaults."""
    config_path = ckpt_path + ".json"
    if os.path.exists(config_path):
        return load_config(config_path)
    return fallback_config


def encode_pretrain_sample(tokenizer, text, max_seq_len):
    """Turn one raw text sample into input_ids and labels.

    Pre-training is plain next-token prediction, so labels mirror input_ids;
    padded positions become -100 and are ignored by the loss.
    """
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 1
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2
    ids = tokenizer(str(text), add_special_tokens=False).input_ids[: max_seq_len - 2]
    ids = [bos_id] + ids + [eos_id]
    ids = ids + [pad_id] * (max_seq_len - len(ids))
    labels = [(-100 if token_id == pad_id else token_id) for token_id in ids]
    return np.array(ids, dtype=np.int32), np.array(labels, dtype=np.int32)


def encode_chat_sample(tokenizer, conversations, max_seq_len):
    """Encode one multi-turn conversation as an SFT training sample."""
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    prompt = render_chat_prompt(tokenizer, conversations)
    ids = tokenizer(prompt, add_special_tokens=False).input_ids[:max_seq_len]
    ids = ids + [pad_id] * (max_seq_len - len(ids))
    labels = generate_sft_labels(tokenizer, ids, max_seq_len)
    return np.array(ids, dtype=np.int32), np.array(labels, dtype=np.int32)


def render_chat_prompt(tokenizer, conversations):
    """Render the chat format MiniMind expects, by hand.

    ``tokenizer.apply_chat_template`` is avoided because a few records in the
    SFT JSONL make the Jinja template raise. Building the string manually is
    more robust and guarantees training and inference use the same format.
    """
    im_start = tokenizer.bos_token or "<|im_start|>"
    im_end = tokenizer.eos_token or "<|im_end|>"
    chunks = []
    for message in conversations:
        role = message.get("role") or message.get("from")
        content = message.get("content")
        if content is None:
            content = message.get("value", "")
        # Upstream JSONL mixes English and Chinese role aliases; normalise both.
        if role in ("human", "用户"):
            role = "user"
        elif role in ("gpt", "bot", "模型"):
            role = "assistant"
        if role not in ("system", "user", "assistant"):
            continue
        chunks.append(f"{im_start}{role}\n{str(content)}{im_end}\n")
    return "".join(chunks)


def generate_sft_labels(tokenizer, input_ids, max_seq_len):
    """Build SFT labels that supervise only the assistant turns.

    user, system and pad positions are set to -100, so the model learns how to
    answer instead of learning to reproduce the prompt.
    """
    labels = [-100] * len(input_ids)
    bos_id = tokenizer(f"{tokenizer.bos_token}assistant\n", add_special_tokens=False).input_ids
    eos_id = tokenizer(f"{tokenizer.eos_token}\n", add_special_tokens=False).input_ids
    i = 0
    while i < len(input_ids):
        if input_ids[i:i + len(bos_id)] == bos_id:
            start = i + len(bos_id)
            end = start
            while end < len(input_ids):
                if input_ids[end:end + len(eos_id)] == eos_id:
                    break
                end += 1
            for j in range(start, min(end + len(eos_id), max_seq_len)):
                if input_ids[j] != (tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0):
                    labels[j] = input_ids[j]
            i = end + len(eos_id) if end < len(input_ids) else len(input_ids)
        else:
            i += 1
    return labels


class PretrainJsonlDataset:
    """Reader for pretrain_t2t_mini.jsonl / pretrain_t2t.jsonl."""
    def __init__(self, data_path, tokenizer, max_seq_len=128, max_samples=256):
        self.samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if max_samples > 0 and len(self.samples) >= max_samples:
                    break
                if not line.strip():
                    continue
                obj = json.loads(line)
                self.samples.append(encode_pretrain_sample(tokenizer, obj.get("text", ""), max_seq_len))
        if not self.samples:
            raise RuntimeError(f"No samples loaded from {data_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class SFTJsonlDataset:
    """Reader for the sft_t2t_mini.jsonl / sft_t2t.jsonl instruction data."""
    def __init__(self, data_path, tokenizer, max_seq_len=128, max_samples=256):
        self.samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if max_samples > 0 and len(self.samples) >= max_samples:
                    break
                if not line.strip():
                    continue
                obj = json.loads(line)
                conversations = obj.get("conversations", [])
                if conversations:
                    self.samples.append(encode_chat_sample(tokenizer, conversations, max_seq_len))
        if not self.samples:
            raise RuntimeError(f"No SFT samples loaded from {data_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class DPOJsonlDataset:
    """Reader for DPO preference data.

    Each record pairs a chosen and a rejected response to the same prompt.
    """
    def __init__(self, data_path, tokenizer, max_seq_len=128, max_samples=256):
        self.samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if max_samples > 0 and len(self.samples) >= max_samples:
                    break
                if not line.strip():
                    continue
                obj = json.loads(line)
                chosen = obj.get("chosen")
                rejected = obj.get("rejected")
                if chosen and rejected:
                    self.samples.append(
                        encode_chat_sample(tokenizer, chosen, max_seq_len)
                        + encode_chat_sample(tokenizer, rejected, max_seq_len)
                    )
        if not self.samples:
            raise RuntimeError(f"No DPO samples loaded from {data_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def _extract_conversations(obj):
    """Best-effort extraction of a conversation from the various RL layouts."""
    for key in ("conversations", "messages", "chosen", "trajectory"):
        value = obj.get(key)
        if isinstance(value, list) and value:
            return value
    prompt = obj.get("prompt") or obj.get("question") or obj.get("input")
    answer = obj.get("answer") or obj.get("response") or obj.get("output")
    if prompt and answer:
        return [{"role": "user", "content": str(prompt)}, {"role": "assistant", "content": str(answer)}]
    text = obj.get("text")
    if text:
        return [{"role": "user", "content": str(text)}]
    return []


def _extract_reward(obj, default=1.0):
    """Read reward/score off a record, falling back to a default value."""
    for key in ("reward", "score", "value", "accuracy"):
        if key in obj:
            try:
                return float(obj[key])
            except Exception:
                return default
    return default


class RewardJsonlDataset:
    """Reader for the reward-annotated samples used by the simplified PPO/GRPO paths."""
    def __init__(self, data_path, tokenizer, max_seq_len=128, max_samples=256):
        self.samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if max_samples > 0 and len(self.samples) >= max_samples:
                    break
                if not line.strip():
                    continue
                obj = json.loads(line)
                conversations = _extract_conversations(obj)
                if not conversations:
                    continue
                input_ids, labels = encode_chat_sample(tokenizer, conversations, max_seq_len)
                reward = np.array(_extract_reward(obj), dtype=np.float32)
                self.samples.append((input_ids, labels, reward))
        if not self.samples:
            raise RuntimeError(f"No reward samples loaded from {data_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def make_lm_dataset(dataset, batch_size, shuffle=True):
    """Wrap a Python dataset as a batched, iterable MindSpore dataset."""
    return GeneratorDataset(dataset, column_names=["input_ids", "labels"], shuffle=shuffle).batch(
        batch_size, drop_remainder=True
    )


def make_dpo_dataset(dataset, batch_size, shuffle=True):
    """The DPO dataset yields chosen and rejected ids/labels side by side."""
    columns = ["chosen_ids", "chosen_labels", "rejected_ids", "rejected_labels"]
    return GeneratorDataset(dataset, column_names=columns, shuffle=shuffle).batch(batch_size, drop_remainder=True)


def make_reward_dataset(dataset, batch_size, shuffle=True):
    """The simplified PPO/GRPO dataset yields input ids, labels and reward."""
    return GeneratorDataset(dataset, column_names=["input_ids", "labels", "rewards"], shuffle=shuffle).batch(
        batch_size, drop_remainder=True
    )


class GradientAccumulator:
    """Accumulate several micro-batches of gradients into one optimizer step.

    This is the training-time workaround for a small card: the micro-batch has to
    stay small enough to fit, but the effective batch has to be large enough for a
    stable gradient. Under PyTorch you would divide the loss by ``accum_steps``
    and call ``backward()`` repeatedly; MindSpore has no ``backward()``, so
    ``ms.value_and_grad`` hands back a *tuple* of gradients per micro-batch. The
    equivalent is to sum those tuples and divide once, when the window closes.

    Dividing once rather than per micro-batch keeps the update identical to a
    single pass over a batch of ``batch_size * accum_steps`` samples, so the
    learning rate does not have to be retuned when the accumulation factor
    changes.
    """

    def __init__(self, accum_steps=1):
        self.accum_steps = max(1, int(accum_steps))
        self._acc = None
        self._count = 0

    def add(self, grads):
        """Add one micro-batch of gradients; return True when an update is due."""
        grads = tuple(grads)
        if self._acc is None:
            self._acc = list(grads)
        else:
            self._acc = [ops.add(acc, g) for acc, g in zip(self._acc, grads)]
        self._count += 1
        return self._count >= self.accum_steps

    def take(self):
        """Return the mean gradient tuple for the window and reset the window."""
        if not self._acc:
            raise RuntimeError("take() called on an empty gradient window")
        scale = 1.0 / float(self.accum_steps)
        grads = tuple(g * scale for g in self._acc)
        self._acc = None
        self._count = 0
        return grads

    @property
    def pending(self):
        """Number of micro-batches accumulated in the current window."""
        return self._count


class StepTimer:
    """Small wall-clock timer used to report training time."""
    def __init__(self):
        self.start = time.time()

    def elapsed(self):
        return time.time() - self.start

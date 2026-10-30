import argparse
import os

import mindspore as ms
import mindspore.ops as ops
import numpy as np
from mindspore import Tensor
from transformers import AutoTokenizer

from ms_common import resolve_config_for_ckpt
from ms_minimind import MiniMindForCausalLMMS, MiniMindMSConfig


def build_generation_prompt(tokenizer, prompt):
    """Build the single-turn chat prompt the SFT stage was trained on."""
    im_start = tokenizer.bos_token or "<|im_start|>"
    im_end = tokenizer.eos_token or "<|im_end|>"
    return f"{im_start}user\n{prompt}{im_end}\n{im_start}assistant\n"


def greedy_generate(model, input_ids, max_new_tokens, eos_id, max_seq_len, min_new_tokens=8):
    """Greedy decoding.

    Always takes the arg-max token. min_new_tokens temporarily suppresses EOS so
    that a lightly trained experimental model still produces something
    observable instead of stopping on the first step.
    """
    ids = list(input_ids)
    for step in range(max_new_tokens):
        window = ids[-max_seq_len:]
        x = Tensor(np.array([window], dtype=np.int32))
        logits = model(x)
        next_logits = logits[:, -1, :]
        next_id = int(ops.Argmax(axis=-1)(next_logits).asnumpy()[0])
        # If the model wants EOS too early, mask it for the first few steps.
        if eos_id is not None and next_id == eos_id and step < min_new_tokens:
            next_logits_np = next_logits.asnumpy()[0]
            next_logits_np[eos_id] = -1e9
            next_id = int(np.argmax(next_logits_np))
        ids.append(next_id)
        if eos_id is not None and next_id == eos_id:
            break
    return ids


def main():
    """Single-shot generation entry point.

    Example:
    python generate.py --ckpt ./checkpoints/sft_ms.ckpt --prompt "What is a cache?"
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", default="/root/minimind-master")
    parser.add_argument("--device", default="GPU", choices=["GPU", "CPU"])
    parser.add_argument("--ckpt", default="./checkpoints/sft_ms.ckpt")
    parser.add_argument("--prompt", default="hello")
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--min_new_tokens", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    args = parser.parse_args()

    ms.set_context(mode=ms.PYNATIVE_MODE, device_target=args.device)

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.source_dir, "model"))
    fallback_config = MiniMindMSConfig(
        vocab_size=len(tokenizer),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        max_seq_len=args.max_seq_len,
        intermediate_size=args.hidden_size * 4,
    )
    config = resolve_config_for_ckpt(args.ckpt, fallback_config)
    model = MiniMindForCausalLMMS(config)
    # The JSON sidecar restores the trained shape, so the weights always fit.
    ms.load_checkpoint(args.ckpt, net=model)
    model.set_train(False)

    text = build_generation_prompt(tokenizer, args.prompt)
    input_ids = tokenizer(text, add_special_tokens=False).input_ids
    output_ids = greedy_generate(
        model=model,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        eos_id=tokenizer.eos_token_id,
        max_seq_len=args.max_seq_len,
        min_new_tokens=args.min_new_tokens,
    )
    new_ids = output_ids[len(input_ids):]
    answer = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    if answer:
        print(answer)
    else:
        print("[empty output: model generated only special/end tokens]")
        print("token ids:", new_ids)
        print("raw:", tokenizer.decode(new_ids, skip_special_tokens=False))


if __name__ == "__main__":
    main()

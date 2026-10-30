import argparse
import os

import mindspore as ms
from transformers import AutoTokenizer

from generate import build_generation_prompt, greedy_generate
from ms_common import resolve_config_for_ckpt
from ms_minimind import MiniMindMSConfig, MiniMindForCausalLMMS


def main():
    """Interactive chat entry point.

    Unlike generate.py this loops on stdin until an empty line is entered.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", default="/root/minimind-master")
    parser.add_argument("--device", default="GPU", choices=["GPU", "CPU"])
    parser.add_argument("--ckpt", default="./checkpoints/sft_ms.ckpt")
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    args = parser.parse_args()

    ms.set_context(mode=ms.PYNATIVE_MODE, device_target=args.device)
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.source_dir, "model"))
    fallback = MiniMindMSConfig(
        vocab_size=len(tokenizer),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        max_seq_len=args.max_seq_len,
        intermediate_size=args.hidden_size * 4,
    )
    config = resolve_config_for_ckpt(args.ckpt, fallback)
    model = MiniMindForCausalLMMS(config)
    ms.load_checkpoint(args.ckpt, net=model)
    model.set_train(False)

    print("MiniMind MindSpore chat. Empty input exits.")
    while True:
        prompt = input("user> ").strip()
        if not prompt:
            break
        # The prompt format must match SFT training or the model emits junk.
        text = build_generation_prompt(tokenizer, prompt)
        input_ids = tokenizer(text, add_special_tokens=False).input_ids
        output_ids = greedy_generate(model, input_ids, args.max_new_tokens, tokenizer.eos_token_id, config.max_seq_len)
        answer = tokenizer.decode(output_ids[len(input_ids):], skip_special_tokens=True).strip()
        if answer:
            print("assistant>", answer)
        else:
            print("assistant> [empty output: model generated only special/end tokens]")


if __name__ == "__main__":
    main()

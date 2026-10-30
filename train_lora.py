import argparse
import os

import mindspore as ms
from mindspore import nn

from ms_common import SFTJsonlDataset, StepTimer, load_tokenizer, make_config, make_lm_dataset, save_model_with_config
from ms_minimind import (
    CausalLMLoss,
    MiniMindForCausalLMMS,
    apply_lora_to_model,
)
from train_smoke import train_step_fn


def main():
    """LoRA fine-tuning entry point.

    Only the newly added low-rank matrices are trained, which suits a card with
    little free memory or fast task adaptation. By default it continues from the
    SFT checkpoint.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", default="/root/minimind-master")
    parser.add_argument("--device", default="GPU", choices=["GPU", "CPU"])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_kv_heads", type=int, default=2)
    parser.add_argument("--use_moe", type=int, default=0, choices=[0, 1])
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--init_ckpt", default="./checkpoints/sft_ms.ckpt")
    parser.add_argument("--data_path", default="")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--save_dir", default="./checkpoints")
    parser.add_argument("--save_name", default="lora_ms.ckpt")
    parser.add_argument("--save_every", type=int, default=100)
    args = parser.parse_args()
    args.grad_accum = max(1, args.grad_accum)

    ms.set_context(mode=ms.PYNATIVE_MODE, device_target=args.device)
    tokenizer = load_tokenizer(args.source_dir)
    data_path = args.data_path or os.path.join(args.source_dir, "dataset", "sft_t2t_mini.jsonl")
    dataset = SFTJsonlDataset(data_path, tokenizer, args.max_seq_len, args.max_samples)
    ms_dataset = make_lm_dataset(dataset, args.batch_size, shuffle=True)

    config = make_config(args, tokenizer)
    model = MiniMindForCausalLMMS(config)
    if args.init_ckpt and os.path.exists(args.init_ckpt):
        ms.load_checkpoint(args.init_ckpt, net=model)
        print(f"loaded base checkpoint: {args.init_ckpt}")
    # Replace selected Dense layers with LoRA wrappers and freeze the base weights.
    model = apply_lora_to_model(model, rank=args.lora_rank, alpha=args.lora_alpha)
    trainable = [p for p in model.trainable_params() if p.requires_grad]
    print("trainable LoRA params:", sum(int(p.size) for p in trainable))
    print(
        "micro-batch %d x grad_accum %d = effective batch %d"
        % (args.batch_size, args.grad_accum, args.batch_size * args.grad_accum)
    )

    loss_net = CausalLMLoss(model)
    optimizer = nn.AdamWeightDecay(trainable, learning_rate=args.lr)
    train_step = train_step_fn(loss_net, optimizer, accum_steps=args.grad_accum)
    timer = StepTimer()

    step = 0
    window_loss = 0.0
    while step < args.steps:
        for input_ids, labels in ms_dataset.create_tuple_iterator():
            loss, updated = train_step(input_ids, labels)
            window_loss += float(loss.asnumpy())
            if not updated:
                continue
            step += 1
            print(f"lora step {step:04d} loss {window_loss / args.grad_accum:.4f}")
            window_loss = 0.0
            if step % args.save_every == 0:
                save_model_with_config(model, config, args.save_dir, args.save_name)
            if step >= args.steps:
                break

    save_model_with_config(model, config, args.save_dir, args.save_name)
    print(f"elapsed seconds: {timer.elapsed():.1f}")
    print("MiniMind MindSpore LoRA training OK")


if __name__ == "__main__":
    main()

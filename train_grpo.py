import argparse
import os

import mindspore as ms
from mindspore import nn

from ms_common import RewardJsonlDataset, StepTimer, load_tokenizer, make_config, make_reward_dataset, save_model_with_config
from ms_minimind import MiniMindForCausalLMMS, WeightedCausalLMLoss
from train_ppo import train_step_fn


def main():
    """Simplified GRPO entry point.

    NOT a full GRPO implementation: there is no grouped rollout and no
    group-normalized advantage. agent_rl_math.jsonl is read as reward-annotated
    samples and updated with the same reward-weighted objective as train_ppo.py.
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
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--init_ckpt", default="./checkpoints/sft_ms.ckpt")
    parser.add_argument("--data_path", default="")
    parser.add_argument("--save_dir", default="./checkpoints")
    parser.add_argument("--save_name", default="grpo_ms.ckpt")
    parser.add_argument("--save_every", type=int, default=100)
    args = parser.parse_args()
    args.grad_accum = max(1, args.grad_accum)

    ms.set_context(mode=ms.PYNATIVE_MODE, device_target=args.device)
    tokenizer = load_tokenizer(args.source_dir)
    data_path = args.data_path or os.path.join(args.source_dir, "dataset", "agent_rl_math.jsonl")
    dataset = RewardJsonlDataset(data_path, tokenizer, args.max_seq_len, args.max_samples)
    ms_dataset = make_reward_dataset(dataset, args.batch_size, shuffle=True)

    config = make_config(args, tokenizer)
    model = MiniMindForCausalLMMS(config)
    if args.init_ckpt and os.path.exists(args.init_ckpt):
        ms.load_checkpoint(args.init_ckpt, net=model)
        print(f"loaded checkpoint: {args.init_ckpt}")

    loss_net = WeightedCausalLMLoss(model)
    optimizer = nn.AdamWeightDecay(model.trainable_params(), learning_rate=args.lr)
    train_step = train_step_fn(loss_net, optimizer, accum_steps=args.grad_accum)
    timer = StepTimer()

    print("GRPO simplified migration: group-reward data is consumed as reward-weighted updates.")
    print(
        "micro-batch %d x grad_accum %d = effective batch %d"
        % (args.batch_size, args.grad_accum, args.batch_size * args.grad_accum)
    )
    step = 0
    window_loss = 0.0
    while step < args.steps:
        for input_ids, labels, rewards in ms_dataset.create_tuple_iterator():
            loss, updated = train_step(input_ids, labels, rewards)
            window_loss += float(loss.asnumpy())
            if not updated:
                continue
            step += 1
            print(f"grpo step {step:04d} loss {window_loss / args.grad_accum:.4f}")
            window_loss = 0.0
            if step % args.save_every == 0:
                save_model_with_config(model, config, args.save_dir, args.save_name)
            if step >= args.steps:
                break
    save_model_with_config(model, config, args.save_dir, args.save_name)
    print(f"elapsed seconds: {timer.elapsed():.1f}")
    print("MiniMind MindSpore simplified GRPO training OK")


if __name__ == "__main__":
    main()

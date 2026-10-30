import argparse
import os

import mindspore as ms
from mindspore import nn

from ms_common import (
    GradientAccumulator,
    RewardJsonlDataset,
    StepTimer,
    load_tokenizer,
    make_config,
    make_reward_dataset,
    save_model_with_config,
)
from ms_minimind import MiniMindForCausalLMMS, WeightedCausalLMLoss


def train_step_fn(loss_net, optimizer, accum_steps=1):
    """Single training step for the simplified PPO path.

    Returns a closure yielding ``(loss, updated)``; see ``train_smoke.py`` for
    the accumulation rationale.
    """
    grad_fn = ms.value_and_grad(loss_net, None, optimizer.parameters)
    accum = GradientAccumulator(accum_steps)

    def train_step(input_ids, labels, rewards):
        loss, grads = grad_fn(input_ids, labels, rewards)
        updated = accum.add(grads)
        if updated:
            optimizer(accum.take())
        return loss, updated

    return train_step


def main():
    """Simplified PPO entry point.

    NOT a full PPO implementation: there is no rollout engine, no clipped
    policy ratio, no advantage estimation and no value model. The reward stored
    in rlaif.jsonl is reused as a sample weight, which is enough to exercise
    the MindSpore training path but is not a policy-gradient estimator.
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
    parser.add_argument("--save_name", default="ppo_ms.ckpt")
    parser.add_argument("--save_every", type=int, default=100)
    args = parser.parse_args()
    args.grad_accum = max(1, args.grad_accum)

    ms.set_context(mode=ms.PYNATIVE_MODE, device_target=args.device)
    tokenizer = load_tokenizer(args.source_dir)
    data_path = args.data_path or os.path.join(args.source_dir, "dataset", "rlaif.jsonl")
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

    print("PPO simplified migration: reward-weighted policy update over RLAIF samples.")
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
            print(f"ppo step {step:04d} loss {window_loss / args.grad_accum:.4f}")
            window_loss = 0.0
            if step % args.save_every == 0:
                save_model_with_config(model, config, args.save_dir, args.save_name)
            if step >= args.steps:
                break
    save_model_with_config(model, config, args.save_dir, args.save_name)
    print(f"elapsed seconds: {timer.elapsed():.1f}")
    print("MiniMind MindSpore simplified PPO training OK")


if __name__ == "__main__":
    main()

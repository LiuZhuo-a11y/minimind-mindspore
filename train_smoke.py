import argparse
import os
import mindspore as ms
from mindspore import nn

from ms_minimind import MiniMindForCausalLMMS, CausalLMLoss
from ms_common import (
    GradientAccumulator,
    PretrainJsonlDataset,
    StepTimer,
    load_tokenizer,
    make_config,
    make_lm_dataset,
    save_model_with_config,
)


def train_step_fn(loss_net, optimizer, accum_steps=1):
    """Single-step trainer with optional gradient accumulation.

    Each call consumes one micro-batch. The returned closure yields
    ``(loss, updated)``, where ``updated`` is True on the micro-batch that
    closes an accumulation window -- that is, the one where the optimizer
    actually stepped.

    With ``accum_steps == 1`` this is the plain loss -> value_and_grad ->
    ``optimizer(grads)`` loop. The accumulation path only changes *when* the
    optimizer is called; the loss computation is untouched.
    """
    grad_fn = ms.value_and_grad(loss_net, None, optimizer.parameters)
    accum = GradientAccumulator(accum_steps)

    def train_step(input_ids, labels):
        loss, grads = grad_fn(input_ids, labels)
        updated = accum.add(grads)
        if updated:
            optimizer(accum.take())
        return loss, updated

    return train_step


def main():
    """Pre-training entry point.

    Reads pretrain_t2t_mini.jsonl by default and trains a MiniMind model from
    scratch. The "smoke" in the name comes from its second role: a short run
    that proves the MindSpore / GPU / data pipeline works end to end.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", default="/root/minimind-master")
    parser.add_argument("--device", default="GPU", choices=["GPU", "CPU"])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--max_seq_len", type=int, default=340)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_kv_heads", type=int, default=4)
    parser.add_argument("--use_moe", type=int, default=0, choices=[0, 1])
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--save_dir", default="./checkpoints")
    parser.add_argument("--save_name", default="pretrain_smoke.ckpt")
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--data_path", default="")
    args = parser.parse_args()
    args.grad_accum = max(1, args.grad_accum)

    ms.set_context(mode=ms.PYNATIVE_MODE, device_target=args.device)

    # Without an explicit --data_path, use the mini pre-training set from MiniMind.
    data_path = args.data_path or os.path.join(args.source_dir, "dataset", "pretrain_t2t_mini.jsonl")
    tokenizer = load_tokenizer(args.source_dir)

    dataset = PretrainJsonlDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        max_samples=args.max_samples,
    )
    ms_dataset = make_lm_dataset(dataset, args.batch_size, shuffle=True)

    config = make_config(args, tokenizer)
    model = MiniMindForCausalLMMS(config)
    loss_net = CausalLMLoss(model)
    optimizer = nn.AdamWeightDecay(model.trainable_params(), learning_rate=args.lr)
    train_step = train_step_fn(loss_net, optimizer, accum_steps=args.grad_accum)

    print("MindSpore version:", ms.__version__)
    print("Device:", args.device)
    print("Dataset samples:", len(dataset))
    print("Vocab size:", config.vocab_size)
    print(
        "micro-batch %d x grad_accum %d = effective batch %d"
        % (args.batch_size, args.grad_accum, args.batch_size * args.grad_accum)
    )
    os.makedirs(args.save_dir, exist_ok=True)
    timer = StepTimer()

    step = 0
    window_loss = 0.0
    # When the iterator is exhausted the outer loop re-enters it, so --steps is
    # always reached regardless of dataset size. --steps counts optimizer
    # updates, not micro-batches: with --grad_accum 4, four batches are consumed
    # per printed step.
    while step < args.steps:
        for input_ids, labels in ms_dataset.create_tuple_iterator():
            loss, updated = train_step(input_ids, labels)
            window_loss += float(loss.asnumpy())
            if not updated:
                continue
            step += 1
            print(f"step {step:04d} loss {window_loss / args.grad_accum:.4f}")
            window_loss = 0.0
            if step % args.save_every == 0:
                save_model_with_config(model, config, args.save_dir, args.save_name)
            if step >= args.steps:
                break

    save_model_with_config(model, config, args.save_dir, args.save_name)
    elapsed = timer.elapsed()
    print(f"elapsed seconds: {elapsed:.1f}")
    print("MiniMind MindSpore smoke training OK")


if __name__ == "__main__":
    main()

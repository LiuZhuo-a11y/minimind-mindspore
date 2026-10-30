import argparse
import os
import mindspore as ms
from mindspore import nn

from ms_minimind import MiniMindForCausalLMMS, CausalLMLoss
from train_smoke import train_step_fn
from ms_common import SFTJsonlDataset, StepTimer, load_tokenizer, make_config, make_lm_dataset, save_model_with_config


def main():
    """Supervised fine-tuning entry point.

    Typical flow: train_smoke.py produces pretrain_ms.ckpt, this script loads
    it and continues on sft_t2t_mini.jsonl so the model learns the Q&A format.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", default="/root/minimind-master")
    parser.add_argument("--device", default="GPU", choices=["GPU", "CPU"])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--max_seq_len", type=int, default=768)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_kv_heads", type=int, default=4)
    parser.add_argument("--use_moe", type=int, default=0, choices=[0, 1])
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--init_ckpt", default="./checkpoints/pretrain_ms.ckpt")
    parser.add_argument("--save_dir", default="./checkpoints")
    parser.add_argument("--save_name", default="sft_ms.ckpt")
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--data_path", default="")
    args = parser.parse_args()
    args.grad_accum = max(1, args.grad_accum)

    ms.set_context(mode=ms.PYNATIVE_MODE, device_target=args.device)

    # Without an explicit --data_path, use the mini SFT set from MiniMind.
    data_path = args.data_path or os.path.join(args.source_dir, "dataset", "sft_t2t_mini.jsonl")
    tokenizer = load_tokenizer(args.source_dir)

    dataset = SFTJsonlDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        max_samples=args.max_samples,
    )
    ms_dataset = make_lm_dataset(dataset, args.batch_size, shuffle=True)

    config = make_config(args, tokenizer)
    model = MiniMindForCausalLMMS(config)
    # SFT normally continues from the pre-training checkpoint; if it is missing
    # the run degrades to random initialization and says so.
    if args.init_ckpt and os.path.exists(args.init_ckpt):
        ms.load_checkpoint(args.init_ckpt, net=model)
        print(f"loaded checkpoint: {args.init_ckpt}")
    else:
        print("pretrain checkpoint not found, SFT starts from random weights")

    loss_net = CausalLMLoss(model)
    optimizer = nn.AdamWeightDecay(model.trainable_params(), learning_rate=args.lr)
    train_step = train_step_fn(loss_net, optimizer, accum_steps=args.grad_accum)

    print("MindSpore version:", ms.__version__)
    print("Device:", args.device)
    print("SFT samples:", len(dataset))
    print("Vocab size:", config.vocab_size)
    print(
        "micro-batch %d x grad_accum %d = effective batch %d"
        % (args.batch_size, args.grad_accum, args.batch_size * args.grad_accum)
    )
    os.makedirs(args.save_dir, exist_ok=True)
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
            print(f"sft step {step:04d} loss {window_loss / args.grad_accum:.4f}")
            window_loss = 0.0
            if step % args.save_every == 0:
                save_model_with_config(model, config, args.save_dir, args.save_name)
            if step >= args.steps:
                break

    save_model_with_config(model, config, args.save_dir, args.save_name)
    elapsed = timer.elapsed()
    print(f"elapsed seconds: {elapsed:.1f}")
    print("MiniMind MindSpore SFT training OK")


if __name__ == "__main__":
    main()

import argparse
import os

import mindspore as ms
from mindspore import Tensor, nn
import mindspore.ops as ops

from ms_common import (
    DPOJsonlDataset,
    GradientAccumulator,
    StepTimer,
    load_tokenizer,
    make_config,
    make_dpo_dataset,
    save_model_with_config,
)
from ms_minimind import MiniMindForCausalLMMS


class DPOLossNet(nn.Cell):
    """Direct Preference Optimization loss.

    policy_model is optimized, ref_model is the frozen reference. The objective
    pushes chosen above rejected in log-probability, with beta controlling how
    tightly the policy stays tied to the reference.
    """
    def __init__(self, policy_model, ref_model, beta=0.1):
        super().__init__()
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.beta = beta
        self.log_softmax = nn.LogSoftmax(axis=-1)

    def sequence_log_prob(self, model, input_ids, labels):
        """Sum the per-token log probabilities of one response."""
        logits = model(input_ids)
        log_probs = self.log_softmax(logits)
        vocab = logits.shape[-1]
        shift_log_probs = log_probs[:, :-1, :]
        shift_labels = labels[:, 1:]
        flat_labels = ops.reshape(shift_labels, (-1,))
        valid = ops.cast(flat_labels != -100, ms.float32)
        safe_labels = ops.maximum(flat_labels, Tensor(0, ms.int32))
        one_hot = ops.one_hot(safe_labels, vocab, Tensor(1.0, ms.float32), Tensor(0.0, ms.float32))
        token_log_probs = ops.sum(ops.reshape(shift_log_probs, (-1, vocab)) * one_hot, axis=-1)
        token_log_probs = ops.reshape(token_log_probs, shift_labels.shape)
        valid_2d = ops.reshape(valid, shift_labels.shape)
        return ops.sum(token_log_probs * valid_2d, axis=1)

    def construct(self, chosen_ids, chosen_labels, rejected_ids, rejected_labels):
        # DPO compares the chosen/rejected log-prob margin under policy and
        # reference; only the policy term carries gradient.
        chosen_policy = self.sequence_log_prob(self.policy_model, chosen_ids, chosen_labels)
        rejected_policy = self.sequence_log_prob(self.policy_model, rejected_ids, rejected_labels)
        chosen_ref = ops.stop_gradient(self.sequence_log_prob(self.ref_model, chosen_ids, chosen_labels))
        rejected_ref = ops.stop_gradient(self.sequence_log_prob(self.ref_model, rejected_ids, rejected_labels))
        logits = (chosen_policy - rejected_policy) - (chosen_ref - rejected_ref)
        return -ops.mean(ops.log_sigmoid(self.beta * logits))


def train_step_fn(loss_net, optimizer, accum_steps=1):
    """Single training step for DPO, with optional gradient accumulation.

    DPO is the most memory-hungry path here: both the policy and the reference
    are evaluated on the chosen and the rejected sequence, so ``batch_size 1``
    is already the default and accumulation is how a usable effective batch is
    reached.
    """
    grad_fn = ms.value_and_grad(loss_net, None, optimizer.parameters)
    accum = GradientAccumulator(accum_steps)

    def train_step(chosen_ids, chosen_labels, rejected_ids, rejected_labels):
        loss, grads = grad_fn(chosen_ids, chosen_labels, rejected_ids, rejected_labels)
        updated = accum.add(grads)
        if updated:
            optimizer(accum.take())
        return loss, updated

    return train_step


def main():
    """DPO training entry point.

    Uses the preference pairs in dpo.jsonl to shift an SFT model toward the
    preferred responses.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", default="/root/minimind-master")
    parser.add_argument("--device", default="GPU", choices=["GPU", "CPU"])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=1)
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
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--init_ckpt", default="./checkpoints/sft_ms.ckpt")
    parser.add_argument("--data_path", default="")
    parser.add_argument("--save_dir", default="./checkpoints")
    parser.add_argument("--save_name", default="dpo_ms.ckpt")
    parser.add_argument("--save_every", type=int, default=100)
    args = parser.parse_args()
    args.grad_accum = max(1, args.grad_accum)

    ms.set_context(mode=ms.PYNATIVE_MODE, device_target=args.device)
    tokenizer = load_tokenizer(args.source_dir)
    data_path = args.data_path or os.path.join(args.source_dir, "dataset", "dpo.jsonl")
    dataset = DPOJsonlDataset(data_path, tokenizer, args.max_seq_len, args.max_samples)
    ms_dataset = make_dpo_dataset(dataset, args.batch_size, shuffle=True)

    config = make_config(args, tokenizer)
    policy = MiniMindForCausalLMMS(config)
    ref = MiniMindForCausalLMMS(config)
    # Only the policy is optimized; the reference model is kept as a baseline.
    if args.init_ckpt and os.path.exists(args.init_ckpt):
        ms.load_checkpoint(args.init_ckpt, net=policy)
        ms.load_checkpoint(args.init_ckpt, net=ref)
        print(f"loaded checkpoint: {args.init_ckpt}")

    loss_net = DPOLossNet(policy, ref, beta=args.beta)
    optimizer = nn.AdamWeightDecay(policy.trainable_params(), learning_rate=args.lr)
    train_step = train_step_fn(loss_net, optimizer, accum_steps=args.grad_accum)
    timer = StepTimer()

    print(
        "micro-batch %d x grad_accum %d = effective batch %d"
        % (args.batch_size, args.grad_accum, args.batch_size * args.grad_accum)
    )
    step = 0
    window_loss = 0.0
    while step < args.steps:
        for batch in ms_dataset.create_tuple_iterator():
            loss, updated = train_step(*batch)
            window_loss += float(loss.asnumpy())
            if not updated:
                continue
            step += 1
            print(f"dpo step {step:04d} loss {window_loss / args.grad_accum:.4f}")
            window_loss = 0.0
            if step % args.save_every == 0:
                save_model_with_config(policy, config, args.save_dir, args.save_name)
            if step >= args.steps:
                break

    save_model_with_config(policy, config, args.save_dir, args.save_name)
    print(f"elapsed seconds: {timer.elapsed():.1f}")
    print("MiniMind MindSpore DPO training OK")


if __name__ == "__main__":
    main()

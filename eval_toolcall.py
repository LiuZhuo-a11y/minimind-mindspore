import argparse
import json
import math
import os
import random
import re
from datetime import datetime

import mindspore as ms
from transformers import AutoTokenizer

from generate import greedy_generate
from ms_common import resolve_config_for_ckpt
from ms_minimind import MiniMindMSConfig, MiniMindForCausalLMMS


TOOLS = [
    # Tool schemas shown to the model. A real deployment would swap these for
    # database or web-search tools.
    {
        "type": "function",
        "function": {
            "name": "calculate_math",
            "description": "Calculate a math expression.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get current date and time.",
            "parameters": {"type": "object", "properties": {"timezone": {"type": "string"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "random_number",
            "description": "Generate a random integer.",
            "parameters": {
                "type": "object",
                "properties": {"min": {"type": "integer"}, "max": {"type": "integer"}},
            },
        },
    },
]


def safe_calculate(expression):
    """Evaluate a math expression in a restricted namespace, not a full eval."""
    allowed = {"sqrt": math.sqrt, "pow": pow, "abs": abs, "round": round}
    return eval(expression.replace("^", "**"), {"__builtins__": {}}, allowed)


def execute_tool(name, args):
    """Dispatch the tool name emitted by the model to a local Python function."""
    if name == "calculate_math":
        return {"result": safe_calculate(str(args.get("expression", "0")))}
    if name == "get_current_time":
        return {"datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "timezone": args.get("timezone", "Asia/Shanghai")}
    if name == "random_number":
        return {"result": random.randint(int(args.get("min", 0)), int(args.get("max", 100)))}
    return {"error": f"unknown tool: {name}"}


def parse_tool_calls(text):
    """Parse <tool_call>...</tool_call> JSON blocks out of the model response."""
    calls = []
    for match in re.findall(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL):
        try:
            data = json.loads(match.strip())
            calls.append(data)
        except json.JSONDecodeError:
            pass
    return calls


def main():
    """Tool-call demo entry point.

    Checks whether the migrated model emits the tool-call format at all, then
    executes the requested mock tool in Python. With the current training budget
    this is a pipeline demo, not a capability benchmark.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", default="/root/minimind-master")
    parser.add_argument("--device", default="GPU", choices=["GPU", "CPU"])
    parser.add_argument("--ckpt", default="./checkpoints/sft_ms.ckpt")
    parser.add_argument("--prompt", default="Use the calculate_math tool to compute 23*37")
    parser.add_argument("--max_new_tokens", type=int, default=160)
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

    messages = [{"role": "user", "content": args.prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=TOOLS)
    input_ids = tokenizer(text, add_special_tokens=False).input_ids
    output_ids = greedy_generate(model, input_ids, args.max_new_tokens, tokenizer.eos_token_id, config.max_seq_len)
    response = tokenizer.decode(output_ids[len(input_ids):], skip_special_tokens=True)
    print("assistant:", response)
    for call in parse_tool_calls(response):
        name = call.get("name", "")
        tool_args = call.get("arguments", {})
        print("tool_call:", name, tool_args)
        print("tool_result:", json.dumps(execute_tool(name, tool_args), ensure_ascii=False))


if __name__ == "__main__":
    main()

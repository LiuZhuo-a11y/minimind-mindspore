"""Minimal web demo for the MindSpore port: one HTML page, one JSON endpoint.

Deliberately dependency-free -- `http.server` from the standard library instead of
Flask or Gradio -- so that serving the model needs no extra install and the whole
surface fits in one readable file. The only thing that matters here is that a
visitor can type a prompt and see what the model actually produces, latency
included.

    python serve.py --source_dir /path/to/minimind --ckpt ./checkpoints/sft_ms.ckpt

Notes for anyone exposing this beyond localhost:

* There is no authentication. Bind to 127.0.0.1 (the default) unless you know
  who can reach the port; ``--host 0.0.0.0`` prints a warning for that reason.
* The model is not thread-safe, so generation is serialised behind a lock: the
  server accepts concurrent connections but answers one generation at a time.
  A request queue is the honest description of this demo's capacity.
* Greedy decoding with no KV cache means every token is a full forward pass over
  the prefix. The page reports tokens/second rather than hiding it.
"""
import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import mindspore as ms
from transformers import AutoTokenizer

from generate import build_generation_prompt, greedy_generate
from ms_common import resolve_config_for_ckpt
from ms_minimind import MiniMindMSConfig, MiniMindForCausalLMMS

MAX_PROMPT_CHARS = 2000
MAX_BODY_BYTES = 64 * 1024

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MiniMind / MindSpore</title>
<style>
  :root { color-scheme: light dark; }
  body { font: 15px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
         max-width: 820px; margin: 0 auto; padding: 32px 20px 80px; }
  h1 { font-size: 20px; margin: 0 0 4px; }
  .sub { margin: 0 0 24px; color: #6b7280; font-size: 13px; }
  .row { display: flex; gap: 8px; align-items: stretch; }
  textarea { flex: 1; min-height: 64px; resize: vertical; padding: 10px 12px;
             border: 1px solid #d1d5db; border-radius: 8px; font: inherit;
             background: transparent; color: inherit; }
  button { padding: 0 18px; border-radius: 8px; border: 1px solid #2563eb;
           background: #2563eb; color: #fff; font: inherit; cursor: pointer; }
  button:disabled { opacity: .5; cursor: progress; }
  #meta { margin: 10px 0 0; font-size: 12px; color: #6b7280; min-height: 16px; }
  pre#out { margin: 20px 0 0; padding: 14px 16px; border: 1px solid #e5e7eb;
            border-radius: 8px; white-space: pre-wrap; word-break: break-word;
            min-height: 96px; background: rgba(127,127,127,.06);
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; }
  .warn { color: #b45309; }
</style>
</head>
<body>
<h1>MiniMind / MindSpore</h1>
<p class="sub">Greedy decoding, no KV cache: each new token is a full forward pass,
so expect a few tokens per second rather than a chat-speed response.</p>
<div class="row">
  <textarea id="q" placeholder="Ask something…  (Ctrl+Enter sends)"></textarea>
  <button id="go">Send</button>
</div>
<p id="meta"></p>
<pre id="out">Output appears here.</pre>
<script>
const q = document.getElementById('q');
const go = document.getElementById('go');
const out = document.getElementById('out');
const meta = document.getElementById('meta');

async function send() {
  const prompt = q.value.trim();
  if (!prompt || go.disabled) return;
  go.disabled = true;
  meta.className = '';
  meta.textContent = 'generating…';
  out.textContent = '';
  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({prompt})
    });
    const data = await res.json();
    if (!res.ok) {
      meta.className = 'warn';
      meta.textContent = data.error || ('HTTP ' + res.status);
      return;
    }
    out.textContent = data.answer ||
      '[empty output: the model produced only special/end tokens]';
    meta.className = data.empty ? 'warn' : '';
    meta.textContent = data.new_tokens + ' tokens in ' + data.seconds.toFixed(2) +
      ' s (' + data.tokens_per_second.toFixed(2) + ' tok/s)';
  } catch (err) {
    meta.className = 'warn';
    meta.textContent = 'request failed: ' + err;
  } finally {
    go.disabled = false;
  }
}

go.addEventListener('click', send);
q.addEventListener('keydown', function (e) {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    send();
  }
});
</script>
</body>
</html>
"""


class ChatService:
    """Wraps one loaded model into a serialised ``prompt -> answer`` call."""

    def __init__(self, model, tokenizer, config, max_new_tokens):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.max_new_tokens = max_new_tokens
        # MindSpore cells are not safe to call from several threads at once; the
        # lock makes concurrent HTTP requests queue instead of corrupting state.
        self._lock = threading.Lock()

    def reply(self, prompt):
        text = build_generation_prompt(self.tokenizer, prompt)
        input_ids = self.tokenizer(text, add_special_tokens=False).input_ids
        started = time.perf_counter()
        with self._lock:
            output_ids = greedy_generate(
                self.model,
                input_ids,
                self.max_new_tokens,
                self.tokenizer.eos_token_id,
                self.config.max_seq_len,
            )
        elapsed = time.perf_counter() - started
        new_ids = output_ids[len(input_ids):]
        answer = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        # A 1 ms floor: below that the timing is measurement noise and the ratio
        # would be meaningless (or a division by zero on a coarse clock).
        scale = max(elapsed, 0.001)
        return {
            "answer": answer,
            "empty": not answer,
            "new_tokens": len(new_ids),
            "seconds": round(elapsed, 3),
            "tokens_per_second": round(len(new_ids) / scale, 1),
        }


_SERVICE = None


class ChatHandler(BaseHTTPRequestHandler):
    server_version = "MiniMindDemo/0.1"

    def _send(self, body, code, content_type):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload, code=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(body, code, "application/json; charset=utf-8")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(PAGE.encode("utf-8"), 200, "text/html; charset=utf-8")
        elif self.path == "/health":
            self._json({"status": "ok"})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path != "/api/chat":
            self._json({"error": "not found"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_BODY_BYTES:
            self._json({"error": "empty or oversized request body"}, 400)
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            self._json({"error": "body is not valid JSON"}, 400)
            return
        prompt = payload.get("prompt") if isinstance(payload, dict) else None
        if not isinstance(prompt, str) or not prompt.strip():
            self._json({"error": "expected a non-empty 'prompt' string"}, 400)
            return
        try:
            self._json(_SERVICE.reply(prompt.strip()[:MAX_PROMPT_CHARS]))
        except Exception as exc:  # a demo should answer with the error, not a traceback
            self._json({"error": "%s: %s" % (type(exc).__name__, exc)}, 500)

    def log_message(self, fmt, *args):
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))


def make_server(service, host, port):
    """Bind a server to ``service``; separated from main() so it can be tested."""
    global _SERVICE
    _SERVICE = service
    httpd = ThreadingHTTPServer((host, port), ChatHandler)
    httpd.daemon_threads = True
    return httpd


def main():
    """Load one checkpoint, then serve the page and the JSON endpoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", default="/root/minimind-master")
    parser.add_argument("--device", default="GPU", choices=["GPU", "CPU"])
    parser.add_argument("--ckpt", default="./checkpoints/sft_ms.ckpt")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    args = parser.parse_args()

    if not os.path.exists(args.ckpt):
        raise SystemExit("checkpoint not found: %s (run train_sft.py first)" % args.ckpt)

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

    service = ChatService(model, tokenizer, config, args.max_new_tokens)
    if args.host not in ("127.0.0.1", "localhost"):
        print("warning: %s is not localhost and this endpoint has no authentication" % args.host)
    try:
        httpd = make_server(service, args.host, args.port)
    except OSError as exc:
        raise SystemExit("cannot bind %s:%d -- %s" % (args.host, args.port, exc))
    print("MiniMind MindSpore demo on http://%s:%d  (Ctrl+C to stop)" % (args.host, args.port))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()

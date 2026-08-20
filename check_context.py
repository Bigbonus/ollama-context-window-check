#!/usr/bin/env python3
"""check_context.py - detect silent prompt truncation in Ollama.

Same test as check_context.sh, with no dependency on jq. Standard library only,
so it runs on Windows as shipped.

Sends the same needle-in-a-haystack prompt twice: once with no num_ctx (server
default), once with an explicit num_ctx. Compares prompt_eval_count and whether
the needle was found.

If run 1 evaluates far fewer prompt tokens than run 2, your prompts are being
truncated before the model sees them. The needle sits on the FIRST line, because
Ollama discards the front of an over-long prompt.

Usage:
    python check_context.py                          # defaults: qwen3:14b, 16384
    python check_context.py qwen3:8b                 # pick a model
    python check_context.py qwen3:8b 32768           # model + num_ctx
    OLLAMA_HOST=http://box:11434 python check_context.py
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

NEEDLE = "XJ7Q-4419-KTM"
FILLER_LINES = 600
FILLER = "Routine log entry. Nothing of interest was recorded during this interval."


def build_prompt() -> str:
    """Haystack with the needle on the first line."""
    return "ACCESS CODE: %s\n" % NEEDLE + "\n".join(FILLER for _ in range(FILLER_LINES))


def run(url: str, model: str, system: str, label: str, extra: dict) -> int | None:
    """One call. Returns prompt_eval_count, or None if the request was refused."""
    options = {"num_predict": 64, "temperature": 0}
    options.update(extra)
    body = {
        "model": model,
        "stream": False,
        "think": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": "What is the ACCESS CODE stated at the top "
                                        "of the log? Answer with the code only."},
        ],
        "options": options,
    }
    request = urllib.request.Request(
        url, json.dumps(body).encode("utf-8"), {"Content-Type": "application/json"})

    print("--- %s" % label)
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            status, data = response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        print("    HTTP status      : %d" % exc.code)
        print("    response         : %s" % exc.read().decode("utf-8", "replace")[:200])
        print("    verdict          : REJECTED (prompt exceeded the served window)\n")
        return None
    except urllib.error.URLError as exc:
        sys.exit("could not reach %s: %s" % (url, exc.reason))

    count = data.get("prompt_eval_count")
    answer = (data.get("message") or {}).get("content", "").replace("\n", " ")[:120]
    print("    HTTP status      : %d" % status)
    print("    prompt_eval_count: %s" % (count if count is not None else "n/a"))
    print("    answer           : %s" % (answer or "(empty)"))
    print("    needle found     : %s\n" % ("yes" if NEEDLE in answer else "NO"))
    return count


def main() -> int:
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen3:14b"
    num_ctx = int(sys.argv[2]) if len(sys.argv) > 2 else 16384
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    if not host.startswith("http"):
        host = "http://" + host
    url = host.rstrip("/") + "/api/chat"

    system = build_prompt()
    print("model : %s" % model)
    print("host  : %s" % host)
    print("needle: %s (first line of the system prompt)" % NEEDLE)
    print("system prompt: %d characters\n" % len(system))

    default = run(url, model, system, "run 1: server default (no num_ctx)", {})
    explicit = run(url, model, system,
                   "run 2: explicit num_ctx=%d" % num_ctx, {"num_ctx": num_ctx})

    print("===")
    if default is None and explicit is not None:
        print("Run 1 was rejected while run 2 succeeded with an explicit window.")
        print("Your server default is smaller than this prompt.")
    elif explicit is None:
        print("Run 2 did not return a usable response.")
        print("Try a smaller num_ctx - the KV cache may not fit in VRAM alongside "
              "the weights.")
    elif default is not None and default < explicit:
        print("TRUNCATION DETECTED.")
        print("  default  : %s prompt tokens evaluated" % default)
        print("  explicit : %s prompt tokens evaluated" % explicit)
        print()
        print("The server discarded part of your prompt and returned HTTP 200 anyway.")
        print("Truncation takes the FRONT, so system prompts and rules go first.")
        print()
        print("Fix, in increasing order of permanence:")
        print('  per request : "options": {"num_ctx": %d}' % num_ctx)
        print("  server-wide : OLLAMA_CONTEXT_LENGTH=%d ollama serve" % num_ctx)
        print("  per model   : PARAMETER num_ctx %d in a Modelfile" % num_ctx)
    else:
        print("No truncation at this prompt size (%s tokens evaluated in both runs)."
              % default)
        print("Raise FILLER_LINES at the top of this script and re-run, or check the")
        print("CONTEXT column in 'ollama ps'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

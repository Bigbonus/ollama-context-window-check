#!/usr/bin/env bash
#
# check_context.sh — detect silent prompt truncation in Ollama.
#
# Sends the same needle-in-a-haystack prompt twice: once with no num_ctx
# (server default), once with an explicit num_ctx. Compares prompt_eval_count
# and whether the needle was found.
#
# If run 1 evaluates far fewer prompt tokens than run 2, your prompts are being
# truncated before the model sees them. The needle sits on the FIRST line,
# because Ollama discards the front of an over-long prompt.
#
# The prompt is passed to jq and to curl through files, never as a command-line
# argument: it is ~44 KB, and Windows caps a command line at about 32 KB.
#
# Usage:
#   ./check_context.sh                          # defaults: qwen3:14b, 16384
#   ./check_context.sh qwen3:8b                 # pick a model
#   ./check_context.sh qwen3:8b 32768           # model + num_ctx
#   OLLAMA_HOST=http://box:11434 ./check_context.sh
#
# Requires: bash, curl, jq. If you don't have jq, use check_context.py instead —
# same test, standard library only.

set -uo pipefail

MODEL="${1:-qwen3:14b}"
NUM_CTX="${2:-16384}"
HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
case "$HOST" in http://*|https://*) ;; *) HOST="http://${HOST}" ;; esac
NEEDLE="XJ7Q-4419-KTM"
FILLER_LINES=600

command -v curl >/dev/null || { echo "curl is required" >&2; exit 1; }
command -v jq   >/dev/null || {
  echo "jq is required — or run check_context.py, which needs neither" >&2; exit 1; }

WORK=$(mktemp -d) || { echo "could not create a temp dir" >&2; exit 1; }
trap 'rm -rf "$WORK"' EXIT

# --- build the haystack, needle on the first line ---------------------------
{
  printf 'ACCESS CODE: %s\n' "$NEEDLE"
  for _ in $(seq 1 "$FILLER_LINES"); do
    printf 'Routine log entry. Nothing of interest was recorded during this interval.\n'
  done
} > "$WORK/haystack.txt"

USER_Q='What is the ACCESS CODE stated at the top of the log? Answer with the code only.'

LAST_COUNT=""   # set by run()
LAST_STATE=""   # ok | overflow | failed

# --- run <label> <extra options json fragment> ------------------------------
run() {
  local label="$1" extra="$2" status http count answer
  LAST_COUNT=""; LAST_STATE=""

  # --rawfile keeps the 44 KB haystack off the command line.
  jq -n \
    --arg model "$MODEL" \
    --rawfile system "$WORK/haystack.txt" \
    --arg user "$USER_Q" \
    --argjson opts "{\"num_predict\":64,\"temperature\":0${extra}}" \
    '{model:$model, stream:false, think:false,
      messages:[{role:"system",content:$system},{role:"user",content:$user}],
      options:$opts}' > "$WORK/body.json" || {
    echo "--- ${label}"
    echo "    could not build the request body with jq"
    echo "    verdict          : REQUEST FAILED"
    echo
    LAST_STATE="failed"; return; }

  # @file keeps it off the command line a second time.
  curl -sS -o "$WORK/resp.json" -w '%{http_code}' "${HOST}/api/chat" \
       -H 'Content-Type: application/json' \
       --data-binary @"$WORK/body.json" > "$WORK/status.txt" 2>"$WORK/curl.err"
  status=$(cat "$WORK/status.txt" 2>/dev/null)
  http=$(cat "$WORK/resp.json" 2>/dev/null)

  echo "--- ${label}"

  if [ -z "$status" ]; then
    echo "    could not reach ${HOST}: $(head -c 200 "$WORK/curl.err")"
    echo "    verdict          : REQUEST FAILED"
    echo
    LAST_STATE="failed"
    return
  fi

  echo "    HTTP status      : ${status}"

  if [ "$status" != "200" ]; then
    echo "    response         : $(printf '%s' "$http" | head -c 200)"
    # Only call it a context rejection if the server said so. Any other 400 is
    # a broken request, and reporting that as truncation would be the same
    # class of mistake this repository is about.
    if printf '%s' "$http" | grep -qi 'context size\|exceed_context_size\|context length'; then
      echo "    verdict          : REJECTED (prompt exceeded the served window)"
      LAST_STATE="overflow"
    else
      echo "    verdict          : REQUEST FAILED (not a context error — see above)"
      LAST_STATE="failed"
    fi
    echo
    return
  fi

  count=$(jq -r '.prompt_eval_count // empty' < "$WORK/resp.json")
  answer=$(jq -r '.message.content // ""' < "$WORK/resp.json" | tr '\n' ' ' | head -c 120)

  echo "    prompt_eval_count: ${count:-n/a}"
  echo "    answer           : ${answer:-(empty)}"
  if [[ "$answer" == *"$NEEDLE"* ]]; then
    echo "    needle found     : yes"
  else
    echo "    needle found     : NO"
  fi
  echo

  LAST_COUNT="$count"
  LAST_STATE="ok"
}

echo "model : ${MODEL}"
echo "host  : ${HOST}"
echo "needle: ${NEEDLE} (first line of the system prompt)"
echo "system prompt: $(wc -c < "$WORK/haystack.txt" | tr -d ' ') characters"
echo

run "run 1: server default (no num_ctx)" ""
COUNT_DEFAULT="$LAST_COUNT"; STATE_DEFAULT="$LAST_STATE"

run "run 2: explicit num_ctx=${NUM_CTX}" ",\"num_ctx\":${NUM_CTX}"
COUNT_EXPLICIT="$LAST_COUNT"; STATE_EXPLICIT="$LAST_STATE"

# --- verdict ----------------------------------------------------------------
echo "==="
if [ "$STATE_DEFAULT" = "failed" ] || [ "$STATE_EXPLICIT" = "failed" ]; then
  echo "A request failed for a reason unrelated to context size. Nothing is proved"
  echo "either way — read the response above before concluding anything."
elif [ "$STATE_DEFAULT" = "overflow" ] && [ "$STATE_EXPLICIT" = "ok" ]; then
  echo "Run 1 was rejected while run 2 succeeded with an explicit window."
  echo "Your server default is smaller than this prompt."
elif [ "$STATE_EXPLICIT" != "ok" ]; then
  echo "Run 2 did not return a usable response."
  echo "Try a smaller num_ctx — the KV cache may not fit in VRAM alongside the weights."
elif [ -n "$COUNT_DEFAULT" ] && [ -n "$COUNT_EXPLICIT" ] && \
     [ "$COUNT_DEFAULT" -lt "$COUNT_EXPLICIT" ]; then
  echo "TRUNCATION DETECTED."
  echo "  default  : ${COUNT_DEFAULT} prompt tokens evaluated"
  echo "  explicit : ${COUNT_EXPLICIT} prompt tokens evaluated"
  echo
  echo "The server discarded part of your prompt and returned HTTP 200 anyway."
  echo "Truncation takes the FRONT, so system prompts and rules go first."
  echo
  echo "Fix, in increasing order of permanence:"
  echo "  per request : \"options\": {\"num_ctx\": ${NUM_CTX}}"
  echo "  server-wide : OLLAMA_CONTEXT_LENGTH=${NUM_CTX} ollama serve"
  echo "  per model   : PARAMETER num_ctx ${NUM_CTX} in a Modelfile"
else
  echo "No truncation at this prompt size (${COUNT_DEFAULT:-?} tokens evaluated in both runs)."
  echo "Raise FILLER_LINES at the top of this script and re-run, or check the"
  echo "CONTEXT column in 'ollama ps'."
fi

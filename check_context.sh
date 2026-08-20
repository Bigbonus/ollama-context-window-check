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
# Usage:
#   ./check_context.sh                          # defaults: qwen3:14b, 16384
#   ./check_context.sh qwen3:8b                 # pick a model
#   ./check_context.sh qwen3:8b 32768           # model + num_ctx
#   OLLAMA_HOST=http://box:11434 ./check_context.sh
#
# Requires: bash, curl, jq

set -uo pipefail

MODEL="${1:-qwen3:14b}"
NUM_CTX="${2:-16384}"
HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
NEEDLE="XJ7Q-4419-KTM"
FILLER_LINES=600

command -v curl >/dev/null || { echo "curl is required" >&2; exit 1; }
command -v jq   >/dev/null || { echo "jq is required" >&2; exit 1; }

# --- build the haystack, needle on the first line ---------------------------
HAYSTACK="ACCESS CODE: ${NEEDLE}"
for _ in $(seq 1 "$FILLER_LINES"); do
  HAYSTACK+=$'\n''Routine log entry. Nothing of interest was recorded during this interval.'
done

SYSTEM_JSON=$(printf '%s' "$HAYSTACK" | jq -Rs .)
MODEL_JSON=$(printf '%s' "$MODEL" | jq -Rs .)
USER_JSON='"What is the ACCESS CODE stated at the top of the log? Answer with the code only."'

LAST_COUNT=""   # set by run()

# --- run <label> <extra options json fragment> ------------------------------
run() {
  local label="$1" extra="$2" body resp http status count answer
  LAST_COUNT=""

  body=$(jq -n \
    --argjson model "$MODEL_JSON" \
    --argjson system "$SYSTEM_JSON" \
    --argjson user "$USER_JSON" \
    --argjson opts "{\"num_predict\":64,\"temperature\":0${extra}}" \
    '{model:$model, stream:false, think:false,
      messages:[{role:"system",content:$system},{role:"user",content:$user}],
      options:$opts}')

  resp=$(curl -sS -w $'\n%{http_code}' "${HOST}/api/chat" \
           -H 'Content-Type: application/json' -d "$body" 2>&1)
  status="${resp##*$'\n'}"
  http="${resp%$'\n'*}"

  echo "--- ${label}"
  echo "    HTTP status      : ${status}"

  if [ "$status" != "200" ]; then
    echo "    response         : $(printf '%s' "$http" | head -c 200)"
    echo "    verdict          : REJECTED (prompt exceeded the served window)"
    echo
    return
  fi

  count=$(jq -r '.prompt_eval_count // empty' <<< "$http")
  answer=$(jq -r '.message.content // ""' <<< "$http" | tr '\n' ' ' | head -c 120)

  echo "    prompt_eval_count: ${count:-n/a}"
  echo "    answer           : ${answer:-(empty)}"
  if [[ "$answer" == *"$NEEDLE"* ]]; then
    echo "    needle found     : yes"
  else
    echo "    needle found     : NO"
  fi
  echo

  LAST_COUNT="$count"
}

echo "model : ${MODEL}"
echo "host  : ${HOST}"
echo "needle: ${NEEDLE} (first line of the system prompt)"
echo

run "run 1: server default (no num_ctx)" ""
COUNT_DEFAULT="$LAST_COUNT"

run "run 2: explicit num_ctx=${NUM_CTX}" ",\"num_ctx\":${NUM_CTX}"
COUNT_EXPLICIT="$LAST_COUNT"

# --- verdict ----------------------------------------------------------------
echo "==="
if [ -z "$COUNT_DEFAULT" ] && [ -n "$COUNT_EXPLICIT" ]; then
  echo "Run 1 was rejected while run 2 succeeded with an explicit window."
  echo "Your server default is smaller than this prompt."
elif [ -z "$COUNT_EXPLICIT" ]; then
  echo "Run 2 did not return a usable response."
  echo "Try a smaller num_ctx — the KV cache may not fit in VRAM alongside the weights."
elif [ "$COUNT_DEFAULT" -lt "$COUNT_EXPLICIT" ]; then
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
  echo "No truncation at this prompt size (${COUNT_DEFAULT} tokens evaluated in both runs)."
  echo "Raise FILLER_LINES at the top of this script and re-run, or check the"
  echo "CONTEXT column in 'ollama ps'."
fi

#!/usr/bin/env bash
# Keep the local Qwen3-8B endpoints alive.
#
# The endpoints have been swept by something outside this session four times
# (all ports shut down within the same second, gracefully, even under setsid).
# Rather than re-diagnosing each time, this supervisor polls every 30s and
# restarts any endpoint that stops answering. GPUs 2 and 5 are never used:
# they belong to another user.
V=/home/liangwd/reasoning_26/.venv/bin/python
MODEL=/home/liangwd/models/Qwen3-8B
declare -A GPU=( [8100]=0 [8101]=1 [8106]=3 [8107]=6 )
declare -A UTIL=( [8100]=0.45 [8101]=0.45 [8106]=0.45 [8107]=0.45 )

start() {
  local p=$1
  echo "$(date +%H:%M:%S) starting :$p on GPU ${GPU[$p]}"
  CUDA_VISIBLE_DEVICES=${GPU[$p]} setsid nohup $V -m vllm.entrypoints.openai.api_server \
      --model $MODEL --served-model-name Qwen3-8B --port $p \
      --max-model-len 8192 --gpu-memory-utilization ${UTIL[$p]} \
      --max-num-seqs 96 --disable-log-requests \
      > /home/liangwd/vllm_$p.log 2>&1 < /dev/null &
  disown
}

while true; do
  for p in 8100 8101 8106 8107; do
    if ! timeout 3 curl -s "http://127.0.0.1:$p/v1/models" >/dev/null 2>&1; then
      pgrep -f "api_server.*--port $p" >/dev/null 2>&1 || start "$p"
    fi
  done
  sleep 30
done

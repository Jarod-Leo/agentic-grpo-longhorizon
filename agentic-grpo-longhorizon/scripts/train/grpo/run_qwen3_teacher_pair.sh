#!/bin/bash
# Both ranks run inside one same-node Slurm step, each bound to one GPU.
set -eo pipefail
: "${RUN_ROOT:?}" "${REPO:?}" "${SLURM_PROCID:?}"
export TEACHER_PORT=${TEACHER_PORT:-18932}
export TEACHER_ENDPOINT=http://127.0.0.1:$TEACHER_PORT
export TEACHER_MODEL=${TEACHER_MODEL:-/projects/_hdd/cabinagentrlarchive/CabinAgent-RL/models/Qwen/Qwen3-32B}
export QWEN3_CONFIG=${QWEN3_CONFIG:-qwen3_opd}
export STUDENT_ENTRY=${STUDENT_ENTRY:-scripts/train/grpo/run_qwen3_formal.sh}
cd "$REPO"
if [ "$SLURM_PROCID" = 1 ]; then
    python scripts/train/grpo/serve_token_teacher.py --model-path "$TEACHER_MODEL" \
        --revision 9216db5781bf21249d130ec9da846c4624c16137 --port "$TEACHER_PORT" &
    teacher_pid=$!
    trap 'kill "$teacher_pid" 2>/dev/null || true' EXIT TERM INT
    while kill -0 "$teacher_pid" 2>/dev/null && [ ! -f "$RUN_ROOT/student-finished-$SLURM_JOB_ID" ]; do sleep 5; done
    if [ -f "$RUN_ROOT/student-finished-$SLURM_JOB_ID" ]; then
        kill "$teacher_pid" 2>/dev/null || true
        wait "$teacher_pid" || true
    else
        wait "$teacher_pid"
        exit 1
    fi
else
    test "$SLURM_PROCID" = 0
    if [ -n "${WANDB_ENV_FILE:-}" ]; then
        test -r "$WANDB_ENV_FILE" || { echo "Cannot read WANDB_ENV_FILE: $WANDB_ENV_FILE" >&2; exit 2; }
        set -a
        source "$WANDB_ENV_FILE"
        set +a
    fi
    trap 'touch "$RUN_ROOT/student-finished-$SLURM_JOB_ID"' EXIT
    python - <<'PY_WAIT'
import json, os, time, urllib.request
for attempt in range(180):
    try:
        with urllib.request.urlopen(os.environ["TEACHER_ENDPOINT"] + "/health", timeout=5) as response:
            info = json.load(response)
        assert info["model_id"] == "Qwen/Qwen3-32B"
        assert info["revision"] == "9216db5781bf21249d130ec9da846c4624c16137"
        print("TEACHER_READY", info, flush=True)
        break
    except OSError:
        time.sleep(5)
else:
    raise RuntimeError("Teacher startup timed out")
PY_WAIT
    bash "$STUDENT_ENTRY"
fi

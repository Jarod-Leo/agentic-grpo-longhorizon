#!/bin/bash
# Both ranks run inside one same-node Slurm step, each bound to one GPU.
set -eo pipefail
: "${RUN_ROOT:?}" "${REPO:?}" "${SLURM_PROCID:?}"
export TEACHER_PORT=${TEACHER_PORT:-18932}
export TEACHER_ENDPOINT=http://127.0.0.1:$TEACHER_PORT
export QWEN3_CONFIG=${QWEN3_CONFIG:-qwen3_opd}
if [ "$QWEN3_CONFIG" = qwen3_grpo_skill_sd ]; then
    export TEACHER_MODEL=${TEACHER_MODEL:-/projects/_hdd/cabinagentrlarchive/CabinAgent-RL/models/Qwen/Qwen3-8B}
    export TEACHER_MODEL_ID=${TEACHER_MODEL_ID:-Qwen/Qwen3-8B}
    export TEACHER_REVISION=${TEACHER_REVISION:-b968826d9c46dd6066d109eabc6255188de91218}
    export TEACHER_ADAPTER_PATH=${TEACHER_ADAPTER_PATH:-/projects/_hdd/cabinagentrlarchive/CabinAgent-RL/checkpoints/e01_vanilla_grpo/formal-seed42-200-v3/global_step_200/actor/lora_adapter}
    : "${TEACHER_ADAPTER_SHA256:?Set the pinned E01 adapter SHA256}"
else
    export TEACHER_MODEL=${TEACHER_MODEL:-/projects/_hdd/cabinagentrlarchive/CabinAgent-RL/models/Qwen/Qwen3-32B}
    export TEACHER_MODEL_ID=${TEACHER_MODEL_ID:-Qwen/Qwen3-32B}
    export TEACHER_REVISION=${TEACHER_REVISION:-9216db5781bf21249d130ec9da846c4624c16137}
fi
export TEACHER_ADAPTER_PATH=${TEACHER_ADAPTER_PATH:-}
export TEACHER_ADAPTER_SHA256=${TEACHER_ADAPTER_SHA256:-}
export STUDENT_ENTRY=${STUDENT_ENTRY:-scripts/train/grpo/run_qwen3_formal.sh}
cd "$REPO"
if [ "$SLURM_PROCID" = 1 ]; then
    teacher_args=()
    if [ "$QWEN3_CONFIG" = qwen3_grpo_skill_sd ]; then
        teacher_args+=(--verify-revision --self-check)
    fi
    if [ -n "$TEACHER_ADAPTER_PATH" ]; then
        : "${TEACHER_ADAPTER_SHA256:?Set the pinned teacher adapter SHA256}"
        teacher_args+=(--adapter-path "$TEACHER_ADAPTER_PATH" --adapter-sha256 "$TEACHER_ADAPTER_SHA256")
    elif [ -n "$TEACHER_ADAPTER_SHA256" ]; then
        echo 'TEACHER_ADAPTER_SHA256 requires TEACHER_ADAPTER_PATH' >&2
        exit 2
    fi
    python scripts/train/grpo/serve_token_teacher.py --model-path "$TEACHER_MODEL" \
        --model-id "$TEACHER_MODEL_ID" --revision "$TEACHER_REVISION" --port "$TEACHER_PORT" \
        "${teacher_args[@]}" > "$RUN_ROOT/token-teacher-$SLURM_JOB_ID.log" 2>&1 &
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
from pathlib import Path
for attempt in range(180):
    try:
        with urllib.request.urlopen(os.environ["TEACHER_ENDPOINT"] + "/health", timeout=5) as response:
            info = json.load(response)
        expected = {
            "status": "ready",
            "model_id": os.environ["TEACHER_MODEL_ID"],
            "revision": os.environ["TEACHER_REVISION"],
            "adapter_path": str(Path(os.environ["TEACHER_ADAPTER_PATH"]).resolve()) if os.environ["TEACHER_ADAPTER_PATH"] else None,
            "adapter_sha256": os.environ["TEACHER_ADAPTER_SHA256"] or None,
        }
        if os.environ["QWEN3_CONFIG"] == "qwen3_grpo_skill_sd":
            expected["revision_verified"] = True
        if any(info.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"Teacher health identity mismatch: {info}")
        (Path(os.environ["RUN_ROOT"]) / f'teacher-health-{os.environ["SLURM_JOB_ID"]}.json').write_text(json.dumps(info, indent=2) + "\n")
        print("TEACHER_READY", info, flush=True)
        break
    except OSError:
        time.sleep(5)
else:
    raise RuntimeError("Teacher startup timed out")
PY_WAIT
    bash "$STUDENT_ENTRY" "$@"
fi

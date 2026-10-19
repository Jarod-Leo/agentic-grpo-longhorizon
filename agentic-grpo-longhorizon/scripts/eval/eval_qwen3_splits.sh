#!/bin/bash
# Evaluate one frozen policy on the train and test splits in the same Slurm step.
set -uo pipefail

MODEL_RUN_ROOT=${RUN_ROOT:?Set RUN_ROOT to the model evaluation directory}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
overall_rc=0

for split in train test; do
    export RUN_ROOT="$MODEL_RUN_ROOT/$split"
    export RUN_DIR="$RUN_ROOT/eval"
    export EVAL_SPLIT="$split"
    export RAY_TMPDIR="/projects/_ssd/jiatian001ssd/q-${SLURM_JOB_ID:?}-${split}"

    if bash "$SCRIPT_DIR/eval_qwen3.sh" "$@"; then
        echo "QWEN3_SPLIT_COMPLETE split=$split dir=$RUN_DIR"
    else
        split_rc=$?
        overall_rc=$split_rc
        echo "QWEN3_SPLIT_FAILED split=$split rc=$split_rc dir=$RUN_DIR" >&2
    fi

    if [ -n "${EVAL_SUITE_MANIFEST:-}" ]; then
        if ! python "$SCRIPT_DIR/summarize_qwen3_suite.py" "$EVAL_SUITE_MANIFEST"; then
            overall_rc=1
            echo "QWEN3_SUITE_SUMMARY_FAILED manifest=$EVAL_SUITE_MANIFEST" >&2
        fi
    fi
done

exit "$overall_rc"

#!/bin/bash
# Optional development tools and Git checkout; run via the shared CPU Slurm shell.
set -eo pipefail
: "${TOOLS_DIR:?Set project SSD tools directory}" "${GIT_WORKTREE:?Set Git checkout directory}"
if [ ! -x "$TOOLS_DIR/bin/ruff" ]; then
    python -m pip install --disable-pip-version-check --no-deps --target "$TOOLS_DIR" ruff==0.12.5
fi
if [ ! -d "$GIT_WORKTREE/.git" ]; then
    git clone --filter=blob:none --depth=1 https://github.com/Jarod-Leo/agentic-grpo-longhorizon.git "$GIT_WORKTREE"
fi

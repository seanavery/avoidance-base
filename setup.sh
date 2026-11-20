#!/usr/bin/env bash
# First-run setup: create a virtualenv and install module requirements.
# Also called by run.sh when .venv is missing, so it must be safe to re-run.
set -euo pipefail

cd "$(dirname "$0")"

VENV_NAME=".venv"
MARKER="${VENV_NAME}/.setup-complete"

log() { echo "[avoidance-base setup] $*"; }
die() { echo "[avoidance-base setup] ERROR: $*" >&2; exit 1; }

# Use an absolute interpreter path. With a bare "python3" and PATH not exported
# to the process, Python cannot resolve sys.executable and `venv` fails with
# "Unable to symlink ''".
PYTHON=$(command -v "${PYTHON:-python3}" || true)
if [ -z "${PYTHON}" ] || ! "$PYTHON" --version >/dev/null 2>&1; then
    die "python3 not found on PATH"
fi
if [ ! -f "${VENV_NAME}/bin/activate" ]; then
    rm -rf "${VENV_NAME}"
    log "creating virtualenv in ${VENV_NAME}"
    if ! "$PYTHON" -m venv --system-site-packages "${VENV_NAME}"; then
        die "failed to create ${VENV_NAME}; install python3-venv and re-run setup"
    fi
fi
rm -f "${MARKER}"
# shellcheck disable=SC1091
source "${VENV_NAME}/bin/activate"
pip install --upgrade pip -q
pip install -r requirements.txt -q
touch "${MARKER}"
log "setup complete"

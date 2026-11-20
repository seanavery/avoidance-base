#!/usr/bin/env bash
# Entrypoint for the avoidance-base module.
set -euo pipefail

cd "$(dirname "$0")"

# viam-server runs setup.sh once (first_run) and records that outside the module
# directory. A reload module is re-extracted on every viam-server start, which
# deletes .venv but keeps that record, so first_run never runs again. Rebuild the
# venv here when it is missing or setup did not finish.
if [ ! -f .venv/.setup-complete ]; then
    echo "[avoidance-base] virtualenv missing or incomplete; running setup.sh" >&2
    ./setup.sh >&2
fi

# shellcheck disable=SC1091
source .venv/bin/activate

# Cloud transforms are small matmuls. Leaving OpenBLAS / MKL free to spawn
# a thread per core turns two 10 Hz cameras into multi-core pegging.
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"

exec python -m src.main "$@"

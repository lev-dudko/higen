#!/usr/bin/env bash
# Smoke-test the GATr-lite pipeline: 10k DR1 + 10k tT, 5 epochs.
#
# Usage (after activating your venv):
#   bash demo/paper1/paper1_demo/gatr_lite/scripts/smoke_test.sh
#
# Should complete in <5 min on a modern desktop GPU and print a final val AUC.
set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_DIR="$(cd "$THIS_DIR/../../.." && pwd)"

SRC_DIR="${SRC_DIR:-$DEMO_DIR}"
SMOKE_DIR="${SMOKE_DIR:-$DEMO_DIR/data/smoke}"
RUN_DIR="${RUN_DIR:-$DEMO_DIR/runs/smoke_$(date +%Y%m%d_%H%M%S)}"
LHE_BASE="${LHE_BASE:-$DEMO_DIR/samples}"

LHE_DR1="${LHE_DR1:-${LHE_BASE}/GG_DR1_NeeuDbB_SM_mu.lhe}"
LHE_TT="${LHE_TT:-${LHE_BASE}/GG_tT_NeeuDbB_SM_mu.lhe}"

N_SMOKE="${N_SMOKE:-10000}"
EPOCHS="${EPOCHS:-5}"
BATCH="${BATCH:-256}"
LR="${LR:-1e-3}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "$SMOKE_DIR" "$RUN_DIR"

cd "$SRC_DIR"
export PYTHONPATH="$SRC_DIR:${PYTHONPATH:-}"

for f in "$LHE_DR1" "$LHE_TT"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: missing LHE input: $f" >&2
        echo "       See demo/paper1/README.md for download URLs." >&2
        exit 1
    fi
done

echo "===== preprocess DR1 (smoke, $N_SMOKE events) ====="
if [[ ! -f "$SMOKE_DIR/dr1.h5" ]]; then
    python -m paper1_demo.gatr_lite.data.preprocess \
        --input "$LHE_DR1" \
        --output "$SMOKE_DIR/dr1.h5" \
        --label 1 \
        --max-events "$N_SMOKE" \
        --pt-cut 10
else
    echo "  reusing existing $SMOKE_DIR/dr1.h5"
fi

echo "===== preprocess tT (smoke, $N_SMOKE events) ====="
if [[ ! -f "$SMOKE_DIR/tT.h5" ]]; then
    python -m paper1_demo.gatr_lite.data.preprocess \
        --input "$LHE_TT" \
        --output "$SMOKE_DIR/tT.h5" \
        --label 0 \
        --max-events "$N_SMOKE" \
        --pt-cut 10
else
    echo "  reusing existing $SMOKE_DIR/tT.h5"
fi

echo "===== train ($EPOCHS epochs, batch=$BATCH, lr=$LR) ====="
python -m paper1_demo.gatr_lite.training.train \
    --signal "$SMOKE_DIR/dr1.h5" \
    --background "$SMOKE_DIR/tT.h5" \
    --out "$RUN_DIR" \
    --batch-size "$BATCH" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --weight-decay 1e-4 \
    --val-frac 0.1 \
    --seed 42 \
    --device "$DEVICE" \
    --early-stop-patience "$EPOCHS" \
    --num-workers 2

echo "===== smoke-test summary ====="
if [[ -f "$RUN_DIR/metrics.csv" ]]; then
    column -s, -t "$RUN_DIR/metrics.csv"
    LAST_AUC=$(awk -F, 'NR>1{auc=$4} END{print auc}' "$RUN_DIR/metrics.csv")
    echo "Final val AUC: ${LAST_AUC}"
fi
echo "Run directory: $RUN_DIR"

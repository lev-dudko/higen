#!/usr/bin/env bash
# Preprocess the three LHE samples (DR1, tT, full tT_tWb) into HDF5.
#
# Override paths via environment variables.  Defaults assume the layout from
# demo/paper1/README.md:
#   demo/paper1/samples/<file>.lhe     - LHE inputs (download from www-hep)
#   demo/paper1/data/<file>.h5         - HDF5 outputs (this script writes them)
#
# Usage:
#   source <your-venv>/bin/activate
#   bash demo/paper1/paper1_demo/gatr_lite/scripts/preprocess_all.sh
set -euo pipefail

# Resolve repository layout from this script's location (3 levels up = demo/paper1).
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_DIR="$(cd "$THIS_DIR/../../.." && pwd)"

SRC_DIR="${SRC_DIR:-$DEMO_DIR}"
DATA_DIR="${DATA_DIR:-$DEMO_DIR/data}"
LHE_BASE="${LHE_BASE:-$DEMO_DIR/samples}"

LHE_DR1="${LHE_DR1:-${LHE_BASE}/GG_DR1_NeeuDbB_SM_mu.lhe}"
LHE_TT="${LHE_TT:-${LHE_BASE}/GG_tT_NeeuDbB_SM_mu.lhe}"
LHE_FULL="${LHE_FULL:-${LHE_BASE}/GG_tT_tWb_NeeuDbB_SM_mu.lhe}"

mkdir -p "$DATA_DIR"

for f in "$LHE_DR1" "$LHE_TT" "$LHE_FULL"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: missing LHE input: $f" >&2
        echo "       See demo/paper1/README.md for download URLs." >&2
        exit 1
    fi
done

cd "$SRC_DIR"
export PYTHONPATH="$SRC_DIR:${PYTHONPATH:-}"

run_one() {
    local lhe="$1"
    local out="$2"
    local label="$3"
    echo "===== preprocess: $(basename "$lhe") -> $(basename "$out") (label=$label) ====="
    python -m paper1_demo.gatr_lite.data.preprocess \
        --input "$lhe" \
        --output "$out" \
        --label "$label" \
        --max-events 0 \
        --pt-cut 10
    python - "$out" <<'PY'
import sys, h5py
p = sys.argv[1]
with h5py.File(p, "r") as f:
    print(f"  file        : {p}")
    print(f"  n_events    : {f['p4'].shape[0]}")
    print(f"  xsec_pb     : {f.attrs.get('xsec_pb')}")
    print(f"  n_total     : {f.attrs.get('n_total_in_file')}")
    print(f"  n_passed    : {f.attrs.get('n_passed_cut')}")
    print(f"  n_drop_can  : {f.attrs.get('n_dropped_canonical')}")
    print(f"  n_drop_pt   : {f.attrs.get('n_dropped_pt')}")
PY
}

run_one "$LHE_DR1"  "$DATA_DIR/dr1.h5"  1
run_one "$LHE_TT"   "$DATA_DIR/tT.h5"   0
run_one "$LHE_FULL" "$DATA_DIR/full.h5" 0

echo
echo "All preprocess jobs finished. Output dir: $DATA_DIR"
ls -la "$DATA_DIR"

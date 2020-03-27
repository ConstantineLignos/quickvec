#!/usr/bin/env bash
set -euo pipefail

for vec in tests/test_data/*.vec; do
  gzip -k "$vec"
  out="${vec%.*}.sqlite"
  python quickvec/convert.py "$vec" "$out"
done

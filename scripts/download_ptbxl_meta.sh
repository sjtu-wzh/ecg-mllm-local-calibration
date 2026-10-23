#!/bin/bash
# PTB-XL 1.0.3 metadata (CC BY 4.0, PhysioNet). Only the two CSV files are needed (no signals).
set -e
cd "$(dirname "$0")/.."; mkdir -p data/ptbxl_meta
for f in ptbxl_database.csv scp_statements.csv; do
  curl -L --retry 5 -o data/ptbxl_meta/$f https://physionet.org/files/ptb-xl/1.0.3/$f
done

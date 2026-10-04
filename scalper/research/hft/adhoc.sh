#!/usr/bin/env bash
set -euo pipefail
python -m research.hft.scan --day 2026-09-15 --out ../out --workers 6

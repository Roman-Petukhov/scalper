#!/usr/bin/env bash
set -euo pipefail
python -m research.hft.spikes_bybit --jobs research/hft/spikes_bybit_jobs.json --out ../out --workers 8

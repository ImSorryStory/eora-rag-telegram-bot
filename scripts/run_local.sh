#!/usr/bin/env bash
set -euo pipefail

python -m app.ingest --urls-file links.txt --local-dir data/sources
python -m app.main
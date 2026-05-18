#!/bin/bash
export KMP_DUPLICATE_LIB_OK=TRUE
cd "$(dirname "$0")"
python scripts/serve_screening_ui.py

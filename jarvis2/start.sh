#!/bin/bash
# Jarvis2 - Advanced modular voice assistant launcher
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Suppress Hugging Face Hub warnings (models are cached locally)
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=1

# Activate venv if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

python main.py "$@"

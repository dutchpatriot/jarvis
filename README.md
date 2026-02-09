# Jarvis

Voice assistant for Linux, powered by local LLMs (Ollama), Whisper STT, and Piper TTS.
Everything runs locally - no cloud, no subscriptions, no data leaving your machine.

## Structure

```
jarvis/
├── assistmint/    # Shared core library (audio, STT, TTS, wake word, NLP, modules)
├── jarvis1/       # V1 - Simple dictation assistant
└── jarvis2/       # V2 - Modular voice assistant
                   #   Chat, Calendar, Coding, Project, Dictation, Terminal
```

## Requirements

- **Linux** (Ubuntu 24.04+)
- **Python** 3.10+
- **NVIDIA GPU** with CUDA (for Whisper STT + Piper TTS acceleration)
- **Ollama** - Local LLM inference (https://ollama.ai)
- **xdotool** + **xclip** - Keyboard simulation for dictation

## Install

```bash
# 1. Clone
git clone https://github.com/dutchpatriot/jarvis.git
cd jarvis

# 2. System dependencies
sudo apt install xdotool xclip

# 3. Set up Jarvis2 (recommended)
cd jarvis2
python3 -m venv venv
source venv/bin/activate

# 4. Install the shared core library
pip install -e ../assistmint

# 5. Install Jarvis2 dependencies
pip install -r requirements.txt

# 6. Pull an Ollama model
ollama pull qwen2.5:0.5b

# 7. Download Piper TTS voices
mkdir -p ~/.local/share/piper/voices
# English:
wget -O ~/.local/share/piper/voices/en_US-lessac-medium.onnx \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
wget -O ~/.local/share/piper/voices/en_US-lessac-medium.onnx.json \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```

## Usage

```bash
# Voice mode (wake word: "Hey Jarvis")
cd jarvis2
./start.sh

# Text mode
./start.sh --type

# With project directory (for project mode)
./start.sh --type --project-dir /path/to/your/project
```

## Jarvis2 Modules

| Module | Trigger | Description |
|--------|---------|-------------|
| **Chat** | Any question | General Q&A via Ollama LLM |
| **Calendar** | "add to calendar", "what's on today" | Manage events (Evolution/Google) |
| **Dictation** | "dictation mode" | Voice-to-text typing into any window |
| **Coding** | "coding mode", "join me" | Voice pair programming |
| **Project** | "project mode" | Explore codebases, multi-file LLM context |
| **Terminal** | "run [command]", "open terminal" | Voice-activated terminal commands |

## Configuration

All settings in `jarvis2/config.py` - models, audio, TTS voices, wake word sensitivity, module settings.

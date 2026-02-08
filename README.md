# Jarvis

Voice assistant for Linux, powered by local LLMs (Ollama), Whisper STT, and Piper TTS.

## Structure

```
jarvis/
├── jarvis1/   # V1 - Simple dictation assistant
└── jarvis2/   # V2 - Modular voice assistant
               #   Commands, Calendar, Coding, Project, Dictation, Q&A
```

## Dependencies

- **assistmint** - Shared core library (`pip install -e ../assistmint`)
- **Ollama** - Local LLM inference
- **CUDA** - GPU acceleration (NVIDIA)
- **xdotool** - Keyboard simulation (dictation)

## Quick Start

```bash
# Jarvis2 (recommended)
cd jarvis2
./start.sh          # Voice mode
./start.sh --type   # Text mode
```

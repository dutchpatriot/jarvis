"""
Terminal colors for Assistmint debug output.
"""
from config import USE_EMOJIS

# ANSI escape codes
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

# Foreground colors
BLACK = "\033[30m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
WHITE = "\033[97m"

# Tag colors mapping
TAG_COLORS = {
    "CMD": YELLOW,
    "STT": CYAN,
    "TTS": MAGENTA,
    "WAKE": GREEN,
    "V2J": BLUE,
    "LEARN": GREEN,
    "SESSION": YELLOW,
    "DICTATE": CYAN,
    "DATE": DIM,
    "ERROR": RED,
}

# Tag emojis (only used if USE_EMOJIS=True)
TAG_EMOJIS = {
    "CMD": "⚡",
    "STT": "🎤",
    "TTS": "🔊",
    "WAKE": "👂",
    "V2J": "🧠",
    "LEARN": "📚",
    "SESSION": "💬",
    "DICTATE": "✍️",
    "DATE": "📅",
    "ERROR": "❌",
}


def tag(name: str, message: str = "") -> str:
    """Format a colored tag with optional message.

    Usage:
        print(tag("CMD", "Processing command"))
        print(tag("STT") + " Transcribing...")
    """
    color = TAG_COLORS.get(name, WHITE)
    emoji = TAG_EMOJIS.get(name, "") if USE_EMOJIS else ""
    prefix = f"{emoji} " if emoji else ""
    if message:
        return f"{prefix}{color}[{name}]{RESET} {message}"
    return f"{prefix}{color}[{name}]{RESET}"


def cmd(msg: str) -> str:
    return tag("CMD", msg)


def stt(msg: str) -> str:
    return tag("STT", msg)


def tts(msg: str) -> str:
    return tag("TTS", msg)


def wake(msg: str) -> str:
    return tag("WAKE", msg)


def v2j(msg: str) -> str:
    return tag("V2J", msg)


def learn(msg: str) -> str:
    return tag("LEARN", msg)


def session(msg: str) -> str:
    return tag("SESSION", msg)


def dictate(msg: str) -> str:
    return tag("DICTATE", msg)


def error(msg: str) -> str:
    return tag("ERROR", msg)


# Test
if __name__ == "__main__":
    print(cmd("Processing: help me"))
    print(stt("Transcribing..."))
    print(tts("Speaking: Hello world"))
    print(wake("Listening... (say 'Hey Jarvis')"))
    print(v2j("Intent: Help (en) conf=1.00"))
    print(learn("Saved: 'calander' → 'calendar'"))
    print(session("Loaded 5 messages"))
    print(dictate("Typing: hello world"))
    print(error("Something went wrong"))

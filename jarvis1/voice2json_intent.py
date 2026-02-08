"""
Voice2json Intent Recognition Module
Uses voice2json via Docker for bilingual (EN/NL) intent recognition.
"""

import subprocess
import json
import os
from typing import Optional, Dict, Any
from colors import v2j, error

# Profile paths
PROFILES = {
    "en": "en-us_kaldi-rhasspy",
    "nl": "nl_kaldi-rhasspy"
}

# Intent to function mapping
INTENT_ACTIONS = {
    "Help": "help",
    "ClearSession": "clear_session",
    "LearnCorrection": "learn_correction",
    "ShowCorrections": "show_corrections",
    "AddCalendar": "add_calendar",
    "CheckCalendar": "check_calendar",
    "ClearCalendar": "clear_calendar",
    "RemoveCalendar": "remove_calendar",
    "Dictate": "dictate",
    "StopDictate": "stop_dictate",
    "Terminal": "terminal",
    "Confirm": "confirm",
    "Deny": "deny",
    "OpenBrowser": "open_browser",
}

# Debug flag - set via config or environment
DEBUG = os.environ.get("VOICE2JSON_DEBUG", "0") == "1"


def _run_voice2json(profile: str, command: str, args: list = None, input_text: str = None) -> Optional[str]:
    """Run voice2json command via Docker."""
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{os.environ['HOME']}:{os.environ['HOME']}",
        "-w", os.getcwd(),
        "-e", f"HOME={os.environ['HOME']}",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "synesthesiam/voice2json",
        "--profile", profile,
        command
    ]
    if args:
        cmd.extend(args)

    try:
        if input_text:
            result = subprocess.run(cmd, input=input_text, capture_output=True, text=True, timeout=10)
        else:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

        if DEBUG:
            print(v2j(f"Command: {' '.join(cmd)}"))
            if result.stderr:
                print(v2j(f"Stderr: {result.stderr[:200]}"))

        return result.stdout.strip() if result.stdout else None
    except subprocess.TimeoutExpired:
        print(error("Docker command timed out"))
        return None
    except Exception as e:
        print(error(f"{e}"))
        return None


def recognize_intent(text: str, language: str = "auto") -> Dict[str, Any]:
    """
    Recognize intent from text using voice2json.

    Args:
        text: The transcribed text to analyze
        language: "en", "nl", or "auto" (tries both)

    Returns:
        Dict with 'intent', 'confidence', 'action', 'language', 'entities'
        Returns {'intent': None, 'action': 'ollama'} if no intent matched
    """
    result = {
        "intent": None,
        "confidence": 0.0,
        "action": "ollama",  # Default: send to LLM
        "language": None,
        "entities": [],
        "text": text
    }

    profiles_to_try = []
    if language == "auto":
        # Try both, prefer the one with higher confidence
        profiles_to_try = [("en", PROFILES["en"]), ("nl", PROFILES["nl"])]
    elif language in PROFILES:
        profiles_to_try = [(language, PROFILES[language])]
    else:
        print(v2j(f"Unknown language: {language}"))
        return result

    best_match = None
    best_confidence = 0.0

    for lang, profile in profiles_to_try:
        output = _run_voice2json(profile, "recognize-intent", ["--text-input", text])

        if output:
            try:
                data = json.loads(output)
                intent_name = data.get("intent", {}).get("name")
                confidence = data.get("intent", {}).get("confidence", 0.0)

                if DEBUG:
                    print(v2j(f"{lang}: intent={intent_name}, confidence={confidence}"))

                if intent_name and confidence > best_confidence:
                    best_confidence = confidence
                    best_match = {
                        "intent": intent_name,
                        "confidence": confidence,
                        "action": INTENT_ACTIONS.get(intent_name, "ollama"),
                        "language": lang,
                        "entities": data.get("entities", []),
                        "slots": data.get("slots", {}),
                        "text": text
                    }
            except json.JSONDecodeError as e:
                if DEBUG:
                    print(v2j(f"JSON parse error: {e}"))

    if best_match and best_confidence > 0.5:  # Threshold for accepting intent
        return best_match

    return result


def transcribe_audio(wav_file: str, language: str = "en") -> Optional[str]:
    """
    Transcribe audio file using voice2json.

    Args:
        wav_file: Path to WAV file (16kHz, 16-bit, mono)
        language: "en" or "nl"

    Returns:
        Transcribed text or None
    """
    if language not in PROFILES:
        print(v2j(f"Unknown language: {language}"))
        return None

    profile = PROFILES[language]

    # Read the wav file and pipe to docker
    try:
        with open(wav_file, "rb") as f:
            wav_data = f.read()

        cmd = [
            "docker", "run", "-i", "--rm",
            "-v", f"{os.environ['HOME']}:{os.environ['HOME']}",
            "-w", os.getcwd(),
            "-e", f"HOME={os.environ['HOME']}",
            "--user", f"{os.getuid()}:{os.getgid()}",
            "synesthesiam/voice2json",
            "--profile", profile,
            "transcribe-wav"
        ]

        result = subprocess.run(cmd, input=wav_data, capture_output=True, timeout=30)

        if result.stdout:
            data = json.loads(result.stdout.decode())
            return data.get("text", "")
        return None
    except Exception as e:
        print(error(f"Transcribe error: {e}"))
        return None


def get_available_intents(language: str = "en") -> list:
    """Get list of available intents for a language."""
    return list(INTENT_ACTIONS.keys())


# Quick test
if __name__ == "__main__":
    # Test English
    print("Testing English:")
    print(recognize_intent("help me"))
    print(recognize_intent("add to calendar"))
    print(recognize_intent("what is the weather today"))  # Should fallback to ollama

    print("\nTesting Dutch:")
    print(recognize_intent("vergeet alles"))
    print(recognize_intent("bekijk mijn agenda"))

    print("\nTesting auto-detect:")
    print(recognize_intent("help", "auto"))
    print(recognize_intent("dicteer", "auto"))

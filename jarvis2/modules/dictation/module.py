"""
Dictation Module - Continuous voice-to-text typing.

Features:
- Continuous listening without wake word
- Sleep/wake modes
- Grammar processing (punctuation, spelling)
- Keyboard shortcuts and actions
"""

import subprocess
import time
from typing import Optional, List

from assistmint.core.modules.base import BaseModule, ModuleResult, ModuleContext, ModuleCapability
from assistmint.core.audio.stt import whisper_speech_to_text
from assistmint.core.audio.tts import speak
from assistmint.core.audio.wake import listen_for_wake_word
from assistmint.core.nlp.filters import is_hallucination
from assistmint.core.logger import dictate as log_dictate, cmd
from modules.dictation.grammar import get_grammar_processor

# Import config values
try:
    from config import DICTATE_SLEEP_WORDS, DICTATE_EMOJIS, WAKE_WORD, LANG_SWITCH_EN, LANG_SWITCH_NL, LANG_SWITCH_AUTO
except ImportError:
    DICTATE_SLEEP_WORDS = ["sleep", "slaap", "pause", "pauze"]
    DICTATE_EMOJIS = False
    WAKE_WORD = "hey_jarvis"
    LANG_SWITCH_EN = ["speak english", "english"]
    LANG_SWITCH_NL = ["speak dutch", "dutch", "nederlands", "spreek nederlands"]
    LANG_SWITCH_AUTO = ["auto language", "automatisch"]

# Import TTS language control
try:
    from assistmint.core.audio.tts import set_language, get_language
except ImportError:
    from text_to_speech import set_language, get_language

# Bilingual control messages (en, nl)
DICTATION_MSGS = {
    "start": ("Dictating. Say stop to end.", "Dicteren. Zeg stop om te stoppen."),
    "end": ("Dictation ended.", "Dicteren gestopt."),
    "stop": ("Stopping dictation.", "Dicteren stoppen."),
    "sleep": ("Sleeping.", "Slapen."),
    "resume": ("Resumed.", "Hervat."),
    "nl": ("Nederlands.", "Nederlands."),
    "en": ("Switched to English.", "Engels."),
    "auto": ("Auto.", "Automatisch."),
    "xdotool_missing": ("xdotool is not installed. Cannot type.", "xdotool niet geïnstalleerd. Kan niet typen."),
}

def _msg(key: str) -> str:
    """Get message in current language."""
    lang = get_language()
    msgs = DICTATION_MSGS.get(key, ("", ""))
    return msgs[1] if lang == "nl" else msgs[0]


class DictationModule(BaseModule):
    """
    Dictation module for continuous voice-to-text typing.

    Uses xdotool to type text directly into the active window.
    """

    def __init__(self):
        super().__init__()
        self._device = None
        self._samplerate = 16000
        self._is_sleeping = False
        self._spell_mode = False  # Dedicated spell mode flag
        self._grammar = get_grammar_processor()

    @property
    def name(self) -> str:
        return "dictation"

    @property
    def capabilities(self) -> ModuleCapability:
        return (
            ModuleCapability.TEXT_INPUT |
            ModuleCapability.TEXT_OUTPUT |
            ModuleCapability.CONTINUOUS |
            ModuleCapability.SYSTEM_ACCESS
        )

    @property
    def description(self) -> str:
        return "Voice-to-text typing (continuous mode)"

    @property
    def triggers(self) -> List[str]:
        return [
            "dictate", "dicteer", "start typing", "type this",
            "start dictation", "begin typing"
        ]

    @property
    def priority(self) -> int:
        return 80  # High priority for explicit dictation requests

    def can_handle(self, text: str, intent: Optional[str] = None) -> float:
        """Check if this is a dictation request."""
        text_lower = text.lower()

        # Check for explicit intent
        if intent == "dictate":
            return 1.0

        # Check for trigger phrases
        if any(t in text_lower for t in self.triggers):
            return 0.95

        return 0.0

    def execute(self, context: ModuleContext) -> ModuleResult:
        """Start dictation mode."""
        self._device = context.selected_device
        self._samplerate = context.samplerate

        self._show_dictation_banner()
        speak(_msg("start"))

        self._dictation_loop()

        return ModuleResult(
            text="Dictation mode ended.",
            success=True,
            continue_listening=False
        )

    def _show_dictation_banner(self):
        """Display compact dictation help when entering dictation mode."""
        R = "\033[0m"
        B = "\033[1m"
        CYAN = "\033[96m"
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        MAGENTA = "\033[95m"
        WHITE = "\033[97m"
        BG_CYAN = "\033[46m"

        wake_phrase = WAKE_WORD.replace('_', ' ').title()
        RED = "\033[91m"
        print(f"""
{B}{WHITE}{BG_CYAN}╔═══════════════════════════════════════════════════════════════════════════╗
║  ✍️  DICTATION MODE - "stop"/"klaar" to end, "sleep"/"slaap" to pause    ║
╚═══════════════════════════════════════════════════════════════════════════╝{R}
{CYAN}┌───────────────────────────────────────────────────────────────────────────┐{R}
{CYAN}│{R} {B}{YELLOW}CONTROL:{R}   sleep/slaap → 💤   "{wake_phrase}" → ✍️   stop/klaar → exit  {CYAN}│{R}
{CYAN}├───────────────────────────────────────────────────────────────────────────┤{R}
{CYAN}│{R} {B}{YELLOW}KEYBOARD:{R}  backspace  delete  enter  tab  space  home  end           {CYAN}│{R}
{CYAN}│{R}            scroll up/down  page up/down  ("5 backspaces", "10 scrolls") {CYAN}│{R}
{CYAN}├───────────────────────────────────────────────────────────────────────────┤{R}
{CYAN}│{R} {B}{RED}CAPS:{R}      "toggle caps lock" → LED aan/uit (persistent)              {CYAN}│{R}
{CYAN}│{R}            "this is a test IN ALL CAPS" → inline UPPERCASE             {CYAN}│{R}
{CYAN}│{R}            "dit is een test IN HOOFDLETTERS" → inline UPPERCASE        {CYAN}│{R}
{CYAN}├───────────────────────────────────────────────────────────────────────────┤{R}
{CYAN}│{R} {B}{GREEN}SPELL:{R}     "spell mode"/"spelmode" → letters only mode               {CYAN}│{R}
{CYAN}│{R}            capital/lower alpha  NATO alphabet  A B C = abc             {CYAN}│{R}
{CYAN}│{R}            "stop spell mode"/"stop spelmode" → exit spell mode         {CYAN}│{R}
{CYAN}├───────────────────────────────────────────────────────────────────────────┤{R}
{CYAN}│{R} {B}{MAGENTA}CASE:{R}      capital X → X    lowercase X → x    all caps X → X      {CYAN}│{R}
{CYAN}├───────────────────────────────────────────────────────────────────────────┤{R}
{CYAN}│{R} {B}{WHITE}PUNCTUATION:{R} period  comma  question mark  exclamation mark         {CYAN}│{R}
{CYAN}│{R}              colon  semicolon  new paragraph  space                    {CYAN}│{R}
{CYAN}└───────────────────────────────────────────────────────────────────────────┘{R}
""")

    def _dictation_loop(self):
        """Main dictation loop."""
        wake_phrase = WAKE_WORD.replace('_', ' ').title()

        while True:
            # Listen for speech (extended listen for longer dictation)
            text = whisper_speech_to_text(self._device, self._samplerate, extended_listen=True)

            if not text:
                continue

            text_lower = text.lower().strip().rstrip('.,!?')
            print(log_dictate(f"Heard: '{text}'"))

            # Check for stop command FIRST (before hallucination filter)
            if text_lower in ["stop", "klaar", "done", "einde"]:
                speak(_msg("end"))
                return

            if self._is_stop_command(text_lower):
                speak(_msg("stop"))
                return

            # Check for sleep command - TRUE silence using wake word detection
            if self._is_sleep_command(text_lower):
                print(log_dictate(f"💤 SLEEPING - mic silenced, say '{wake_phrase}' to wake"))
                speak(_msg("sleep"), interruptable=False)

                # Extra delay to let speaker audio fully dissipate and mic settle
                import time
                time.sleep(1.5)
                print(log_dictate("[DEBUG] Post-TTS delay complete, starting wake word detection"))

                # Use wake word detection with higher threshold for sleep mode
                # warmup_delay=2.0 ignores first 2 seconds to avoid false triggers
                detected = listen_for_wake_word(self._device, self._samplerate, warmup_delay=2.0)

                if detected:
                    print(log_dictate("✍️ Dictation resumed"))
                    speak(_msg("resume"), interruptable=False)
                else:
                    # Wake word detection failed/error - end dictation
                    speak(_msg("end"))
                    return

                continue  # Back to main dictation loop

            # Filter hallucinations
            if is_hallucination(text, strict=False):
                print(log_dictate(f"Skipped hallucination: '{text}'"))
                continue

            # Language switching (only on short commands)
            word_count = len(text_lower.split())
            if word_count <= 4:
                # Debug: show what was heard
                print(log_dictate(f"[DEBUG] Short command: '{text_lower}'"))

                # Common Whisper mishearings for "Dutch"
                dutch_mishearings = ["ditch", "touch", "such", "much", "douche", "deutsch",
                                     "neder lands", "nether lands", "need a lands"]
                if any(m in text_lower for m in dutch_mishearings):
                    set_language("nl")
                    speak(_msg("nl"), interruptable=False)
                    print(log_dictate("[Corrected mishearing to Dutch]"))
                    continue

                if word_count <= 3:
                    if any(t in text_lower for t in LANG_SWITCH_EN):
                        set_language("en")
                        speak(_msg("en"), interruptable=False)
                        continue
                    if any(t in text_lower for t in LANG_SWITCH_NL):
                        set_language("nl")
                        speak(_msg("nl"), interruptable=False)
                        continue
                    if any(t in text_lower for t in LANG_SWITCH_AUTO):
                        set_language(None)
                        speak(_msg("auto"), interruptable=False)
                        continue

            # Check for help command
            if text_lower in ["help", "commands"]:
                self._show_help()
                continue

            # === SPELL MODE TOGGLE ===
            # Check END first! "stop spelmode" contains "spelmode" so would match start otherwise
            if self._is_spell_mode_end(text_lower):
                self._spell_mode = False
                self._show_spell_mode_indicator(False)
                continue

            if self._is_spell_mode_start(text_lower):
                self._spell_mode = True
                self._show_spell_mode_indicator(True)
                continue

            # Process grammar and type (with spell mode awareness)
            self._process_and_type(text)

    def _is_stop_command(self, text_lower: str) -> bool:
        """Check if text is a stop command."""
        stop_phrases = [
            "stop dictation", "stop dictating", "end dictation",
            "exit dictation", "quit dictation", "stop typing",
            "stop dicteer", "einde dicteer", "klaar met dicteren"
        ]
        return any(phrase in text_lower for phrase in stop_phrases)

    def _is_sleep_command(self, text_lower: str) -> bool:
        """Check if text is a sleep command."""
        for sleep_word in DICTATE_SLEEP_WORDS:
            if sleep_word.lower() in text_lower:
                return True
        return False

    def _is_spell_mode_start(self, text_lower: str) -> bool:
        """Check if text starts spell mode."""
        start_phrases = [
            "spell mode", "spelmode", "spel mode", "spelling mode",
            "start spelling", "begin spelling", "letters mode"
        ]
        return any(phrase in text_lower for phrase in start_phrases)

    def _is_spell_mode_end(self, text_lower: str) -> bool:
        """Check if text ends spell mode."""
        end_phrases = [
            # English
            "end spell mode", "stop spell mode", "exit spell mode",
            "stop spelling", "end spelling", "normal mode", "exit spelling",
            # Dutch
            "stop spelmode", "einde spelmode", "uit spelmode",
            "stop spel mode", "einde spel mode", "uit spel mode",
            # Mixed/short
            "spelmode uit", "spel mode uit", "spell off", "spelling off"
        ]
        return any(phrase in text_lower for phrase in end_phrases)

    def _show_spell_mode_indicator(self, active: bool):
        """Show spell mode indicator in terminal (no TTS)."""
        R = "\033[0m"
        B = "\033[1m"
        BG_GREEN = "\033[42m"
        BG_RED = "\033[41m"
        WHITE = "\033[97m"

        if active:
            print(f"\n{B}{WHITE}{BG_GREEN}  🔤 SPELL MODE ACTIVE - say 'stop spelmode' to exit  {R}\n")
        else:
            print(f"\n{B}{WHITE}{BG_RED}  ✍️  SPELL MODE OFF - normal dictation  {R}\n")

    def _process_and_type(self, text: str):
        """Process text through grammar and type it."""
        # Strip trailing punctuation from short commands (Whisper adds "." to single words)
        word_count = len(text.split())
        if word_count <= 2:
            text = text.rstrip('.,!?')

        # If in spell mode, pre-process letters THEN use existing grammar for keys
        if self._spell_mode:
            # Step 1: Convert letters (A B C → abc, NATO → letters)
            spelled = self._convert_spell_letters(text)
            log_dictate(f"Spell converted: '{text}' → '{spelled}'")

            # Step 2: Use EXISTING grammar processor for keyboard actions
            # This handles backspace, enter, tab, space, etc.
            self._grammar.set_emojis_enabled(DICTATE_EMOJIS)
            processed, had_key_actions = self._grammar.process(spelled)

            if processed:
                log_dictate(f"Spell typing: {processed}")
                try:
                    subprocess.run(
                        ["xdotool", "type", "--clearmodifiers", "--", processed],
                        check=True, timeout=5
                    )
                except Exception as e:
                    log_dictate(f"Typing error: {e}")
            return

        # Normal dictation - apply grammar processing
        self._grammar.set_emojis_enabled(DICTATE_EMOJIS)
        processed, had_key_actions = self._grammar.process(text)

        if not processed and not had_key_actions:
            return

        if processed:
            # Add leading space if text starts with capital (new sentence after previous)
            if processed[0].isupper() and not processed[0].isdigit():
                processed = " " + processed

            # Type the text using xdotool
            log_dictate(f"Typing: {processed}")
            try:
                subprocess.run(
                    ["xdotool", "type", "--clearmodifiers", "--", processed],
                    check=True,
                    timeout=5
                )
            except subprocess.TimeoutExpired:
                log_dictate("Typing timed out")
            except subprocess.CalledProcessError as e:
                log_dictate(f"Typing error: {e}")
            except FileNotFoundError:
                log_dictate("xdotool not installed")
                speak(_msg("xdotool_missing"))

    def _convert_spell_letters(self, text: str) -> str:
        """
        Convert letters in spell mode - ONLY letter conversion.

        Keyboard actions (backspace, space, etc.) are handled by the
        existing grammar processor - no duplication!

        Handles:
        - Single letters: A, B, C → abc
        - Stuck letters: ABCD → abcd (splits them!)
        - NATO alphabet: alpha, bravo → ab
        - Case modifiers: capital/kapitaal A → A
        - Numbers: one, twee → 12
        - Preserves keyboard action words for grammar processor
        """
        from assistmint.core.constants import NATO_ALPHABET, NUMBER_WORDS, KEY_ACTIONS, NUM_WORDS_COUNT
        import re

        # Clean input - remove commas, dots
        text = text.replace(',', ' ').replace('.', ' ')
        text = ' '.join(text.split())

        # Fix stuck Whisper mishearings of "hoofdletter"
        mishearing_patterns = [
            (r'hoogletta([a-z])', r'hoofdletter \1'),
            (r'hoogsta([a-z])', r'hoofdletter \1'),
            (r'hoogta([a-z])', r'hoofdletter \1'),
            (r'hoogletter([a-z])', r'hoofdletter \1'),
            (r'hoofletter([a-z])', r'hoofdletter \1'),
            (r'hoodletter([a-z])', r'hoofdletter \1'),
            (r'hoofdletter([a-z])', r'hoofdletter \1'),
        ]
        for pattern, replacement in mishearing_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Split stuck letters but preserve known words
        expanded = []
        for word in text.split():
            wl = word.lower()
            # Keep known words intact (NATO, modifiers, KEY ACTIONS, numbers)
            if (wl in NATO_ALPHABET or wl in NUMBER_WORDS or wl in KEY_ACTIONS or
                wl in NUM_WORDS_COUNT or
                wl in ["capital", "kapitaal", "upper", "uppercase", "hoofdletter",
                       "hoogletta", "hoogsta", "hoogta", "hoogletter", "hoofletter",
                       "small", "lower", "lowercase", "kleine", "klein", "letter",
                       "number", "digit", "cijfer"]):
                expanded.append(word)
            # Split stuck letters (ABCD → A B C D)
            elif len(word) > 1 and word.isalpha() and word.isupper():
                expanded.extend(list(word))
            else:
                expanded.append(word)

        # Convert letters, preserve keyboard actions as-is
        result = []
        next_upper = False
        next_lower = False
        caps_lock = self._grammar._caps_lock

        i = 0
        while i < len(expanded):
            word = expanded[i]
            wl = word.lower()

            # Case modifiers
            if wl in ["capital", "kapitaal", "upper", "uppercase", "hoofdletter",
                      "hoogletta", "hoogsta", "hoogta", "hoogletter", "hoofletter"]:
                next_upper = True
                i += 1
                continue
            if wl in ["small", "lower", "lowercase", "kleine", "klein"]:
                next_lower = True
                i += 1
                continue
            if wl == "letter":
                i += 1
                continue

            # Single letter → convert
            if len(wl) == 1 and wl.isalpha():
                letter = wl.upper() if (next_upper or caps_lock) else wl
                if next_lower:
                    letter = wl.lower()
                next_upper = next_lower = False
                result.append(letter)
                i += 1
                continue

            # NATO → letter
            if wl in NATO_ALPHABET:
                letter = NATO_ALPHABET[wl]
                if next_upper or caps_lock:
                    letter = letter.upper()
                if next_lower:
                    letter = letter.lower()
                next_upper = next_lower = False
                result.append(letter)
                i += 1
                continue

            # Number word → digit
            if wl in NUMBER_WORDS:
                result.append(NUMBER_WORDS[wl])
                i += 1
                continue

            # PRESERVE keyboard actions - grammar processor handles these!
            if wl in KEY_ACTIONS or wl in NUM_WORDS_COUNT:
                result.append(" " + word + " ")  # Add spaces so grammar can find it
                i += 1
                continue

            # Unknown - pass through
            result.append(word)
            i += 1

        return ''.join(result)

    def _show_help(self):
        """Show dictation help."""
        help_text = """Dictation commands:
        'stop dictation' - exit dictation mode
        'sleep' or 'pause' - pause dictation, wake with wake word
        'backspace' - delete character
        'enter' or 'new line' - press enter
        'tab' - press tab
        'period', 'comma', 'question mark' - punctuation
        'capital X' - uppercase next word
        NATO alphabet for spelling: alpha, bravo, charlie, etc.
        Numbers: 'number one', 'digit 5', etc.
        """
        speak("Dictation commands. Say 'stop dictation' to exit. Say 'sleep' to pause.")
        print(help_text)

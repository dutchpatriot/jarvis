"""
Dictation Grammar - Text transformation rules.

Handles:
- NATO alphabet spelling
- Punctuation commands
- Case transformations
- Keyboard actions
- Emoji replacements
"""

import re
import subprocess
from typing import Tuple

from assistmint.core.constants import (
    NATO_ALPHABET, NUMBER_WORDS, NUM_WORDS_COUNT,
    KEY_ACTIONS, PUNCTUATION, EMOJI_MAP, SCROLL_ACTIONS
)
from assistmint.core.logger import dictate

# Import configurable triggers and settings
try:
    from config import INLINE_CAPS_TRIGGERS, TOGGLE_CAPS_TRIGGERS, SCROLL_REVERSE
except ImportError:
    INLINE_CAPS_TRIGGERS = ["in all caps", "in caps", "all caps"]
    TOGGLE_CAPS_TRIGGERS = ["toggle caps lock", "toggle caps"]
    SCROLL_REVERSE = True


class GrammarProcessor:
    """
    Processes dictation text according to grammar rules.

    Transforms voice commands into typed text.
    """

    def __init__(self, emojis_enabled: bool = False):
        self._emojis_enabled = emojis_enabled
        self._caps_lock = False  # Caps lock state

    def set_emojis_enabled(self, enabled: bool):
        """Enable or disable emoji replacements."""
        self._emojis_enabled = enabled

    def set_caps_lock(self, enabled: bool):
        """Set caps lock state."""
        self._caps_lock = enabled
        dictate(f"Caps Lock: {'ON' if enabled else 'OFF'}")

    def process(self, text: str) -> Tuple[str, bool]:
        """
        Process text through grammar rules.

        Args:
            text: Raw transcribed text

        Returns:
            Tuple of (processed_text, had_key_actions)
        """
        had_key_actions = False
        text_lower = text.lower()

        # INLINE CAPS: "this is a test in all caps" → "THIS IS A TEST"
        # Triggers from config.py - add your own!
        for trigger in INLINE_CAPS_TRIGGERS:
            if trigger in text_lower:
                idx = text_lower.find(trigger)
                before_trigger = text[:idx].strip()
                after_trigger = text[idx + len(trigger):].strip()

                # Prefer text BEFORE trigger (natural speech: "this is a test in all caps")
                # Fall back to text AFTER trigger ("in all caps this is a test")
                content = before_trigger if before_trigger else after_trigger
                if content:
                    # Remove trailing punctuation from before_trigger
                    content = content.rstrip('.,!?')
                    text = content.upper()
                    dictate(f"Inline caps: {text}")
                    return text, True
                break

        # REAL Caps Lock key toggle (LED lights up!) - then continue with rest of text
        # Triggers from config.py - add Whisper mishearings as you find them
        for cmd in TOGGLE_CAPS_TRIGGERS:
            if cmd in text_lower:
                subprocess.run(["xdotool", "key", "Caps_Lock"], check=False)
                self._caps_lock = not self._caps_lock
                dictate(f"Caps Lock toggled → {'ON' if self._caps_lock else 'OFF'}")
                # Strip command (simple replace, no regex)
                text = text_lower.replace(cmd, '').strip()
                had_key_actions = True
                break

        # Software caps lock - then continue with rest of text
        for cmd in ["caps lock on", "caps on", "hoofdletters aan", "all caps on"]:
            if cmd in text_lower:
                self.set_caps_lock(True)
                text = re.sub(r'\b' + re.escape(cmd) + r'\b', '', text, flags=re.IGNORECASE)
                text_lower = text.lower()
                break
        for cmd in ["caps lock off", "caps off", "hoofdletters uit", "all caps off"]:
            if cmd in text_lower:
                self.set_caps_lock(False)
                text = re.sub(r'\b' + re.escape(cmd) + r'\b', '', text, flags=re.IGNORECASE)
                text_lower = text.lower()
                break

        # FIX: Split stuck-together letters (ABCD → A, B, C, D)
        # Whisper sometimes outputs "ABCD" instead of "A, B, C, D"
        text = self._split_stuck_letters(text)

        # SPELL MODE - Check if input is primarily NATO alphabet words
        # Count NATO words vs regular words
        words = text.lower().split()
        nato_matches = []
        non_nato_words = []

        for word in words:
            clean_word = word.strip('.,!?')
            if clean_word in NATO_ALPHABET:
                nato_matches.append(clean_word)
            elif clean_word in ["upper", "capital", "kapitaal", "hoofdletter", "lower", "small", "kleine", "letter"]:
                pass  # Skip modifiers
            elif clean_word not in NUMBER_WORDS and clean_word not in ["number", "digit", "cijfer"]:
                non_nato_words.append(word)

        # If mostly NATO alphabet, process as spelling (no spaces between letters)
        if len(nato_matches) >= 2 and len(nato_matches) > len(non_nato_words):
            result = self._process_spelling(text)
            return result, False

        # Process case instructions for words (+ common Whisper mishearings of "hoofdletter")
        text = re.sub(
            r'\b(capital|kapitaal|uppercase|hoofdletter|hoogletta|hoogsta|hoogta|hoogletter|hoofletter|hoodletter)\s+(\w+)',
            lambda m: m.group(2).upper(),
            text, flags=re.IGNORECASE
        )
        text = re.sub(
            r'\b(lowercase|small|kleine letter|kleine|klein)\s+(\w+)',
            lambda m: m.group(2).lower(),
            text, flags=re.IGNORECASE
        )
        text = re.sub(
            r'\ball caps\s+(\w+)',
            lambda m: m.group(1).upper(),
            text, flags=re.IGNORECASE
        )

        # Single NATO alphabet words (mixed with regular text)
        for word, letter in NATO_ALPHABET.items():
            text = re.sub(
                r'\b(upper|capital|hoofdletter)\s+' + word + r'\b',
                lambda m, l=letter: l.upper(),
                text, flags=re.IGNORECASE
            )
            text = re.sub(
                r'\b(lower|kleine)\s+' + word + r'\b',
                lambda m, l=letter: l,
                text, flags=re.IGNORECASE
            )
        for word, letter in NATO_ALPHABET.items():
            text = re.sub(
                r'\bletter\s+' + word + r'\b',
                lambda m, l=letter: l,
                text, flags=re.IGNORECASE
            )

        # Direct letter spelling
        text = re.sub(
            r'\b(upper|capital|hoofdletter)\s+([a-z])\b',
            lambda m: m.group(2).upper(),
            text, flags=re.IGNORECASE
        )
        text = re.sub(
            r'\b(lower|kleine)\s+([a-z])\b',
            lambda m: m.group(2).lower(),
            text, flags=re.IGNORECASE
        )

        # Numbers
        for word, digit in NUMBER_WORDS.items():
            text = re.sub(
                r'\b(number|digit|cijfer)\s+' + word + r'\b',
                lambda m, d=digit: d,
                text, flags=re.IGNORECASE
            )
        text = re.sub(
            r'\b(number|digit|cijfer)\s+(\d)\b',
            lambda m: m.group(2),
            text, flags=re.IGNORECASE
        )

        # Keyboard actions - process and execute
        text, had_actions = self._process_key_actions(text)
        had_key_actions = had_key_actions or had_actions

        # Scroll actions - process and execute
        text, had_scroll = self._process_scroll_actions(text)
        had_key_actions = had_key_actions or had_scroll

        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text).strip()

        # Punctuation and grammar
        for word, symbol in PUNCTUATION.items():
            text = re.sub(
                r'\b' + re.escape(word) + r'\b',
                lambda m, s=symbol: s,
                text, flags=re.IGNORECASE
            )

        # Emoji replacements (if enabled)
        if self._emojis_enabled:
            for word, emoji in EMOJI_MAP.items():
                text = re.sub(
                    r'\b' + word + r'\b',
                    emoji,
                    text, flags=re.IGNORECASE
                )

        # Clean up again
        text = text.strip()

        # Fix missing spaces after punctuation (Whisper often omits them)
        text = re.sub(r'([.!?])([A-Za-z0-9])', r'\1 \2', text)
        # Fix lowercase/digit followed by capital (new sentence)
        text = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', text)

        # Apply caps lock - uppercase all text if enabled
        if self._caps_lock and text:
            text = text.upper()
            dictate(f"Caps applied: {text}")

        return text, had_key_actions

    def _split_stuck_letters(self, text: str) -> str:
        """
        Split stuck-together uppercase letters into separate letters.

        Whisper sometimes outputs "ABCD" instead of "A, B, C, D".
        This splits "ABCD" into "A B C D" but preserves normal words.

        Rules:
        - Only split if ALL CAPS and 2-6 letters (likely spelled letters)
        - Preserve words like "NATO", "USA", "PDF" (common acronyms in context)
        - Preserve normal words like "Hello", "the", etc.
        """
        words = text.split()
        result = []

        for word in words:
            # Strip punctuation for checking
            clean = word.strip('.,!?')

            # Check if it's ALL CAPS and 2-6 letters (likely stuck letters)
            # But NOT if it looks like a real acronym in a sentence
            if (clean.isupper() and
                clean.isalpha() and
                2 <= len(clean) <= 6 and
                clean not in ["OK", "TV", "PC", "CD", "DVD", "USB", "PDF", "USA", "UK", "EU", "NL"]):
                # Split into individual letters with spaces
                split_letters = ' '.join(list(clean))
                # Preserve any trailing punctuation
                trailing = word[len(clean):] if len(word) > len(clean) else ""
                result.append(split_letters + trailing)
                dictate(f"Split letters: '{clean}' → '{split_letters}'")
            else:
                result.append(word)

        return ' '.join(result)

    def _process_spelling(self, text: str) -> str:
        """
        Process NATO alphabet spelling - letters without spaces.

        Examples:
            "alpha bravo charlie" → "abc"
            "capital alpha bravo" → "Ab" (first letter capital)
            With caps lock on: "alpha bravo" → "AB"
        """
        words = text.lower().split()
        result = []
        next_upper = False
        next_lower = False

        for word in words:
            clean_word = word.strip('.,!?')

            # Check for case modifiers (+ Whisper mishearings)
            if clean_word in ["upper", "capital", "kapitaal", "hoofdletter", "uppercase",
                              "hoogletta", "hoogsta", "hoogta", "hoogletter", "hoofletter", "hoodletter"]:
                next_upper = True
                continue
            if clean_word in ["lower", "small", "kleine", "klein", "lowercase"]:
                next_lower = True
                continue
            if clean_word == "letter":
                continue  # Skip "letter" prefix

            # Check if it's a NATO word
            if clean_word in NATO_ALPHABET:
                letter = NATO_ALPHABET[clean_word]

                # Apply case
                if next_upper:
                    letter = letter.upper()
                    next_upper = False
                elif next_lower:
                    letter = letter.lower()
                    next_lower = False
                elif self._caps_lock:
                    letter = letter.upper()

                result.append(letter)

            # Check if it's a number word
            elif clean_word in NUMBER_WORDS:
                result.append(NUMBER_WORDS[clean_word])

            # Check for digit prefix
            elif clean_word in ["number", "digit", "cijfer"]:
                continue  # Skip, next word will be the number

        spelled = ''.join(result)
        dictate(f"Spelled: {spelled}")
        return spelled

    def _process_key_actions(self, text: str) -> Tuple[str, bool]:
        """Process and execute keyboard actions."""
        had_actions = False

        # Handle numbered key actions: "three backspaces", "5 tabs"
        for num_word, num_val in NUM_WORDS_COUNT.items():
            for key_word, key_name in KEY_ACTIONS.items():
                pattern = r'\b' + num_word + r'\s+' + key_word + r'\b'
                if re.search(pattern, text, flags=re.IGNORECASE):
                    for _ in range(num_val):
                        subprocess.run(["xdotool", "key", key_name], check=False)
                    dictate(f"Key: {key_name} x{num_val}")
                    text = re.sub(pattern, '', text, flags=re.IGNORECASE)
                    had_actions = True

        # Handle digit + key: "3 backspaces"
        for key_word, key_name in KEY_ACTIONS.items():
            pattern = r'\b(\d+)\s+' + key_word + r'\b'
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                num_val = int(match.group(1))
                for _ in range(num_val):
                    subprocess.run(["xdotool", "key", key_name], check=False)
                dictate(f"Key: {key_name} x{num_val}")
                text = re.sub(pattern, '', text, flags=re.IGNORECASE)
                had_actions = True

        # Handle single key actions
        for word, key in KEY_ACTIONS.items():
            pattern = r'\b' + word + r'\b'
            if re.search(pattern, text, flags=re.IGNORECASE):
                count = len(re.findall(pattern, text, flags=re.IGNORECASE))
                for _ in range(count):
                    subprocess.run(["xdotool", "key", key], check=False)
                    dictate(f"Key: {key}")
                text = re.sub(pattern, '', text, flags=re.IGNORECASE)
                had_actions = True

        return text, had_actions

    def _process_scroll_actions(self, text: str) -> Tuple[str, bool]:
        """Process and execute scroll actions using xdotool click."""
        had_actions = False
        text_lower = text.lower()

        # Apply scroll direction reversal if configured
        def get_button(btn):
            if SCROLL_REVERSE:
                return 5 if btn == 4 else 4  # Swap 4 and 5
            return btn

        # Handle numbered scroll actions: "five scroll ups", "3 page downs"
        for num_word, num_val in NUM_WORDS_COUNT.items():
            for scroll_phrase, button in SCROLL_ACTIONS.items():
                # Pattern: "five scroll ups" or "five times scroll up"
                pattern = r'\b' + num_word + r'\s+(?:times?\s+)?' + re.escape(scroll_phrase) + r's?\b'
                if re.search(pattern, text, flags=re.IGNORECASE):
                    actual_btn = get_button(button)
                    for _ in range(num_val):
                        subprocess.run(["xdotool", "click", str(actual_btn)], check=False)
                    dictate(f"Scroll: button {actual_btn} x{num_val}")
                    text = re.sub(pattern, '', text, flags=re.IGNORECASE)
                    had_actions = True

        # Handle digit + scroll: "5 scroll ups", "3 page downs"
        for scroll_phrase, button in SCROLL_ACTIONS.items():
            pattern = r'\b(\d+)\s+(?:times?\s+)?' + re.escape(scroll_phrase) + r's?\b'
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                actual_btn = get_button(button)
                num_val = int(match.group(1))
                for _ in range(num_val):
                    subprocess.run(["xdotool", "click", str(actual_btn)], check=False)
                dictate(f"Scroll: button {actual_btn} x{num_val}")
                text = re.sub(pattern, '', text, flags=re.IGNORECASE)
                had_actions = True

        # Handle single scroll actions
        for phrase, button in SCROLL_ACTIONS.items():
            pattern = r'\b' + re.escape(phrase) + r'\b'
            if re.search(pattern, text, flags=re.IGNORECASE):
                actual_btn = get_button(button)
                count = len(re.findall(pattern, text, flags=re.IGNORECASE))
                for _ in range(count):
                    subprocess.run(["xdotool", "click", str(actual_btn)], check=False)
                    dictate(f"Scroll: button {actual_btn}")
                text = re.sub(pattern, '', text, flags=re.IGNORECASE)
                had_actions = True

        return text, had_actions


# Global instance
_grammar_processor = None


def get_grammar_processor() -> GrammarProcessor:
    """Get global grammar processor instance."""
    global _grammar_processor
    if _grammar_processor is None:
        _grammar_processor = GrammarProcessor()
    return _grammar_processor


def process_grammar(text: str, emojis_enabled: bool = False) -> str:
    """Process text through grammar rules (convenience function)."""
    processor = get_grammar_processor()
    processor.set_emojis_enabled(emojis_enabled)
    result, _ = processor.process(text)
    return result

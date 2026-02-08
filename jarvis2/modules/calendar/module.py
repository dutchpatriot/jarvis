"""
Calendar Module - Calendar management via Evolution, Google, or local file.

Handles:
- Add events
- Check events
- Remove events
- Clear calendar
"""

from typing import Optional, List

from assistmint.core.modules.base import BaseModule, ModuleResult, ModuleContext, ModuleCapability
from assistmint.core.audio.tts import speak, get_language
from assistmint.core.audio.stt import whisper_speech_to_text
from assistmint.core.logger import cmd

import json
import re


def _llm_extract(prompt: str, lang: str = None) -> str:
    """
    Simple LLM call for extraction (no history, no TTS).

    Uses the centralized system prompts from config based on language.
    """
    import requests
    try:
        from config import (
            DEFAULT_MODEL, SYSTEM_PROMPT, SYSTEM_PROMPT_NL,
            OLLAMA_API_URL, OLLAMA_PARSE_TIMEOUT,
            EXTRACTION_MAX_TOKENS, EXTRACTION_TEMPERATURE
        )
    except ImportError:
        DEFAULT_MODEL = "qwen2.5:7b"
        SYSTEM_PROMPT = "You are a helpful assistant."
        SYSTEM_PROMPT_NL = "Je bent een behulpzame assistent. Antwoord alleen in het Nederlands."
        OLLAMA_API_URL = "http://localhost:11434"
        OLLAMA_PARSE_TIMEOUT = 15
        EXTRACTION_MAX_TOKENS = 300
        EXTRACTION_TEMPERATURE = 0.1

    # Use centralized system prompt based on language (from config.py)
    if lang == "nl":
        system_prompt = SYSTEM_PROMPT_NL  # Dutch: "Je bent Nederlander..."
    elif lang == "en":
        system_prompt = SYSTEM_PROMPT     # English: "You are a helpful voice assistant..."
    else:
        # Default to English for unknown languages
        system_prompt = SYSTEM_PROMPT

    url = f"{OLLAMA_API_URL}/v1/chat/completions"
    payload = {
        "model": DEFAULT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "max_tokens": EXTRACTION_MAX_TOKENS,
        "temperature": EXTRACTION_TEMPERATURE
    }

    try:
        response = requests.post(url, json=payload, timeout=OLLAMA_PARSE_TIMEOUT)
        if response.status_code == 200:
            return response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception as e:
        print(cmd(f"LLM call failed: {e}"))
    return ""


def _extract_calendar_with_llm(text: str, lang: str = "en") -> dict:
    """
    Use LLM to extract calendar event details from natural language.

    Returns dict with: event, date, start, end, location, description, reminder
    Any field can be None if not found.

    Args:
        text: User input text
        lang: Language ('en' or 'nl') - determines system prompt
    """
    # Ensure text ends with space (LLM parsing issue fix)
    text = text.strip() + " "

    # Bilingual extraction prompt - STRICT: use EXACT values from input!
    if lang == "nl":
        prompt = f"""Extraheer agenda-afspraak details uit deze tekst. Geef ALLEEN geldige JSON terug.

Tekst: "{text}"

BELANGRIJK: Gebruik de EXACTE waarden uit de tekst! NIET omzetten!
- Als user zegt "20 januari 2026" → date: "20 januari 2026" (NIET "volgende week")
- Als user zegt "17:00" → start: "17:00" (NIET "5 uur")
- Als user zegt "twee uur van tevoren" → reminder: 120 (2 uur = 120 minuten)

Extraheer:
- event: naam/titel van de afspraak
- date: EXACTE datum zoals genoemd (20 januari, morgen, maandag, etc.)
- start: EXACTE starttijd zoals genoemd (17:00, 3 uur, half 4)
- end: eindtijd (null als niet genoemd)
- location: locatie/adres (null als niet genoemd)
- description: extra notities (null als niet genoemd)
- reminder: GETAL in minuten (1 uur = 60, 2 uur = 120, 30 min = 30)

JSON:"""
    else:
        prompt = f"""Extract calendar event details from this text. Return ONLY valid JSON.

Text: "{text}"

IMPORTANT: Use EXACT values from text! DO NOT convert!
- If user says "January 20 2026" → date: "January 20 2026" (NOT "next week")
- If user says "5pm" → start: "5pm" (NOT "17:00")
- If user says "two hours before" → reminder: 120 (2 hours = 120 minutes)

Extract:
- event: event name/title
- date: EXACT date as stated (January 20, tomorrow, monday, etc.)
- start: EXACT start time as stated (5pm, 15:00, half 4)
- end: end time (null if not mentioned)
- location: location/address (null if not mentioned)
- description: notes (null if not mentioned)
- reminder: NUMBER in minutes (1 hour = 60, 2 hours = 120, 30 min = 30)

JSON:"""

    response = _llm_extract(prompt, lang=lang)

    # Check if LLM call failed
    if not response:
        print(cmd("LLM extraction failed: no response"))
        return {"event": None, "date": None, "start": None, "end": None, "location": None, "description": None, "reminder": 30, "_llm_failed": True}

    print(cmd(f"LLM extraction: {response[:100]}..."))

    # Extract JSON from response
    try:
        json_match = re.search(r'\{[^{}]+\}', response)
        if json_match:
            data = json.loads(json_match.group())
            return {
                "event": data.get("event"),
                "date": data.get("date"),
                "start": data.get("start"),
                "end": data.get("end"),
                "location": data.get("location"),
                "description": data.get("description"),
                "reminder": data.get("reminder", 30)  # Default 30 min reminder
            }
    except json.JSONDecodeError as e:
        print(cmd(f"JSON parse failed: {e}"))

    return {"event": None, "date": None, "start": None, "end": None, "location": None, "description": None, "reminder": 30}

# Import config values
try:
    from config import (
        CALENDAR_MAX_RETRIES, CALENDAR_CANCEL_WORDS, CALENDAR_PROMPTS,
        CALENDAR_ASK_LANGUAGE, CALENDAR_LANG_EN, CALENDAR_LANG_NL,
        CALENDAR_WORDS, CALENDAR_REMOVE_PREFIXES, CALENDAR_CHECK_WORDS,
        CALENDAR_CLEAR_WORDS, CALENDAR_REMOVE_WORDS, CALENDAR_ADD_WORDS,
        CALENDAR_REMOVE_ALL_PHRASES
    )
except ImportError:
    CALENDAR_MAX_RETRIES = 2
    CALENDAR_CANCEL_WORDS = ["cancel", "stop", "never mind"]
    CALENDAR_PROMPTS = {}
    CALENDAR_ASK_LANGUAGE = False
    CALENDAR_LANG_EN = ["english", "engels"]
    CALENDAR_LANG_NL = ["dutch", "nederlands"]
    CALENDAR_WORDS = ["calendar", "agenda", "event", "meeting", "appointment", "afspraak"]
    CALENDAR_REMOVE_PREFIXES = ("remove ", "delete ", "verwijder ", "wis ")
    CALENDAR_CHECK_WORDS = ["what", "check", "show", "bekijk", "toon"]
    CALENDAR_CLEAR_WORDS = ["clear", "delete all", "wis alles"]
    CALENDAR_REMOVE_WORDS = ["remove event", "verwijder afspraak"]
    CALENDAR_ADD_WORDS = ["add", "create", "voeg toe"]
    CALENDAR_REMOVE_ALL_PHRASES = ["remove all", "alles verwijderen", "allemaal"]


class CalendarModule(BaseModule):
    """
    Calendar module for managing events.

    Supports multiple backends:
    - Evolution (GNOME/Linux)
    - Google Calendar (via gcalcli)
    - Local .reminders file
    """

    def __init__(self):
        super().__init__()
        self._device = None
        self._samplerate = 16000

    @property
    def name(self) -> str:
        return "calendar"

    @property
    def capabilities(self) -> ModuleCapability:
        return (
            ModuleCapability.TEXT_INPUT |
            ModuleCapability.TEXT_OUTPUT |
            ModuleCapability.MULTI_TURN |
            ModuleCapability.CALENDAR
        )

    @property
    def description(self) -> str:
        return "Calendar management (add, check, remove events)"

    @property
    def triggers(self) -> List[str]:
        return [
            "add to calendar", "calendar", "agenda", "schedule",
            "check calendar", "check my calendar", "what's on my calendar",
            "remove event", "delete event", "clear calendar"
        ]

    @property
    def priority(self) -> int:
        return 70  # High priority for calendar-specific requests

    def can_handle(self, text: str, intent: Optional[str] = None) -> float:
        """Check if this looks like a calendar request."""
        text_lower = text.lower()

        # Check for explicit calendar intent
        if intent in ["add_calendar", "check_calendar", "clear_calendar", "remove_calendar"]:
            return 1.0

        # Check for calendar keywords (using config.py values)
        has_cal = any(w in text_lower for w in CALENDAR_WORDS)

        # Also check if starts with remove prefix (e.g., "Remove meeting...")
        starts_with_remove = text_lower.startswith(CALENDAR_REMOVE_PREFIXES)
        if starts_with_remove:
            return 0.95  # High confidence for removal

        if has_cal:
            # Check for action words (using config.py values)
            if any(w in text_lower for w in CALENDAR_ADD_WORDS):
                return 0.95  # High confidence for add requests
            if any(w in text_lower for w in CALENDAR_CHECK_WORDS):
                return 0.9
            if any(w in text_lower for w in CALENDAR_CLEAR_WORDS):
                return 0.9
            if any(w in text_lower for w in CALENDAR_REMOVE_WORDS):
                return 0.9
            return 0.7

        return 0.0

    def execute(self, context: ModuleContext) -> ModuleResult:
        """Execute calendar action based on input."""
        self._device = context.selected_device
        self._samplerate = context.samplerate

        text_lower = context.text_lower

        # PRIORITY 1: Use voice2json intent if available (most accurate)
        # Get intent from context if passed through
        intent = getattr(context, 'intent', None)
        if intent:
            print(cmd(f"Calendar routing by intent: {intent}"))
            if intent == "add_calendar":
                return self._handle_add(initial_text=context.text)
            elif intent == "check_calendar":
                return self._handle_check()
            elif intent == "clear_calendar":
                return self._handle_clear()
            elif intent == "remove_calendar":
                return self._handle_remove()

        # PRIORITY 2: Keyword fallback (when no intent)
        # All trigger words are now in config.py for easy customization
        has_cal = any(w in text_lower for w in CALENDAR_WORDS)

        # Check order matters! Check/clear/remove before add (more specific first)
        # REMOVE check: "remove" or "delete" at START of sentence = removal intent
        starts_with_remove = text_lower.startswith(CALENDAR_REMOVE_PREFIXES)

        if has_cal and any(w in text_lower for w in CALENDAR_CHECK_WORDS):
            return self._handle_check()
        elif has_cal and any(w in text_lower for w in CALENDAR_CLEAR_WORDS):
            return self._handle_clear()
        elif starts_with_remove or (has_cal and any(w in text_lower for w in CALENDAR_REMOVE_WORDS)):
            # "Remove meeting..." or "Remove event..." or "Verwijder afspraak..."
            return self._handle_remove()
        elif has_cal and any(w in text_lower for w in CALENDAR_ADD_WORDS):
            return self._handle_add(initial_text=context.text)
        elif has_cal:
            # Has calendar word but no clear action - assume add
            return self._handle_add(initial_text=context.text)
        else:
            return ModuleResult(
                text="What would you like to do with your calendar?",
                success=True
            )

    def _get_prompt(self, key: str) -> str:
        """Get bilingual prompt based on current language."""
        lang = get_language()
        prompts = CALENDAR_PROMPTS.get(key, (key, key))
        return prompts[1] if lang == "nl" else prompts[0]

    def _validate_date_or_week(self, query: str):
        """
        Validate date or week query.

        Returns: (is_valid: bool, error_msg: str or None)
        """
        from assistmint.calendar_manager import parse_date

        q_lower = query.lower()
        week_keywords = ["week", "deze week", "volgende week", "week van"]
        if any(w in q_lower for w in week_keywords):
            valid_patterns = ["this week", "next week", "week of", "deze week", "volgende week", "week van"]
            if any(p in q_lower for p in valid_patterns):
                return True, None
            return False, self._get_prompt("couldnt_understand_week")
        result = parse_date(query, silent=True)
        if result:
            return True, None
        return False, self._get_prompt("couldnt_understand_date")

    def _process_week_query(self, query: str, action_func):
        """
        Process a week query and call the appropriate action function.

        Args:
            query: User's date/week query
            action_func: Either check_calendar or clear_calendar function
        """
        q_lower = query.lower()
        if any(w in q_lower for w in ["week", "deze week", "volgende week"]):
            if "this week" in q_lower or "deze week" in q_lower:
                action_func(date="this week", week=True)
            elif "next week" in q_lower or "volgende week" in q_lower:
                action_func(date="next week", week=True)
            elif "week of" in q_lower or "week van" in q_lower:
                specific_week_start = q_lower.replace("week of", "").replace("week van", "").strip()
                action_func(specific_week_start=specific_week_start, week=True)
        else:
            action_func(date=query)

    def _ask_with_retry(self, prompt_key: str, validator=None, retry_prompt_key=None):
        """Ask a question and retry if not understood."""
        from assistmint.calendar_manager import parse_time, parse_date

        attempts = 0
        max_attempts = CALENDAR_MAX_RETRIES + 1
        first_ask = True

        while attempts < max_attempts:
            prompt = self._get_prompt(prompt_key) if prompt_key in CALENDAR_PROMPTS else prompt_key
            retry_prompt = self._get_prompt(retry_prompt_key) if retry_prompt_key and retry_prompt_key in CALENDAR_PROMPTS else retry_prompt_key

            if first_ask:
                speak(prompt)
                first_ask = False
            elif attempts > 0:
                if retry_prompt:
                    speak(retry_prompt)
                else:
                    speak(f"{self._get_prompt('ask_again')} {prompt}")

            response = whisper_speech_to_text(self._device, self._samplerate).strip()

            # Check for cancel words
            if any(cancel in response.lower() for cancel in CALENDAR_CANCEL_WORDS):
                speak(self._get_prompt("cancelled"))
                return None, True

            # Validate response
            if validator:
                success, error_msg = validator(response)
                if success:
                    return response, False
                else:
                    attempts += 1
                    if attempts < max_attempts:
                        speak(error_msg if error_msg else self._get_prompt("didnt_catch"))
                    else:
                        speak(f"{error_msg} {self._get_prompt('lets_start_over')}")
                        return None, False
            else:
                if response:
                    return response, False
                attempts += 1
                if attempts < max_attempts:
                    speak(self._get_prompt("didnt_catch"))
                else:
                    speak(self._get_prompt("lets_start_over"))
                    return None, False

        return None, False

    def _handle_add(self, initial_text: str = None) -> ModuleResult:
        """
        Add event to calendar - KISS version.

        Flow:
        1. Ask "English or Dutch?" → force language for entire flow
        2. Ask "What's the event?" → user speaks naturally
        3. LLM extracts → ask missing fields if needed
        4. Confirm → add only if "yes" AND all validated
        """
        MAX_FIELD_RETRIES = 5
        print(cmd("Calendar ADD"))

        # STEP 1: Ask language ONCE at the start
        lang = self._ask_language()
        if lang is None:
            return ModuleResult(text="Cancelled.", success=False)

        # STEP 2: Proceed with the calendar flow in chosen language
        # Pass initial_text if provided (natural language with event details)
        return self._handle_add_oneshot(MAX_FIELD_RETRIES, lang, initial_text=initial_text)

    def _ask_language(self) -> Optional[str]:
        """Ask user: English or Dutch? Returns 'en' or 'nl', or None if cancelled."""
        from assistmint.core.audio.tts import set_language

        # Use English voice for this neutral question
        speak("English or Dutch?", lang="en")

        response = whisper_speech_to_text(self._device, self._samplerate).strip().lower()

        if any(cancel in response for cancel in CALENDAR_CANCEL_WORDS):
            speak("Cancelled.", lang="en")
            return None

        # Detect language choice
        dutch_words = ["dutch", "nederlands", "nl", "holland", "hollands", "nederlandstalig"]
        english_words = ["english", "engels", "en", "eng"]

        if any(w in response for w in dutch_words):
            set_language("nl")  # Set GLOBAL language so ChatModule/LLM uses Dutch system prompt
            return "nl"
        elif any(w in response for w in english_words):
            set_language("en")  # Set GLOBAL language
            return "en"
        else:
            # Default to English if unclear
            speak("I'll use English.", lang="en")
            set_language("en")
            return "en"

    def _handle_add_oneshot(self, max_retries: int, lang: str, initial_text: str = None) -> ModuleResult:
        """
        One-shot mode: LLM extracts from spoken message, validate, confirm.

        Args:
            max_retries: Max retries for field validation
            lang: Language ('en' or 'nl')
            initial_text: If provided, use this instead of asking "What's the event?"
        """
        from assistmint.calendar_manager import add_event_to_calendar_extended, parse_time, parse_date
        from datetime import timedelta

        try:
            from config import CALENDAR_DEFAULT_DURATION
        except ImportError:
            CALENDAR_DEFAULT_DURATION = 60

        # Check if we already have text with event details (natural language input)
        if initial_text and len(initial_text) > 20:
            # User already said something like "meeting on thursday at 3pm"
            # Use it directly for extraction
            user_message = initial_text
            print(cmd(f"Using initial text: {user_message}"))
        else:
            # Ask for the event
            if lang == "nl":
                speak("Wat is de afspraak?", lang="nl")
            else:
                speak("What's the event?", lang="en")

            user_message = whisper_speech_to_text(self._device, self._samplerate).strip()

            if any(cancel in user_message.lower() for cancel in CALENDAR_CANCEL_WORDS):
                speak(self._get_prompt("cancelled"), lang=lang)
                return ModuleResult(text="Cancelled.", success=False)

        print(cmd(f"One-shot input: {user_message}"))

        # LLM extraction - gets all fields (uses centralized system prompt based on lang)
        extracted = _extract_calendar_with_llm(user_message, lang=lang)
        print(cmd(f"LLM extracted: {extracted}"))

        # Check if LLM failed to respond
        if extracted.get("_llm_failed"):
            if lang == "nl":
                speak("Sorry, ik kan de AI niet bereiken.", lang="nl")
            else:
                speak("Sorry, I can't reach the AI.", lang="en")
            return ModuleResult(text="LLM unavailable", success=False)

        event_name = extracted.get("event")
        event_date = extracted.get("date")
        start_time = extracted.get("start")
        end_time = extracted.get("end")
        location = extracted.get("location")
        description = extracted.get("description")
        reminder = extracted.get("reminder", 30)

        # ===== VALIDATE and FIX missing REQUIRED fields =====
        # Event name (required)
        if not event_name:
            event_name = self._ask_field_with_retry(
                "what_event", None, max_retries, lang
            )
            if event_name is None:
                return ModuleResult(text="Cancelled.", success=False)

        # Date (required)
        validated_date = None
        if event_date:
            validated_date = parse_date(event_date, silent=True)

        if not validated_date:
            event_date = self._ask_field_with_retry(
                "what_date", "date", max_retries, lang
            )
            if event_date is None:
                return ModuleResult(text="Cancelled.", success=False)
            validated_date = parse_date(event_date, silent=True)
            if not validated_date:
                if lang == "nl":
                    speak("Ik kon de datum niet valideren. Geannuleerd.", lang="nl")
                else:
                    speak("I couldn't validate the date. Cancelled.", lang="en")
                return ModuleResult(text="Date validation failed.", success=False)

        # Start time (required)
        validated_start = None
        if start_time:
            validated_start = parse_time(start_time, silent=True)

        if not validated_start:
            start_time = self._ask_field_with_retry(
                "what_start_time", "time", max_retries, lang
            )
            if start_time is None:
                return ModuleResult(text="Cancelled.", success=False)
            validated_start = parse_time(start_time, silent=True)
            if not validated_start:
                if lang == "nl":
                    speak("Ik kon de starttijd niet valideren. Geannuleerd.", lang="nl")
                else:
                    speak("I couldn't validate the start time. Cancelled.", lang="en")
                return ModuleResult(text="Time validation failed.", success=False)

        # End time (optional - use default duration if not provided)
        validated_end = None
        if end_time:
            validated_end = parse_time(end_time, silent=True)

        if not validated_end:
            # validated_start is a string like "3:00 PM" - convert to datetime to add duration
            from datetime import datetime
            try:
                # Parse the time string (handles "3:00 PM" format)
                start_dt = datetime.strptime(validated_start, "%I:%M %p")
                end_dt = start_dt + timedelta(minutes=CALENDAR_DEFAULT_DURATION)
                validated_end = end_dt.strftime("%I:%M %p")  # e.g., "4:00 PM"
                end_time = validated_end
            except (ValueError, TypeError) as e:
                print(cmd(f"End time calculation failed: {e}"))
                # Fallback: just use start + 1 hour as string
                validated_end = validated_start
                end_time = validated_start

        # Location, description, reminder are OPTIONAL - no need to ask

        # ===== FINAL CONFIRMATION =====
        return self._confirm_and_add_extended(
            event_name, event_date, start_time, end_time,
            validated_date, validated_start, validated_end,
            location, description, reminder, lang
        )

    def _handle_add_stepbystep(self, max_retries: int) -> ModuleResult:
        """Step-by-step mode: ask each field individually with validation."""
        from assistmint.calendar_manager import add_event_to_calendar, parse_time, parse_date
        from datetime import timedelta

        try:
            from config import CALENDAR_DEFAULT_DURATION
        except ImportError:
            CALENDAR_DEFAULT_DURATION = 60

        lang = get_language()

        # ===== 1. Event name =====
        event_name = self._ask_field_with_retry("what_event", None, max_retries, lang)
        if event_name is None:
            return ModuleResult(text="Cancelled.", success=False)

        # ===== 2. Date =====
        event_date = self._ask_field_with_retry("what_date", "date", max_retries, lang)
        if event_date is None:
            return ModuleResult(text="Cancelled.", success=False)

        validated_date = parse_date(event_date, silent=True)
        if not validated_date:
            if lang == "nl":
                speak("Ik kon de datum niet valideren na meerdere pogingen. Laten we opnieuw beginnen.")
            else:
                speak("I couldn't validate the date after multiple attempts. Let's start over.")
            return ModuleResult(text="Date validation failed.", success=False)

        # ===== 3. Start time =====
        start_time = self._ask_field_with_retry("what_start_time", "time", max_retries, lang)
        if start_time is None:
            return ModuleResult(text="Cancelled.", success=False)

        validated_start = parse_time(start_time, silent=True)
        if not validated_start:
            if lang == "nl":
                speak("Ik kon de starttijd niet valideren. Laten we opnieuw beginnen.")
            else:
                speak("I couldn't validate the start time. Let's start over.")
            return ModuleResult(text="Time validation failed.", success=False)

        # ===== 4. End time (optional) =====
        if lang == "nl":
            speak("Hoe laat eindigt het? Zeg 'skip' voor standaard duur.")
        else:
            speak("What time does it end? Say 'skip' for default duration.")

        end_response = whisper_speech_to_text(self._device, self._samplerate).strip().lower()

        if any(cancel in end_response for cancel in CALENDAR_CANCEL_WORDS):
            speak(self._get_prompt("cancelled"))
            return ModuleResult(text="Cancelled.", success=False)

        if "skip" in end_response or "sla over" in end_response or "standaard" in end_response or "default" in end_response:
            # validated_start is a string like "3:00 PM" - convert to datetime to add duration
            from datetime import datetime
            try:
                start_dt = datetime.strptime(validated_start, "%I:%M %p")
                end_dt = start_dt + timedelta(minutes=CALENDAR_DEFAULT_DURATION)
                validated_end = end_dt.strftime("%I:%M %p")
                end_time = validated_end
            except (ValueError, TypeError):
                validated_end = validated_start
                end_time = validated_start
        else:
            validated_end = parse_time(end_response, silent=True)
            if validated_end:
                end_time = end_response
            else:
                # Use default duration
                from datetime import datetime
                try:
                    start_dt = datetime.strptime(validated_start, "%I:%M %p")
                    end_dt = start_dt + timedelta(minutes=CALENDAR_DEFAULT_DURATION)
                    validated_end = end_dt.strftime("%I:%M %p")
                    end_time = validated_end
                except (ValueError, TypeError):
                    validated_end = validated_start
                    end_time = validated_start

        # ===== FINAL CONFIRMATION =====
        return self._confirm_and_add(
            event_name, event_date, start_time, end_time,
            validated_date, validated_start, validated_end, lang
        )

    def _ask_field_with_retry(self, prompt_key: str, field_type: str, max_retries: int, lang: str) -> Optional[str]:
        """
        Ask for a specific field with CONFIRMATION and retry logic.

        Flow:
        1. Ask the question
        2. Listen to response
        3. Repeat back: "I heard: X. Correct?"
        4. If no → ask again
        5. If yes → return value

        Args:
            prompt_key: Key for CALENDAR_PROMPTS
            field_type: 'date', 'time', or None (no validation)
            max_retries: Maximum number of retries
            lang: Current language (FORCED for all speak calls)

        Returns:
            The validated response, or None if cancelled/failed
        """
        from assistmint.calendar_manager import parse_time, parse_date

        prompt = self._get_prompt(prompt_key)
        retry_count = 0

        while retry_count < max_retries:
            # Ask the question
            if retry_count == 0:
                speak(prompt, lang=lang)
            else:
                if lang == "nl":
                    speak("Zeg het nog een keer.", lang="nl")
                else:
                    speak("Say it again.", lang="en")

            response = whisper_speech_to_text(self._device, self._samplerate).strip()

            # Check for cancel
            if any(cancel in response.lower() for cancel in CALENDAR_CANCEL_WORDS):
                speak(self._get_prompt("cancelled"), lang=lang)
                return None

            # Empty response
            if not response:
                retry_count += 1
                if lang == "nl":
                    speak("Ik heb niets gehoord.", lang="nl")
                else:
                    speak("I didn't hear anything.", lang="en")
                continue

            # Validate if needed
            if field_type == "date":
                if not parse_date(response, silent=True):
                    retry_count += 1
                    speak(self._get_prompt("couldnt_understand_date"), lang=lang)
                    continue
            elif field_type == "time":
                if not parse_time(response, silent=True):
                    retry_count += 1
                    speak(self._get_prompt("couldnt_understand_time"), lang=lang)
                    continue

            # === CONFIRM WHAT WE HEARD ===
            if lang == "nl":
                speak(f"Ik verstond: {response}. Klopt dat?", lang="nl")
            else:
                speak(f"I heard: {response}. Is that correct?", lang="en")

            confirmation = whisper_speech_to_text(self._device, self._samplerate).strip().lower()

            # Check for cancel in confirmation
            if any(cancel in confirmation for cancel in CALENDAR_CANCEL_WORDS):
                speak(self._get_prompt("cancelled"), lang=lang)
                return None

            # Check confirmation
            yes_words = ["yes", "ja", "correct", "klopt", "yep", "yeah", "jep", "goed", "ok", "okay"]
            no_words = ["no", "nee", "niet", "fout", "wrong", "opnieuw", "again"]

            if any(w in confirmation for w in yes_words):
                return response  # Confirmed!
            elif any(w in confirmation for w in no_words):
                retry_count += 1
                continue  # Ask again
            else:
                # Unclear - assume yes if response was valid
                return response

        # Max retries reached
        if lang == "nl":
            speak(f"Na {max_retries} pogingen lukt het niet. Geannuleerd.", lang="nl")
        else:
            speak(f"After {max_retries} attempts, I couldn't understand. Cancelled.", lang="en")
        return None

    def _confirm_and_add_extended(
        self, event_name: str, event_date: str, start_time: str, end_time: str,
        validated_date, validated_start, validated_end,
        location: str, description: str, reminder: int, lang: str
    ) -> ModuleResult:
        """
        Final confirmation and add event with all fields.

        STRICT: Never adds event unless user says yes/ja AND all data is validated.
        All speak() calls use lang= to prevent voice switching.
        """
        from assistmint.calendar_manager import add_event_to_calendar_extended

        # Format confirmation message with optional fields
        if lang == "nl":
            confirm_msg = f"{event_name} op {event_date} van {start_time} tot {end_time}"
            if location:
                confirm_msg += f" bij {location}"
            confirm_msg += ". Klopt dit?"
        else:
            confirm_msg = f"{event_name} on {event_date} from {start_time} to {end_time}"
            if location:
                confirm_msg += f" at {location}"
            confirm_msg += ". Correct?"

        speak(confirm_msg, lang=lang)
        confirmation = whisper_speech_to_text(self._device, self._samplerate).strip().lower()

        print(cmd(f"Confirmation response: {confirmation}"))

        # STRICT: Only accept explicit confirmation
        yes_words = ["yes", "ja", "correct", "klopt", "yep", "yeah", "jep", "absoluut", "zeker", "prima", "goed"]
        no_words = ["no", "nee", "niet", "cancel", "stop", "annuleer", "fout", "wrong"]

        if any(w in confirmation for w in yes_words):
            # Double-check all REQUIRED fields are validated
            if validated_date and validated_start and validated_end and event_name:
                add_event_to_calendar_extended(
                    event_name, start_time, end_time,
                    date=event_date,
                    location=location,
                    description=description,
                    reminder_minutes=reminder if reminder else 30
                )
                if lang == "nl":
                    return ModuleResult(text=f"Toegevoegd: {event_name}.", success=True)
                else:
                    return ModuleResult(text=f"Added: {event_name}.", success=True)
            else:
                if lang == "nl":
                    speak("Er mist nog informatie. Geannuleerd.", lang="nl")
                else:
                    speak("Some information is still missing. Cancelled.", lang="en")
                return ModuleResult(text="Validation incomplete.", success=False)

        elif any(w in confirmation for w in no_words):
            if lang == "nl":
                speak("Oké, geannuleerd.", lang="nl")
            else:
                speak("Okay, cancelled.", lang="en")
            return ModuleResult(text="Cancelled by user.", success=False)

        else:
            # STRICT: Unclear response = DO NOT ADD
            if lang == "nl":
                speak("Niet begrepen. Geen event toegevoegd.", lang="nl")
            else:
                speak("Didn't understand. No event added.", lang="en")
            return ModuleResult(text="No clear confirmation.", success=False)

    def _confirm_and_add(
        self, event_name: str, event_date: str, start_time: str, end_time: str,
        validated_date, validated_start, validated_end, lang: str
    ) -> ModuleResult:
        """Legacy confirmation without extended fields."""
        return self._confirm_and_add_extended(
            event_name, event_date, start_time, end_time,
            validated_date, validated_start, validated_end,
            None, None, 30, lang
        )

    def _handle_check(self) -> ModuleResult:
        """Check calendar for events, then offer to add/remove."""
        from assistmint.calendar_manager import check_calendar

        print(cmd("Calendar CHECK"))
        lang = get_language()

        query, cancelled = self._ask_with_retry(
            "which_date_or_week",
            validator=self._validate_date_or_week,
            retry_prompt_key="retry_date_or_week"
        )
        if cancelled or not query:
            return ModuleResult(text="Cancelled.", success=False)

        self._process_week_query(query, check_calendar)

        # === FOLLOW-UP: Ask what user wants to do ===
        import time
        time.sleep(0.5)  # Brief pause after reading events

        if lang == "nl":
            speak("Wilt u iets toevoegen of verwijderen? Of zeg 'klaar' om te stoppen.", lang="nl")
        else:
            speak("Would you like to add or remove something? Or say 'done' to finish.", lang="en")

        response = whisper_speech_to_text(self._device, self._samplerate).strip().lower()
        print(cmd(f"Check follow-up response: {response}"))

        # Check for done/cancel
        done_words = ["done", "klaar", "nee", "no", "nothing", "niets", "stop", "cancel"]
        if any(w in response for w in done_words):
            if lang == "nl":
                return ModuleResult(text="Oké.", success=True)
            else:
                return ModuleResult(text="Okay.", success=True)

        # Check for add
        add_words = ["add", "toevoegen", "nieuw", "new", "create", "maak", "yes", "ja"]
        if any(w in response for w in add_words):
            return self._handle_add()

        # Check for remove/clear
        remove_words = ["remove", "verwijder", "delete", "wis", "clear", "leeg"]
        if any(w in response for w in remove_words):
            # Ask if they want to remove specific or clear all
            if any(w in response for w in ["all", "alles", "clear", "leeg", "wis"]):
                return self._handle_clear()
            else:
                return self._handle_remove()

        # Unclear response - just finish
        if lang == "nl":
            return ModuleResult(text="Oké, geen wijzigingen.", success=True)
        else:
            return ModuleResult(text="Okay, no changes.", success=True)

    def _handle_clear(self) -> ModuleResult:
        """Clear calendar events."""
        from assistmint.calendar_manager import clear_calendar

        print(cmd("Calendar CLEAR"))

        query, cancelled = self._ask_with_retry(
            "which_date_to_clear",
            validator=self._validate_date_or_week,
            retry_prompt_key="retry_date_or_week"
        )
        if cancelled or not query:
            return ModuleResult(text="Cancelled.", success=False)

        self._process_week_query(query, clear_calendar)

        return ModuleResult(text="", success=True)

    def _handle_remove(self) -> ModuleResult:
        """Remove specific event from calendar with interactive selection."""
        from assistmint.calendar_manager import get_events_on_date, remove_event_by_uid, parse_date

        lang = get_language()
        print(cmd(f"Calendar REMOVE (lang={lang})"))

        # Step 1: Ask for date
        def validate_date(d):
            result = parse_date(d, silent=True)
            if result:
                return True, None
            return False, self._get_prompt("couldnt_understand_date")

        event_date, cancelled = self._ask_with_retry(
            "event_date_to_remove",
            validator=validate_date,
            retry_prompt_key="retry_date"
        )
        if cancelled or not event_date:
            return ModuleResult(text="Cancelled.", success=False)

        # Step 2: Get events on that date
        events = get_events_on_date(event_date)

        if not events:
            msg = "Geen afspraken op die datum." if lang == "nl" else "No events on that date."
            speak(msg)
            return ModuleResult(text=msg, success=True)

        # Step 3: List events numbered
        print(f"[CAL] Listing {len(events)} events:")
        if lang == "nl":
            speak(f"Er zijn {len(events)} afspraken:")
        else:
            speak(f"You have {len(events)} events:")

        for i, ev in enumerate(events, 1):
            time_part = f" at {ev['time_str']}" if ev['time_str'] else ""
            if lang == "nl":
                time_part = f" om {ev['time_str']}" if ev['time_str'] else ""
            print(f"  {i}. {ev['name']}{time_part} (uid: {ev['uid'][:20]}...)")
            speak(f"{i}. {ev['name']}{time_part}")

        # Step 4: Ask which to remove
        if lang == "nl":
            speak("Welk nummer wil je verwijderen? Of zeg 'alles' voor allemaal.")
        else:
            speak("Which number would you like to remove? Or say 'all' for all.")

        response = whisper_speech_to_text(self._device, self._samplerate).strip().lower()

        if not response or response in ["cancel", "annuleer", "stop", "nevermind"]:
            speak("Cancelled." if lang == "en" else "Geannuleerd.")
            return ModuleResult(text="Cancelled.", success=False)

        # Step 5: Parse response and remove
        removed_count = 0
        print(f"[CAL] Selection response: '{response}'")

        # Check for "remove all" / "delete all" / "alles verwijderen" etc.
        # NOTE: "all" or "alles" alone is NOT enough - must be full phrase (see config.py)
        matched_all = [p for p in CALENDAR_REMOVE_ALL_PHRASES if p in response]
        if matched_all:
            print(f"[CAL] Matched 'all' phrase: {matched_all}")
            # Remove all events
            for ev in events:
                if remove_event_by_uid(ev['uid']):
                    removed_count += 1
            msg = f"Removed {removed_count} events." if lang == "en" else f"{removed_count} afspraken verwijderd."
            speak(msg)
            return ModuleResult(text=msg, success=True)

        # Parse number(s) from response
        print(f"[CAL] Raw response: '{response}'")
        numbers = re.findall(r'\b\d+\b', response)  # Word boundary for digits
        print(f"[CAL] Digit matches: {numbers}")

        # Also check for number words (EN + NL) - with word boundaries!
        word_to_num = {
            # English - longer phrases first to avoid partial matches
            "the first": 1, "the second": 2, "the third": 3, "the fourth": 4, "the fifth": 5,
            "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
            "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
            # Dutch - longer phrases first
            "nummer een": 1, "nummer twee": 2, "nummer drie": 3,
            "nummer vier": 4, "nummer vijf": 5,
            "de eerste": 1, "de tweede": 2, "de derde": 3,
            "de vierde": 4, "de vijfde": 5,
            "eerste": 1, "tweede": 2, "derde": 3, "vierde": 4, "vijfde": 5,
            "zesde": 6, "zevende": 7, "achtste": 8, "negende": 9, "tiende": 10,
            "een": 1, "twee": 2, "drie": 3, "vier": 4, "vijf": 5,
            "zes": 6, "zeven": 7, "acht": 8, "negen": 9, "tien": 10,
        }
        for word, num in word_to_num.items():
            # Use word boundary regex to avoid partial matches (e.g., "een" in "geen")
            if re.search(rf'\b{re.escape(word)}\b', response):
                numbers.append(str(num))
                print(f"[CAL] Word match: '{word}' -> {num}")

        if not numbers:
            speak("I didn't understand which event." if lang == "en" else "Ik begreep niet welke afspraak.")
            return ModuleResult(text="Not understood.", success=False)

        # Remove selected events
        print(f"[CAL] Numbers to remove: {numbers}")
        print(f"[CAL] Total events: {len(events)}")
        removed_names = []
        for num_str in set(numbers):  # Use set to avoid duplicates
            num = int(num_str)
            print(f"[CAL] Processing number {num}")
            if 1 <= num <= len(events):
                ev = events[num - 1]
                print(f"[CAL] Removing: {ev['name']} (uid: {ev['uid'][:20]}...)")
                success = remove_event_by_uid(ev['uid'])
                print(f"[CAL] Remove result: {success}")
                if success:
                    removed_count += 1
                    removed_names.append(ev['name'])
            else:
                print(f"[CAL] Number {num} out of range (1-{len(events)})")

        if removed_count > 0:
            if removed_count == 1:
                msg = f"Removed '{removed_names[0]}'." if lang == "en" else f"'{removed_names[0]}' verwijderd."
            else:
                msg = f"Removed {removed_count} events." if lang == "en" else f"{removed_count} afspraken verwijderd."
            speak(msg)
            return ModuleResult(text=msg, success=True)
        else:
            msg = "No events removed." if lang == "en" else "Geen afspraken verwijderd."
            speak(msg)
            return ModuleResult(text=msg, success=False)

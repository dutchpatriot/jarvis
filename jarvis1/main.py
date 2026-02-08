import argparse
import time
from speech_recognition import list_microphones, select_microphone_and_samplerate, get_default_microphone, whisper_speech_to_text
from assistmint.calendar_manager import (
    add_event_to_calendar, check_calendar, remove_event, clear_calendar,
    set_speak_func, set_calendar_config
)
from ollama import ask_ollama, select_ollama_model, set_model, load_session, clear_session, has_pending_calendar, clear_pending_calendar
from text_to_speech import speak, set_language, get_language
from wake_word import listen_for_wake_word, init_wake_word
from corrections import apply_corrections, add_correction, list_corrections
from voice2json_intent import recognize_intent
from colors import cmd, v2j, learn, dictate
from config import (DICTATE_EMOJIS, DICTATE_SLEEP_WORDS, WAKE_WORD,
                    LANG_SWITCH_EN, LANG_SWITCH_NL, LANG_SWITCH_AUTO,
                    CALENDAR_MAX_RETRIES, CALENDAR_CANCEL_WORDS, CALENDAR_PROMPTS,
                    CALENDAR_ASK_LANGUAGE, CALENDAR_LANG_EN, CALENDAR_LANG_NL,
                    LOG_CMD_LENGTH, LOG_OUTPUT_LENGTH,
                    CALENDAR_BACKEND, CALENDAR_ID, CALENDAR_DEFAULT_DURATION)

# Initialize calendar with V1's TTS and config
set_speak_func(speak)
set_calendar_config(
    backend=CALENDAR_BACKEND,
    calendar_id=CALENDAR_ID,
    default_duration=CALENDAR_DEFAULT_DURATION,
)

# Track last transcription for learning mode
_last_transcription = ""
# Quiet help mode (no TTS for commands)
quiet_help = False
# Use voice2json intent recognition
use_voice2json = True


def _show_dictation_help():
    """Display compact dictation help when entering dictation mode."""
    # ANSI colors
    R = "\033[0m"
    B = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    MAGENTA = "\033[95m"
    WHITE = "\033[97m"
    BG_CYAN = "\033[46m"

    wake_phrase = WAKE_WORD.replace('_', ' ').title()
    print(f"""
{B}{WHITE}{BG_CYAN}╔═══════════════════════════════════════════════════════════════════════╗
║  ✍️  DICTATION MODE - "stop"/"klaar" to end, "sleep"/"slaap" to pause  ║
╚═══════════════════════════════════════════════════════════════════════╝{R}
{CYAN}┌─────────────────────────────────────────────────────────────────────┐{R}
{CYAN}│{R} {B}{YELLOW}CONTROL:{R}   sleep/slaap/pause → 💤   "{wake_phrase}" → ✍️  (no STT) {CYAN}│{R}
{CYAN}├─────────────────────────────────────────────────────────────────────┤{R}
{CYAN}│{R} {B}{YELLOW}KEYBOARD:{R}  backspace  delete  enter  tab  ("three backspaces")  {CYAN}│{R}
{CYAN}├─────────────────────────────────────────────────────────────────────┤{R}
{CYAN}│{R} {B}{GREEN}SPELL:{R}     capital/lower alpha  letter alpha  capital/lower A   {CYAN}│{R}
{CYAN}│{R}            number 5 / digit five   (NATO + names: alpha/albert)   {CYAN}│{R}
{CYAN}├─────────────────────────────────────────────────────────────────────┤{R}
{CYAN}│{R} {B}{MAGENTA}CASE:{R}      capital X → X    lowercase X → x    all caps X → X    {CYAN}│{R}
{CYAN}├─────────────────────────────────────────────────────────────────────┤{R}
{CYAN}│{R} {B}{WHITE}PUNCTUATION:{R} period  comma  question mark  exclamation mark       {CYAN}│{R}
{CYAN}│{R}              colon  semicolon  new paragraph  space               {CYAN}│{R}
{CYAN}├─────────────────────────────────────────────────────────────────────┤{R}
{CYAN}│{R} {B}{WHITE}SYMBOLS:{R}   at sign @   hashtag #   slash /   backslash \\          {CYAN}│{R}
{CYAN}│{R}            underscore _   hyphen -   quote "   single quote '     {CYAN}│{R}
{CYAN}│{R}            open/close parenthesis ( )  bracket [ ]  brace {{ }}    {CYAN}│{R}
{CYAN}├─────────────────────────────────────────────────────────────────────┤{R}
{CYAN}│{R} {B}{WHITE}EMOJIS:{R}    house 🏠  heart ❤️  smile 😊  sun ☀️  star ⭐  fire 🔥     {CYAN}│{R}
{CYAN}│{R}            dog 🐕  cat 🐈  coffee ☕  pizza 🍕  + many more        {CYAN}│{R}
{CYAN}├─────────────────────────────────────────────────────────────────────┤{R}
{CYAN}│{R} {B}{YELLOW}LANGUAGE:{R} "English" / "Nederlands" / "Dutch" → switch voice        {CYAN}│{R}
{CYAN}│{R}            (only works on 1-3 word commands, not in sentences)   {CYAN}│{R}
{CYAN}├─────────────────────────────────────────────────────────────────────┤{R}
{CYAN}│{R} {DIM}NATO: alpha/albert bravo/boy charlie delta/david echo foxtrot/fox{R}{CYAN}│{R}
{CYAN}│{R} {DIM}      golf/george hotel/henry india juliet/john kilo/king lima{R} {CYAN}│{R}
{CYAN}│{R} {DIM}      mike november oscar papa quebec/queen romeo sierra/sam{R}   {CYAN}│{R}
{CYAN}│{R} {DIM}      tango/tom uniform/uncle victor whiskey xray yankee zulu{R}  {CYAN}│{R}
{CYAN}└─────────────────────────────────────────────────────────────────────┘{R}
""")


def _show_help():
    """Display help menu."""
    # ANSI colors
    R = "\033[0m"       # Reset
    B = "\033[1m"       # Bold
    DIM = "\033[2m"     # Dim
    # Foreground
    WHITE = "\033[97m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    MAGENTA = "\033[95m"
    BLUE = "\033[94m"
    # Background
    BG_BLUE = "\033[44m"
    BG_BLACK = "\033[40m"

    print(f"""
{B}{WHITE}{BG_BLUE}╔══════════════════════════════════════════════════════════════════════════════════════╗
║                    {GREEN}★ ASSISTMINT ★{WHITE}  Voice Assistant                                  ║
║            {DIM}Whisper STT │ Voice2json Intent │ Ollama LLM{R}{B}{WHITE}{BG_BLUE}                           ║
╚══════════════════════════════════════════════════════════════════════════════════════╝{R}

{B}{CYAN}  ENGLISH                              NEDERLANDS{R}
{DIM}  ───────────────────────────────────────────────────────────────────────{R}

{YELLOW}  📅 CALENDAR{R}                           {YELLOW}📅 KALENDER{R}
     {WHITE}"Add to calendar"{R}                     {WHITE}"Voeg toe aan agenda"{R}
     {WHITE}"Check my calendar"{R}                   {WHITE}"Bekijk mijn agenda"{R}
     {WHITE}"Remove event"{R}                        {WHITE}"Verwijder afspraak"{R}
     {WHITE}"Clear my calendar"{R}                   {WHITE}"Wis mijn agenda"{R}

{GREEN}  🧠 SESSION{R}                            {GREEN}🧠 SESSIE{R}
     {WHITE}"Clear session"{R}                       {WHITE}"Vergeet alles"{R}
     {WHITE}"Forget everything"{R}                   {WHITE}"Wis sessie"{R}

{MAGENTA}  🎓 LEARNING{R}                           {MAGENTA}🎓 LEREN{R}
     {WHITE}"Learn that"{R} / {WHITE}"Correct that"{R}         {WHITE}"Leer dat"{R} / {WHITE}"Corrigeer dat"{R}
     {WHITE}"Show corrections"{R}                    {WHITE}"Toon correcties"{R}

{CYAN}  ✍️  DICTATION{R}                          {CYAN}✍️  DICTATIE{R}
     {WHITE}"Dictate"{R} → {DIM}"Stop" to end{R}             {WHITE}"Dicteer"{R} → {DIM}"Stop" / "Klaar"{R}

{BLUE}  💻 TERMINAL{R}                            {BLUE}💬 QUESTIONS{R}
     {WHITE}"Run command"{R} / {WHITE}"Terminal"{R}            {DIM}Just ask anything → Ollama{R}

{DIM}  ═══════════════════════════════════════════════════════════════════════{R}
  {B}{WHITE}DICTATION GRAMMAR{R}
  {YELLOW}Keyboard:{R}  {DIM}"backspace" "delete" "enter" "tab" + "three backspaces" "5 tabs"{R}
  {YELLOW}Case:{R}      {DIM}"capital X" "lowercase X" "all caps X" "hoofdletter X"{R}
  {YELLOW}Spell:{R}     {DIM}"capital/lower A" "letter alpha" "capital alpha" "number 5" "cijfer vijf"{R}
  {YELLOW}Punct:{R}     {DIM}"period/punt" "comma/komma" "question mark" "exclamation mark"{R}
  {YELLOW}Format:{R}    {DIM}"new paragraph" "space/spatie"{R}
  {YELLOW}Symbols:{R}   {DIM}"at sign" "hashtag" "slash" "backslash" "underscore" "hyphen" "asterisk"{R}
  {YELLOW}Brackets:{R}  {DIM}"open/close parenthesis" "open/close bracket" "open/close brace"{R}
  {YELLOW}Quotes:{R}    {DIM}"quote" "single quote" "aanhalingsteken" "apostrof"{R}
  {YELLOW}Emojis:{R}    {DIM}"house" 🏠  "heart" ❤️  "smile" 😊  "sun" ☀️  "fire" 🔥 + many more{R}
{DIM}  ═══════════════════════════════════════════════════════════════════════{R}
  {B}{WHITE}TERMINAL SPELL MODE{R}
  {DIM}Example: "letter lima letter sierra space hyphen letter lima letter alpha" → ls -la{R}
  {DIM}Or names: "letter london letter sam space hyphen letter london letter albert" → ls -la{R}
{DIM}  ═══════════════════════════════════════════════════════════════════════{R}
  {B}{WHITE}NATO + NAMES{R}
  {DIM}alpha/albert bravo/boy charlie delta/david echo foxtrot/fox golf/george{R}
  {DIM}hotel/henry india juliet/john kilo/king lima/london mike/michael november{R}
  {DIM}oscar papa/peter quebec/queen romeo/roger sierra/sam tango/tom uniform{R}
  {DIM}victor whiskey/william xray yankee/yellow zulu/zebra{R}
{DIM}  ═══════════════════════════════════════════════════════════════════════{R}
""")
    if not quiet_help:
        speak("Calendar: add to calendar, check my agenda. Session: clear session or vergeet alles. Dictation: say dictate to type. Or just ask me anything.")


def _handle_learn_correction(selected_device, samplerate):
    """Handle learning mode - correct misrecognitions."""
    global _last_transcription
    if _last_transcription:
        print(learn("Learning mode activated"))
        speak("What should it be?")
        correct_phrase = whisper_speech_to_text(selected_device, samplerate).strip()
        if correct_phrase:
            add_correction(_last_transcription, correct_phrase)
            speak(f"Got it. I'll remember that {_last_transcription} means {correct_phrase}.")
        else:
            speak("Sorry, I didn't catch that.")
    else:
        speak("Nothing to correct yet. Say something first.")


def _handle_show_corrections():
    """Show stored corrections."""
    corrections = list_corrections()
    if corrections:
        speak(f"You have {len(corrections)} corrections stored.")
    else:
        speak("No corrections stored yet.")


def _get_prompt(key):
    """Get bilingual prompt based on current language setting.

    Returns the Dutch version if language is set to 'nl', otherwise English.
    """
    lang = get_language()
    prompts = CALENDAR_PROMPTS.get(key, (key, key))  # Fallback to key itself
    return prompts[1] if lang == "nl" else prompts[0]


def _ask_calendar_language(selected_device, samplerate):
    """Ask user which language they want to use for this calendar action.

    Returns True if language was set, False if cancelled.
    Only asks if CALENDAR_ASK_LANGUAGE is True.
    """
    if not CALENDAR_ASK_LANGUAGE:
        return True  # Skip, use current language

    speak(_get_prompt("which_language"))
    response = whisper_speech_to_text(selected_device, samplerate).strip().lower()

    # Check for cancel
    if any(cancel in response for cancel in CALENDAR_CANCEL_WORDS):
        speak(_get_prompt("cancelled"))
        return False

    # Detect language choice
    if any(keyword in response for keyword in CALENDAR_LANG_NL):
        set_language("nl")
        speak("Nederlands.", interruptable=False)
    elif any(keyword in response for keyword in CALENDAR_LANG_EN):
        set_language("en")
        speak("English.", interruptable=False)
    # else: keep current language (user might have said something unclear, proceed anyway)

    return True


def _check_language_switch(response):
    """Check if response is a language switch command and apply it.

    Returns True if language was switched (caller should re-ask), False otherwise.
    Only triggers on short responses (1-3 words) to avoid accidental switches.
    """
    response_lower = response.lower()
    word_count = len(response_lower.split())

    # Only switch on short isolated commands
    if word_count > 3:
        return False

    # Check for English switch
    if any(cmd in response_lower for cmd in LANG_SWITCH_EN):
        set_language("en")
        speak("Switched to English.", interruptable=False)
        return True

    # Check for Dutch switch
    if any(cmd in response_lower for cmd in LANG_SWITCH_NL):
        set_language("nl")
        speak("Nederlands.", interruptable=False)
        return True

    # Check for auto switch
    if any(cmd in response_lower for cmd in LANG_SWITCH_AUTO):
        set_language(None)
        speak("Auto.", interruptable=False)
        return True

    return False


def _ask_with_retry(prompt_key, selected_device, samplerate, validator=None, retry_prompt_key=None):
    """Ask a question and retry if not understood.

    Args:
        prompt_key: Key for CALENDAR_PROMPTS or direct text
        selected_device: Microphone device
        samplerate: Audio sample rate
        validator: Optional function(response) -> (success, error_msg)
                   If None, any non-empty response is accepted
        retry_prompt_key: Key for retry prompt in CALENDAR_PROMPTS

    Returns:
        (response, cancelled) tuple - response is the valid answer, cancelled=True if user said cancel
    """
    attempts = 0
    max_attempts = CALENDAR_MAX_RETRIES + 1  # +1 for initial attempt
    first_ask = True

    while attempts < max_attempts:
        # Get prompts fresh each time (language may have changed)
        prompt = _get_prompt(prompt_key) if prompt_key in CALENDAR_PROMPTS else prompt_key
        retry_prompt = _get_prompt(retry_prompt_key) if retry_prompt_key and retry_prompt_key in CALENDAR_PROMPTS else retry_prompt_key

        if first_ask:
            speak(prompt)
            first_ask = False
        elif attempts > 0:
            if retry_prompt:
                speak(retry_prompt)
            else:
                speak(f"{_get_prompt('ask_again')} {prompt}")

        response = whisper_speech_to_text(selected_device, samplerate).strip()

        # Check for language switch commands (don't count as attempt)
        if _check_language_switch(response):
            first_ask = True  # Re-ask in new language
            continue

        # Check for cancel words
        if any(cancel in response.lower() for cancel in CALENDAR_CANCEL_WORDS):
            speak(_get_prompt("cancelled"))
            return None, True

        # Validate response
        if validator:
            success, error_msg = validator(response)
            if success:
                return response, False
            else:
                attempts += 1
                if attempts < max_attempts:
                    speak(error_msg if error_msg else _get_prompt("didnt_catch"))
                else:
                    speak(f"{error_msg} {_get_prompt('lets_start_over')}")
                    return None, False
        else:
            # No validator - accept any non-empty response
            if response:
                return response, False
            attempts += 1
            if attempts < max_attempts:
                speak(_get_prompt("didnt_catch"))
            else:
                speak(_get_prompt("lets_start_over"))
                return None, False

    return None, False


def _handle_add_calendar(selected_device, samplerate):
    """Add event to calendar with retry logic."""
    from calendar_manager import parse_time, parse_date

    print(cmd("Calendar ADD"))

    # Ask for language preference
    if not _ask_calendar_language(selected_device, samplerate):
        return

    # Ask for event name (no validation needed)
    event_name, cancelled = _ask_with_retry(
        "what_event",
        selected_device, samplerate
    )
    if cancelled or not event_name:
        return

    # Ask for start time with validation
    def validate_time(t):
        result = parse_time(t, silent=True)
        if result:
            return True, None
        return False, _get_prompt("couldnt_understand_time")

    start_time, cancelled = _ask_with_retry(
        "what_start_time",
        selected_device, samplerate,
        validator=validate_time,
        retry_prompt_key="retry_time"
    )
    if cancelled or not start_time:
        return

    # Ask for end time with validation
    end_time, cancelled = _ask_with_retry(
        "what_end_time",
        selected_device, samplerate,
        validator=validate_time,
        retry_prompt_key="retry_time"
    )
    if cancelled or not end_time:
        return

    # Ask for date with validation
    def validate_date(d):
        result = parse_date(d, silent=True)
        if result:
            return True, None
        return False, _get_prompt("couldnt_understand_date")

    event_date, cancelled = _ask_with_retry(
        "what_date",
        selected_device, samplerate,
        validator=validate_date,
        retry_prompt_key="retry_date"
    )
    if cancelled or not event_date:
        return

    add_event_to_calendar(event_name, start_time, end_time, date=event_date)


def _handle_check_calendar(selected_device, samplerate):
    """Check calendar for events with retry logic."""
    from calendar_manager import parse_date

    print(cmd("Calendar CHECK"))

    # Ask for language preference
    if not _ask_calendar_language(selected_device, samplerate):
        return

    def validate_date_or_week(q):
        q_lower = q.lower()
        # Week queries are always valid (support both EN and NL)
        week_keywords = ["week", "deze week", "volgende week", "week van"]
        if any(w in q_lower for w in week_keywords):
            valid_patterns = ["this week", "next week", "week of", "deze week", "volgende week", "week van"]
            if any(p in q_lower for p in valid_patterns):
                return True, None
            return False, _get_prompt("couldnt_understand_week")
        # Otherwise validate as a date
        result = parse_date(q, silent=True)
        if result:
            return True, None
        return False, _get_prompt("couldnt_understand_date")

    query, cancelled = _ask_with_retry(
        "which_date_or_week",
        selected_device, samplerate,
        validator=validate_date_or_week,
        retry_prompt_key="retry_date_or_week"
    )
    if cancelled or not query:
        return

    q_lower = query.lower()
    if any(w in q_lower for w in ["week", "deze week", "volgende week"]):
        if "this week" in q_lower or "deze week" in q_lower:
            check_calendar(date="this week", week=True)
        elif "next week" in q_lower or "volgende week" in q_lower:
            check_calendar(date="next week", week=True)
        elif "week of" in q_lower or "week van" in q_lower:
            specific_week_start = q_lower.replace("week of", "").replace("week van", "").strip()
            check_calendar(specific_week_start=specific_week_start, week=True)
    else:
        check_calendar(date=query)


def _handle_clear_calendar(selected_device, samplerate):
    """Clear calendar events with retry logic."""
    from calendar_manager import parse_date

    print(cmd("Calendar CLEAR"))

    # Ask for language preference
    if not _ask_calendar_language(selected_device, samplerate):
        return

    def validate_date_or_week(q):
        q_lower = q.lower()
        week_keywords = ["week", "deze week", "volgende week", "week van"]
        if any(w in q_lower for w in week_keywords):
            valid_patterns = ["this week", "next week", "week of", "deze week", "volgende week", "week van"]
            if any(p in q_lower for p in valid_patterns):
                return True, None
            return False, _get_prompt("couldnt_understand_week")
        result = parse_date(q, silent=True)
        if result:
            return True, None
        return False, _get_prompt("couldnt_understand_date")

    query, cancelled = _ask_with_retry(
        "which_date_to_clear",
        selected_device, samplerate,
        validator=validate_date_or_week,
        retry_prompt_key="retry_date_or_week"
    )
    if cancelled or not query:
        return

    q_lower = query.lower()
    if any(w in q_lower for w in ["week", "deze week", "volgende week"]):
        if "this week" in q_lower or "deze week" in q_lower:
            clear_calendar(date="this week", week=True)
        elif "next week" in q_lower or "volgende week" in q_lower:
            clear_calendar(date="next week", week=True)
        elif "week of" in q_lower or "week van" in q_lower:
            specific_week_start = q_lower.replace("week of", "").replace("week van", "").strip()
            clear_calendar(date=specific_week_start, week=True)
    else:
        clear_calendar(date=query)


def _handle_remove_calendar(selected_device, samplerate):
    """Remove specific event from calendar with retry logic."""
    from calendar_manager import parse_date

    print(cmd("Calendar REMOVE"))

    # Ask for language preference
    if not _ask_calendar_language(selected_device, samplerate):
        return

    # Ask for event name (no validation needed)
    event_name, cancelled = _ask_with_retry(
        "event_name_to_remove",
        selected_device, samplerate
    )
    if cancelled or not event_name:
        return

    # Ask for date with validation
    def validate_date(d):
        result = parse_date(d, silent=True)
        if result:
            return True, None
        return False, _get_prompt("couldnt_understand_date")

    event_date, cancelled = _ask_with_retry(
        "event_date_to_remove",
        selected_device, samplerate,
        validator=validate_date,
        retry_prompt_key="retry_date"
    )
    if cancelled or not event_date:
        return

    remove_event(event_name, event_date)


def _handle_dictation(selected_device, samplerate):
    """Handle dictation mode - type text into active window."""
    import subprocess
    import re

    _show_dictation_help()
    speak("Dictating. Say stop to end.")

    # Collect all dictated text for summary
    transcript = []

    # NATO phonetic alphabet (+ common Whisper mishearings)
    nato = {
        "alpha": "a", "alfa": "a", "albert": "a",
        "bravo": "b", "beta": "b", "boy": "b",
        "charlie": "c", "charles": "c",
        "delta": "d", "david": "d",
        "echo": "e", "edward": "e",
        "foxtrot": "f", "fox": "f", "frank": "f",
        "golf": "g", "george": "g",
        "hotel": "h", "henry": "h",
        "india": "i", "indigo": "i",
        "juliet": "j", "julia": "j", "john": "j",
        "kilo": "k", "king": "k",
        "lima": "l", "london": "l", "louis": "l",
        "mike": "m", "michael": "m", "mary": "m",
        "november": "n", "nancy": "n", "nora": "n",
        "oscar": "o", "oliver": "o",
        "papa": "p", "peter": "p", "paul": "p",
        "quebec": "q", "queen": "q",
        "romeo": "r", "robert": "r", "roger": "r",
        "sierra": "s", "sugar": "s", "sam": "s",
        "tango": "t", "tom": "t", "tommy": "t",
        "uniform": "u", "uncle": "u",
        "victor": "v", "victoria": "v",
        "whiskey": "w", "whisky": "w", "william": "w",
        "xray": "x", "x-ray": "x",
        "yankee": "y", "yellow": "y", "young": "y",
        "zulu": "z", "zebra": "z", "zero letter": "z",
    }

    # Number words for spelling
    number_words = {
        "zero": "0", "nul": "0", "one": "1", "een": "1", "two": "2", "twee": "2",
        "three": "3", "drie": "3", "four": "4", "vier": "4", "five": "5", "vijf": "5",
        "six": "6", "zes": "6", "seven": "7", "zeven": "7", "eight": "8", "acht": "8",
        "nine": "9", "negen": "9"
    }

    # Number words for key repetition
    num_words = {
        "one": 1, "een": 1, "two": 2, "twee": 2, "three": 3, "drie": 3,
        "four": 4, "vier": 4, "five": 5, "vijf": 5, "six": 6, "zes": 6,
        "seven": 7, "zeven": 7, "eight": 8, "acht": 8, "nine": 9, "negen": 9,
        "ten": 10, "tien": 10
    }

    # Keyboard actions
    key_actions = {
        "backspace": "BackSpace", "backspaces": "BackSpace", "wissen": "BackSpace",
        "delete": "Delete", "deletes": "Delete", "verwijderen": "Delete",
        "enter": "Return", "enters": "Return", "nieuwe regel": "Return", "new line": "Return",
        "tab": "Tab", "tabs": "Tab", "tabje": "Tab",
    }

    # Punctuation and symbols
    replacements = {
        "period": ".", "punt": ".", "point": ".",
        "comma": ",", "komma": ",",
        "question mark": "?", "vraagteken": "?",
        "exclamation mark": "!", "uitroepteken": "!",
        "colon": ":", "dubbele punt": ":",
        "semicolon": ";", "puntkomma": ";",
        "new paragraph": "\n\n", "nieuwe paragraaf": "\n\n",
        "space": " ", "spatie": " ",
        "at sign": "@", "apenstaartje": "@",
        "hashtag": "#", "hash": "#",
        "dollar sign": "$", "dollar": "$",
        "percent": "%", "procent": "%",
        "ampersand": "&", "en teken": "&",
        "asterisk": "*", "sterretje": "*",
        "underscore": "_", "liggend streepje": "_",
        "hyphen": "-", "min": "-", "dash": "-",
        "slash": "/", "schuine streep": "/",
        "backslash": "\\",
        "open parenthesis": "(", "haakje openen": "(",
        "close parenthesis": ")", "haakje sluiten": ")",
        "open bracket": "[", "close bracket": "]",
        "open brace": "{", "close brace": "}",
        "quote": '"', "aanhalingsteken": '"',
        "single quote": "'", "apostrof": "'",
    }

    # Emoji map
    emoji_map = {
        # Objects (+ plurals)
        "house": "🏠", "houses": "🏠", "home": "🏡", "homes": "🏡",
        "car": "🚗", "cars": "🚗", "phone": "📱", "phones": "📱",
        "computer": "💻", "computers": "💻", "book": "📖", "books": "📖",
        "clock": "🕐", "clocks": "🕐", "calendar": "📅", "mail": "📧", "email": "📧",
        "camera": "📷", "cameras": "📷", "music": "🎵", "movie": "🎬", "movies": "🎬",
        "key": "🔑", "keys": "🔑", "light": "💡", "lights": "💡",
        "money": "💰", "gift": "🎁", "gifts": "🎁", "balloon": "🎈", "balloons": "🎈",
        "rocket": "🚀", "rockets": "🚀", "plane": "✈️", "planes": "✈️",
        "train": "🚂", "trains": "🚂", "bus": "🚌", "bicycle": "🚲", "bicycles": "🚲",
        "boat": "⛵", "boats": "⛵", "umbrella": "☂️", "umbrellas": "☂️",
        # People & body
        "heart": "❤️", "hearts": "❤️", "love": "💕", "kiss": "💋", "kisses": "💋",
        "hand": "✋", "hands": "✋", "thumbs up": "👍", "thumbs down": "👎",
        "clap": "👏", "wave": "👋", "pray": "🙏", "muscle": "💪", "muscles": "💪",
        "eye": "👁️", "eyes": "👁️", "brain": "🧠", "baby": "👶", "babies": "👶",
        "man": "👨", "men": "👨", "woman": "👩", "women": "👩",
        # Faces
        "smile": "😊", "smiles": "😊", "laugh": "😂", "wink": "😉", "cry": "😢", "sad": "😢",
        "angry": "😠", "cool": "😎", "thinking": "🤔", "surprised": "😮", "love face": "😍",
        "sick": "🤒", "sleepy": "😴", "crazy": "🤪", "devil": "😈", "angel": "😇",
        # Animals (+ plurals)
        "dog": "🐕", "dogs": "🐕", "cat": "🐈", "cats": "🐈",
        "bird": "🐦", "birds": "🐦", "fish": "🐟", "butterfly": "🦋", "butterflies": "🦋",
        "bee": "🐝", "bees": "🐝", "pig": "🐷", "pigs": "🐷", "cow": "🐄", "cows": "🐄",
        "horse": "🐴", "horses": "🐴", "monkey": "🐵", "monkeys": "🐵",
        "elephant": "🐘", "elephants": "🐘", "lion": "🦁", "lions": "🦁",
        "tiger": "🐯", "tigers": "🐯", "bear": "🐻", "bears": "🐻",
        "rabbit": "🐰", "rabbits": "🐰", "snake": "🐍", "snakes": "🐍",
        "frog": "🐸", "frogs": "🐸", "chicken": "🐔", "chickens": "🐔",
        "penguin": "🐧", "penguins": "🐧", "whale": "🐋", "whales": "🐋",
        # Food & drink (+ plurals)
        "apple": "🍎", "apples": "🍎", "banana": "🍌", "bananas": "🍌",
        "orange": "🍊", "oranges": "🍊", "pizza": "🍕", "pizzas": "🍕",
        "burger": "🍔", "burgers": "🍔", "coffee": "☕", "beer": "🍺", "beers": "🍺",
        "wine": "🍷", "cake": "🎂", "cakes": "🎂", "ice cream": "🍦",
        "cookie": "🍪", "cookies": "🍪", "bread": "🍞", "cheese": "🧀",
        "egg": "🥚", "eggs": "🥚", "chicken leg": "🍗",
        # Nature & weather (+ plurals)
        "sun": "☀️", "moon": "🌙", "star": "⭐", "stars": "⭐",
        "cloud": "☁️", "clouds": "☁️", "rain": "🌧️", "snow": "❄️",
        "fire": "🔥", "rainbow": "🌈", "rainbows": "🌈",
        "flower": "🌸", "flowers": "🌸", "tree": "🌳", "trees": "🌳",
        "leaf": "🍃", "leaves": "🍃", "earth": "🌍", "ocean": "🌊",
        "mountain": "⛰️", "mountains": "⛰️", "thunder": "⚡",
        # Symbols
        "check": "✓", "checkmark": "✓", "cross": "✗", "warning": "⚠️", "stop sign": "🛑",
        "arrow": "➡️", "sparkle": "✨", "sparkles": "✨", "diamond": "💎", "diamonds": "💎",
        "crown": "👑", "crowns": "👑", "trophy": "🏆", "trophies": "🏆",
        "medal": "🏅", "medals": "🏅", "flag": "🚩", "flags": "🚩",
        "lock": "🔒", "bell": "🔔", "bells": "🔔", "magnifier": "🔍",
        # Dutch words (+ plurals)
        "huis": "🏠", "huizen": "🏠", "auto": "🚗", "autos": "🚗",
        "telefoon": "📱", "telefoons": "📱", "hart": "❤️", "harten": "❤️",
        "lach": "😊", "zon": "☀️", "maan": "🌙", "ster": "⭐", "sterren": "⭐",
        "bloem": "🌸", "bloemen": "🌸", "boom": "🌳", "bomen": "🌳",
        "hond": "🐕", "honden": "🐕", "kat": "🐈", "katten": "🐈",
        "vogel": "🐦", "vogels": "🐦", "vis": "🐟", "vissen": "🐟",
        "vuur": "🔥", "regen": "🌧️", "sneeuw": "❄️",
        "koffie": "☕", "bier": "🍺", "wijn": "🍷", "boek": "📖", "boeken": "📖",
    }

    # Known Whisper hallucinations (generated on silence/noise/mumbling)
    whisper_hallucinations = [
        # YouTube-style hallucinations
        "you", "thank you", "thanks for watching", "thank you for watching",
        "subscribe", "like and subscribe", "see you next time", "bye",
        "thanks", "thank you so much", "you you", "you you you",
        "thank you thank you", "thank you thank you thank you",
        "you you you you", "thanks thanks", "thanks thanks thanks",
        # Dutch TV/media hallucinations
        "tv gelderland", "tv gelderland 2021", "tv gelderland 2020", "tv gelderland 2019",
        "nos journaal", "rtl nieuws", "omroep gelderland", "omroep brabant",
        "ondertiteling", "ondertiteling tuvalu", "ondertitels", "copyright",
        # Single words / fillers
        "the", "a", "i", "it", "so", "and", "but", "or", "um", "uh", "oh",
        "hmm", "hm", "ah", "eh", "er", "mm", "mhm", "yeah", "yep", "nope",
        # Apologies (common hallucination)
        "i'm sorry", "sorry", "my apologies", "excuse me", "pardon",
        # Music/sound descriptions
        "music", "music playing", "applause", "laughter", "silence",
        "background music", "upbeat music", "soft music",
        # Repeated phrases
        "all right", "alright", "okay okay", "yes yes", "no no",
        # Mumbling artifacts
        "blah", "blah blah", "la la", "da da", "na na",
        # Empty acknowledgments
        "got it", "i see", "right", "right right", "sure", "sure sure",
    ]

    def is_hallucination(t):
        """Check if text is likely a Whisper hallucination."""
        t_lower = t.lower().strip().rstrip('.,!?')
        # Check known hallucinations
        if t_lower in whisper_hallucinations:
            return True
        # Check for repeated single word (e.g., "You You You")
        words = t_lower.split()
        if len(words) >= 2 and len(set(words)) == 1:
            return True
        # Check for repeated phrases (e.g., "I'm sorry. I'm sorry. I'm sorry.")
        # Split by sentence-ending punctuation and check if phrases repeat
        phrases = [p.strip() for p in re.split(r'[.!?]+', t_lower) if p.strip()]
        if len(phrases) >= 2 and len(set(phrases)) == 1:
            return True
        # Check for very short meaningless output
        if len(t_lower) <= 2 and t_lower not in ["ok"]:
            return True
        return False

    while True:
        text = whisper_speech_to_text(selected_device, samplerate, extended_listen=True).strip()

        if not text:
            continue

        # Skip Whisper hallucinations
        if is_hallucination(text):
            print(dictate(f"Skipped hallucination: '{text}'"))
            continue

        # === CHECK CONTROL COMMANDS FIRST (before any text processing) ===
        # Normalize: lowercase, strip whitespace and common punctuation
        raw_lower = text.lower().strip().rstrip('.,!?')

        # Stop command - exit dictation
        if raw_lower in ["stop", "klaar", "done", "einde"]:
            speak("Dictation ended.")
            break

        # Sleep mode - TRUE silence using wake word detection (no Whisper = no transcription)
        if raw_lower in DICTATE_SLEEP_WORDS:
            wake_phrase = WAKE_WORD.replace('_', ' ').title()
            print(dictate(f"💤 SLEEPING - mic silenced, say '{wake_phrase}' to wake"))
            speak("Sleeping.", interruptable=False)

            # Use wake word detection - NO transcription while sleeping
            detected = listen_for_wake_word(selected_device, samplerate)

            if detected:
                print(dictate("✍️ Dictation resumed"))
                speak("Resumed.", interruptable=False)
            else:
                # Wake word detection failed/error - end dictation
                speak("Dictation ended.")
                return

            continue  # Back to main dictation loop

        # Language switching (only on short isolated commands, not in sentences)
        dictate_word_count = len(raw_lower.split())

        # Debug: show what was heard for short commands
        if dictate_word_count <= 4:
            print(dictate(f"[DEBUG] Short command heard: '{raw_lower}'"))

        # Common Whisper mishearings for "Dutch" / "Nederlands"
        dutch_mishearings = ["ditch", "touch", "such", "much", "douche", "deutsch",
                             "neder lands", "nether lands", "need a lands"]
        if any(m in raw_lower for m in dutch_mishearings):
            set_language("nl")
            speak("Nederlands.", interruptable=False)
            print(dictate(f"[Corrected mishearing to Dutch]"))
            continue

        if dictate_word_count <= 3:
            if any(t in raw_lower for t in LANG_SWITCH_EN):
                set_language("en")
                speak("Switched to English.", interruptable=False)
                continue
            if any(t in raw_lower for t in LANG_SWITCH_NL):
                set_language("nl")
                speak("Nederlands.", interruptable=False)
                continue
            if any(t in raw_lower for t in LANG_SWITCH_AUTO):
                set_language(None)
                speak("Auto.", interruptable=False)
                continue

        # === PROCESS ALL TRANSFORMATIONS ===

        # Process case instructions for words
        text = re.sub(r'\b(capital|uppercase|hoofdletter)\s+(\w+)', lambda m: m.group(2).upper(), text, flags=re.IGNORECASE)
        text = re.sub(r'\b(lowercase|kleine letter)\s+(\w+)', lambda m: m.group(2).lower(), text, flags=re.IGNORECASE)
        text = re.sub(r'\ball caps\s+(\w+)', lambda m: m.group(1).upper(), text, flags=re.IGNORECASE)

        # SPELL MODE - NATO alphabet
        for word, letter in nato.items():
            text = re.sub(r'\b(upper|capital|hoofdletter)\s+' + word + r'\b', lambda m, l=letter: l.upper(), text, flags=re.IGNORECASE)
            text = re.sub(r'\b(lower|kleine)\s+' + word + r'\b', lambda m, l=letter: l, text, flags=re.IGNORECASE)
        for word, letter in nato.items():
            text = re.sub(r'\bletter\s+' + word + r'\b', lambda m, l=letter: l, text, flags=re.IGNORECASE)

        # Direct letter spelling: "upper A" → "A", "lower b" → "b"
        text = re.sub(r'\b(upper|capital|hoofdletter)\s+([a-z])\b', lambda m: m.group(2).upper(), text, flags=re.IGNORECASE)
        text = re.sub(r'\b(lower|kleine)\s+([a-z])\b', lambda m: m.group(2).lower(), text, flags=re.IGNORECASE)

        # Numbers: "number 5" / "digit five" / "cijfer vijf" → "5"
        for word, digit in number_words.items():
            text = re.sub(r'\b(number|digit|cijfer)\s+' + word + r'\b', lambda m, d=digit: d, text, flags=re.IGNORECASE)
        text = re.sub(r'\b(number|digit|cijfer)\s+(\d)\b', lambda m: m.group(2), text, flags=re.IGNORECASE)

        # Keyboard actions (actual key presses via xdotool)
        # Handle numbered key actions: "three backspaces", "5 tabs", etc.
        for num_word, num_val in num_words.items():
            for key_word, key_name in key_actions.items():
                pattern = r'\b' + num_word + r'\s+' + key_word + r'\b'
                if re.search(pattern, text, flags=re.IGNORECASE):
                    for _ in range(num_val):
                        subprocess.run(["xdotool", "key", key_name], check=False)
                    print(dictate(f"Key: {key_name} x{num_val}"))
                    text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # Handle digit + key: "3 backspaces", "5 tabs"
        for key_word, key_name in key_actions.items():
            pattern = r'\b(\d+)\s+' + key_word + r'\b'
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                num_val = int(match.group(1))
                for _ in range(num_val):
                    subprocess.run(["xdotool", "key", key_name], check=False)
                print(dictate(f"Key: {key_name} x{num_val}"))
                text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # Handle single key actions
        for word, key in key_actions.items():
            pattern = r'\b' + word + r'\b'
            if re.search(pattern, text, flags=re.IGNORECASE):
                count = len(re.findall(pattern, text, flags=re.IGNORECASE))
                for _ in range(count):
                    subprocess.run(["xdotool", "key", key], check=False)
                    print(dictate(f"Key: {key}"))
                text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text).strip()

        # Punctuation and grammar
        for word, symbol in replacements.items():
            text = re.sub(r'\b' + word + r'\b', lambda m, s=symbol: s, text, flags=re.IGNORECASE)

        # Emoji replacements (if enabled)
        if DICTATE_EMOJIS:
            for word, emoji in emoji_map.items():
                text = re.sub(r'\b' + word + r'\b', emoji, text, flags=re.IGNORECASE)

        # Clean up again after all replacements
        text = text.strip()

        # If nothing left to type after processing, stay in dictation mode
        if not text:
            continue

        # Type the processed text
        print(dictate(f"Typing: {text}"))
        subprocess.run(["xdotool", "type", "--", text + " "], check=False)
        transcript.append(text)

    # Show transcript summary and copy to clipboard
    if transcript:
        full_text = " ".join(transcript)
        print(f"\n{dictate('═' * 60)}")
        print(dictate("DICTATION TRANSCRIPT:"))
        print(f"{dictate('─' * 60)}")
        print(full_text)
        print(f"{dictate('─' * 60)}")
        print(dictate(f"Words: {len(full_text.split())} | Characters: {len(full_text)}"))
        print(f"{dictate('═' * 60)}\n")

        # Copy to clipboard
        try:
            subprocess.run(["xclip", "-selection", "clipboard"], input=full_text.encode(), check=True)
            print(dictate("📋 Copied to clipboard!"))
            speak(f"Done. {len(full_text.split())} words copied to clipboard.")
        except FileNotFoundError:
            print(dictate("Install xclip to enable clipboard: sudo apt install xclip"))
            speak(f"Done. {len(full_text.split())} words dictated.")
    else:
        speak("No text was dictated.")


def _handle_open_browser():
    """Handle opening browser."""
    import subprocess
    print(cmd("Opening browser"))
    # Try common browsers in order of preference
    browsers = ["firefox", "google-chrome", "chromium-browser", "brave-browser"]
    for browser in browsers:
        try:
            subprocess.Popen([browser], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            speak("Opening browser.")
            return
        except FileNotFoundError:
            continue
    speak("No browser found.")


def _handle_terminal(selected_device, samplerate):
    """Handle terminal command execution."""
    import subprocess
    import re
    from voice2json_intent import recognize_intent

    # Known Whisper hallucinations
    hallucinations = [
        "you", "thank you", "thanks", "you you", "you you you", "the", "a", "i",
        "um", "uh", "hmm", "ah", "eh", "mm", "oh", "yeah", "right", "okay",
        "music", "silence", "applause", "laughter", "sorry", "i'm sorry",
    ]

    print(cmd("TERMINAL command"))
    speak("What command?")
    command = whisper_speech_to_text(selected_device, samplerate, extended_listen=True).strip()

    # Skip hallucinations
    if command.lower().strip().rstrip('.') in hallucinations:
        print(cmd(f"Skipped hallucination: '{command}'"))
        speak("I didn't catch that. Try again.")
        return

    if command:
        # Process case instructions for words
        command = re.sub(r'\b(capital|uppercase|hoofdletter)\s+(\w+)', lambda m: m.group(2).upper(), command, flags=re.IGNORECASE)
        command = re.sub(r'\b(lowercase|kleine letter)\s+(\w+)', lambda m: m.group(2).lower(), command, flags=re.IGNORECASE)
        command = re.sub(r'\ball caps\s+(\w+)', lambda m: m.group(1).upper(), command, flags=re.IGNORECASE)

        # SPELL MODE - NATO phonetic alphabet (+ common Whisper mishearings)
        nato = {
            "alpha": "a", "alfa": "a", "albert": "a",
            "bravo": "b", "beta": "b", "boy": "b",
            "charlie": "c", "charles": "c",
            "delta": "d", "david": "d",
            "echo": "e", "edward": "e",
            "foxtrot": "f", "fox": "f", "frank": "f",
            "golf": "g", "george": "g",
            "hotel": "h", "henry": "h",
            "india": "i", "indigo": "i",
            "juliet": "j", "julia": "j", "john": "j",
            "kilo": "k", "king": "k",
            "lima": "l", "london": "l", "louis": "l",
            "mike": "m", "michael": "m", "mary": "m",
            "november": "n", "nancy": "n", "nora": "n",
            "oscar": "o", "oliver": "o",
            "papa": "p", "peter": "p", "paul": "p",
            "quebec": "q", "queen": "q",
            "romeo": "r", "robert": "r", "roger": "r",
            "sierra": "s", "sugar": "s", "sam": "s",
            "tango": "t", "tom": "t", "tommy": "t",
            "uniform": "u", "uncle": "u",
            "victor": "v", "victoria": "v",
            "whiskey": "w", "whisky": "w", "william": "w",
            "xray": "x", "x-ray": "x",
            "yankee": "y", "yellow": "y", "young": "y",
            "zulu": "z", "zebra": "z", "zero letter": "z",
        }
        for word, letter in nato.items():
            command = re.sub(r'\b(upper|capital|hoofdletter)\s+' + word + r'\b', lambda m, l=letter: l.upper(), command, flags=re.IGNORECASE)
            command = re.sub(r'\b(lower|kleine)\s+' + word + r'\b', lambda m, l=letter: l, command, flags=re.IGNORECASE)
        for word, letter in nato.items():
            command = re.sub(r'\bletter\s+' + word + r'\b', lambda m, l=letter: l, command, flags=re.IGNORECASE)
        command = re.sub(r'\b(upper|capital|hoofdletter)\s+([a-z])\b', lambda m: m.group(2).upper(), command, flags=re.IGNORECASE)
        command = re.sub(r'\b(lower|kleine)\s+([a-z])\b', lambda m: m.group(2).lower(), command, flags=re.IGNORECASE)
        # Numbers
        number_words = {
            "zero": "0", "nul": "0", "one": "1", "een": "1", "two": "2", "twee": "2",
            "three": "3", "drie": "3", "four": "4", "vier": "4", "five": "5", "vijf": "5",
            "six": "6", "zes": "6", "seven": "7", "zeven": "7", "eight": "8", "acht": "8",
            "nine": "9", "negen": "9"
        }
        for word, digit in number_words.items():
            command = re.sub(r'\b(number|digit|cijfer)\s+' + word + r'\b', lambda m, d=digit: d, command, flags=re.IGNORECASE)
        command = re.sub(r'\b(number|digit|cijfer)\s+(\d)\b', lambda m: m.group(2), command, flags=re.IGNORECASE)
        # Symbols for commands
        cmd_symbols = {
            "hyphen": "-", "dash": "-", "min": "-",
            "underscore": "_", "liggend streepje": "_",
            "slash": "/", "schuine streep": "/",
            "backslash": "\\",
            "dot": ".", "period": ".", "punt": ".",
            "space": " ", "spatie": " ",
        }
        for word, symbol in cmd_symbols.items():
            command = re.sub(r'\b' + word + r'\b', lambda m, s=symbol: s, command, flags=re.IGNORECASE)

        # Convert common Linux commands to lowercase (Whisper often capitalizes them)
        linux_commands = [
            "ls", "cd", "pwd", "cat", "grep", "find", "rm", "cp", "mv", "mkdir", "rmdir",
            "chmod", "chown", "sudo", "apt", "pip", "python", "python3", "git", "docker",
            "ssh", "scp", "curl", "wget", "tar", "zip", "unzip", "nano", "vim", "vi",
            "echo", "touch", "head", "tail", "less", "more", "man", "which", "whereis",
            "ps", "top", "htop", "kill", "killall", "df", "du", "free", "uname", "whoami",
            "hostname", "ifconfig", "ip", "ping", "netstat", "ss", "systemctl", "journalctl",
            "make", "cmake", "gcc", "g++", "npm", "node", "yarn", "cargo", "rustc",
        ]
        # Convert first word (command name) to lowercase if it's a known command
        words = command.split()
        if words:
            first_word_lower = words[0].lower()
            if first_word_lower in linux_commands:
                words[0] = first_word_lower
                command = ' '.join(words)

        # Filter non-ASCII and punctuation for TTS
        command_safe = ''.join(c for c in command if ord(c) < 128 and c not in '.?!,;:')
        command_safe = command_safe.strip()
        print(cmd(f"Command requested: {command}"))

        if not command_safe:
            speak("I didn't understand the command.")
            return

        speak(f"Run {command_safe}, yes or no?")
        confirm = whisper_speech_to_text(selected_device, samplerate).strip().lower()

        # Check for confirm/deny intent
        confirmed = False
        if use_voice2json:
            intent = recognize_intent(confirm, language="auto")
            if intent["action"] == "confirm":
                confirmed = True
            elif intent["action"] == "deny":
                speak("Cancelled.")
                return

        # Fallback keyword check
        if not confirmed and any(w in confirm for w in ["yes", "ja", "yep", "do it", "go ahead"]):
            confirmed = True

        if confirmed:
            print(cmd(f"Executing: {command}"))
            try:
                # Check if TERMINAL_NEW_WINDOW is enabled in config
                use_new_window = getattr(__import__('config'), 'TERMINAL_NEW_WINDOW', False)

                if use_new_window:
                    # Run in a new terminal window (stays open for 30 sec or until keypress)
                    wrapped_cmd = f'bash -c "{command}; echo; echo Press Enter to close...; read"'
                    # Try different terminal emulators
                    for term in ["gnome-terminal --", "xterm -e", "konsole -e", "xfce4-terminal -e"]:
                        try:
                            subprocess.Popen(f'{term} {wrapped_cmd}', shell=True)
                            speak(f"Running {command_safe} in new terminal.")
                            break
                        except:
                            continue
                else:
                    # Run in same terminal, capture output
                    result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
                    output = result.stdout or result.stderr or "Command completed with no output."
                    if LOG_OUTPUT_LENGTH > 0 and len(output) > LOG_OUTPUT_LENGTH:
                        output = output[:LOG_OUTPUT_LENGTH] + "... truncated"

                    # Display output prominently
                    print("\n" + "="*60)
                    print(f"$ {command}")
                    print("-"*60)
                    print(output)
                    print("="*60 + "\n")

                    # Speak a summary (don't read huge outputs)
                    lines = output.strip().split('\n')
                    if len(lines) > 5:
                        speak(f"Done. {len(lines)} lines of output shown on screen.")
                    else:
                        speak(output)
            except subprocess.TimeoutExpired:
                speak("Command timed out after 30 seconds.")
            except Exception as e:
                speak(f"Error: {str(e)}")
        else:
            speak("Cancelled.")


def process_voice_command(transcription, selected_device, samplerate):
    """Process a voice command after wake word detection."""
    global _last_transcription

    # Apply corrections from learning
    transcription = apply_corrections(transcription)
    transcription_lower = transcription.lower().strip().rstrip('.,!?')

    # Filter hallucinations early
    main_hallucinations = [
        "you", "thank you", "thanks", "you you", "you you you", "the", "a", "i",
        "um", "uh", "hmm", "ah", "eh", "mm", "oh", "yeah", "right", "okay",
        "music", "silence", "applause", "laughter", "sorry", "i'm sorry",
        "thank you for watching", "subscribe", "like and subscribe",
        # Prevent yes/no loops (not meaningful commands on their own)
        "yes", "ja", "yep", "nope", "no", "nee", "yes yes", "no no",
    ]
    if transcription_lower in main_hallucinations:
        print(cmd(f"Skipped hallucination: '{transcription}'"))
        return
    # Skip very short gibberish (1-2 chars)
    if len(transcription_lower) <= 2 and transcription_lower not in ["ok", "hi"]:
        print(cmd(f"Skipped short noise: '{transcription}'"))
        return

    print(cmd(f"Processing: {transcription_lower}"))

    # Try voice2json intent recognition first (if enabled)
    if use_voice2json:
        intent_result = recognize_intent(transcription_lower, language="auto")
        if intent_result["intent"]:
            print(v2j(f"Intent: {intent_result['intent']} ({intent_result['language']}) conf={intent_result['confidence']:.2f}"))
            action = intent_result["action"]

            # Handle recognized intents
            if action == "help":
                _show_help()
                return
            elif action == "clear_session":
                clear_session()
                speak("Session cleared.")
                return
            elif action == "learn_correction":
                _handle_learn_correction(selected_device, samplerate)
                return
            elif action == "show_corrections":
                _handle_show_corrections()
                return
            elif action == "add_calendar":
                _handle_add_calendar(selected_device, samplerate)
                return
            elif action == "check_calendar":
                _handle_check_calendar(selected_device, samplerate)
                return
            elif action == "clear_calendar":
                _handle_clear_calendar(selected_device, samplerate)
                return
            elif action == "remove_calendar":
                _handle_remove_calendar(selected_device, samplerate)
                return
            elif action == "dictate":
                _handle_dictation(selected_device, samplerate)
                return
            elif action == "terminal":
                _handle_terminal(selected_device, samplerate)
                return
            elif action == "open_browser":
                _handle_open_browser()
                return
            # confirm/deny are handled in context, ollama is fallback
        else:
            print(v2j(f"No intent matched, using keyword fallback"))

    # Learning mode triggers
    learn_triggers = ["learn that", "correct that", "fix that", "that's wrong"]
    if any(trigger in transcription_lower for trigger in learn_triggers):
        if _last_transcription:
            print(learn("Learning mode activated"))
            speak("What should it be?")
            correct_phrase = whisper_speech_to_text(selected_device, samplerate).strip()
            if correct_phrase:
                add_correction(_last_transcription, correct_phrase)
                speak(f"Got it. I'll remember that {_last_transcription} means {correct_phrase}.")
            else:
                speak("Sorry, I didn't catch that.")
        else:
            speak("Nothing to correct yet. Say something first.")
        return

    # Show corrections
    if "show corrections" in transcription_lower or "list corrections" in transcription_lower:
        corrections = list_corrections()
        if corrections:
            speak(f"You have {len(corrections)} corrections stored.")
        else:
            speak("No corrections stored yet.")
        return

    # Clear session
    if "clear session" in transcription_lower or "forget everything" in transcription_lower or "vergeet alles" in transcription_lower:
        clear_session()
        speak("Session cleared.")
        return

    # Language switching - only trigger on short commands (isolated words, not in sentences)
    word_count = len(transcription_lower.split())
    if word_count <= 3:  # Short command like "English" or "Speak Nederlands"
        if any(t in transcription_lower for t in LANG_SWITCH_EN):
            set_language("en")
            clear_session()  # Clear old context to prevent language mixing
            speak("Switched to English voice.")
            return
        if any(t in transcription_lower for t in LANG_SWITCH_NL):
            set_language("nl")
            clear_session()  # Clear old context to prevent language mixing
            speak("Overgeschakeld naar Nederlands. Sessie gewist.")
            return
        if any(t in transcription_lower for t in LANG_SWITCH_AUTO):
            set_language(None)
            speak("Language detection is now automatic.")
            return

    # Help command
    help_triggers = ["help me", "help", "what can you do", "commands", "commando's", "lijst", "list commands", "show commands", "options", "opties", "menu"]
    if any(t in transcription_lower for t in help_triggers) or transcription_lower in ["help", "commands", "menu"]:
        print(cmd(f"Matched help trigger in: '{transcription_lower}'"))
        # Blue text: \033[94m, Reset: \033[0m
        blue = "\033[94m"
        reset = "\033[0m"
        print(f"""
{blue}╔════════════════════════════════════════════════════════════════════════════════╗
║                              AVAILABLE COMMANDS                                ║
╠════════════════════════════════════════════════════════════════════════════════╣
║ 📅 CALENDAR              │ 💬 QUESTIONS            │ 🎓 LEARNING               ║
║    • "Add to calendar"   │    • Just ask anything  │    • "Learn that"         ║
║    • "Check my agenda"   │                         │    • "Correct that"       ║
║    • "Remove event"      │                         │    • "Show corrections"   ║
╠════════════════════════════════════════════════════════════════════════════════╣
║ 🧠 SESSION               │ 💻 TERMINAL             │ ✍️ DICTATION              ║
║    • "Clear session"     │    • "Run command"      │    • "Dictate"/"Dicteer"  ║
║    • "Vergeet alles"     │    • "Execute"          │    • "Stop" to end        ║
╠════════════════════════════════════════════════════════════════════════════════╣
║ 🌐 LANGUAGE SWITCHING (say word alone, 1-3 words only)                         ║
║    • "English" / "Nederlands" / "Dutch" - switch voice                         ║
║    • "Auto" / "Automatisch" - auto-detect language                             ║
║    NOTE: Won't trigger in sentences like "I want to learn English"             ║
╠════════════════════════════════════════════════════════════════════════════════╣
║ DICTATION GRAMMAR                                                              ║
║ Case:    "capital X" "lowercase X" "all caps X" "hoofdletter X"                ║
║ Punct:   "period/punt" "comma/komma" "question mark" "exclamation mark"        ║
║ Format:  "new line" "new paragraph" "tab" "space"                              ║
║ Symbols: "at sign" "hashtag" "slash" "underscore" "hyphen" "asterisk"          ║
║ Brackets: "open/close parenthesis" "open/close bracket" "open/close brace"    ║
║ Quotes:  "quote" "single quote" "aanhalingsteken" "apostrof"                   ║
╚════════════════════════════════════════════════════════════════════════════════╝{reset}
""")
        if not quiet_help:
            speak("Calendar: say add to calendar, check my agenda, or remove event. Questions: just ask me anything. Learning: say learn that or correct that. Session: say clear session to forget everything.")
        return

    # Shutdown command
    shutdown_triggers = [
        "kill yourself", "shut down", "shutdown", "exit", "quit",
        "goodbye jarvis", "bye jarvis", "stop jarvis",
        "sluit af", "afsluiten", "stop jezelf", "doei jarvis"
    ]
    if any(trigger in transcription_lower for trigger in shutdown_triggers):
        print(cmd("Shutdown requested"))
        speak("Goodbye!", interruptable=False)
        import sys
        sys.exit(0)

    # Store for potential learning
    _last_transcription = transcription

    # PRIORITY CHECK: Dictation mode (check BEFORE calendar to avoid "dictate" → "date" misrouting)
    dictate_triggers = ["dictate", "diktate", "dik tate", "dicteer", "dicteren", "dictatie", "type this", "start typing"]
    if any(w in transcription_lower for w in dictate_triggers):
        _handle_dictation(selected_device, samplerate)
        return

    # PRIORITY CHECK: Terminal commands
    terminal_triggers = ["run command", "execute", "terminal", "shell"]
    if any(w in transcription_lower for w in terminal_triggers):
        _handle_terminal(selected_device, samplerate)
        return

    # Calendar keywords (flexible matching, including common misspellings)
    cal_words = ["calendar", "calander", "agenda", "schedule", "event", "meeting"]
    has_cal = any(w in transcription_lower for w in cal_words)

    if has_cal and any(w in transcription_lower for w in ["add", "put", "create", "new", "schedule"]):
        print(cmd(f"Calendar ADD matched: has_cal={has_cal}, triggers=['add','put','create','new','schedule']"))
        speak("What is the event?")
        event_name = whisper_speech_to_text(selected_device, samplerate).strip()

        speak("What time does the event start?")
        start_time = whisper_speech_to_text(selected_device, samplerate).strip()

        speak("What time does the event end?")
        end_time = whisper_speech_to_text(selected_device, samplerate).strip()

        speak("What date is the event on?")
        event_date = whisper_speech_to_text(selected_device, samplerate).strip()

        add_event_to_calendar(event_name, start_time, end_time, date=event_date)

    elif has_cal and any(w in transcription_lower for w in ["what", "check", "show", "list", "today", "tomorrow"]):
        print(cmd(f"Calendar CHECK matched: has_cal={has_cal}, triggers=['what','check','show','list','today','tomorrow']"))
        speak("For which date or week?")
        query = whisper_speech_to_text(selected_device, samplerate).strip()

        if "week" in query.lower():
            if "this week" in query.lower():
                check_calendar(date="this week", week=True)
            elif "next week" in query.lower():
                check_calendar(date="next week", week=True)
            elif "week of" in query.lower():
                specific_week_start = query.lower().replace("week of", "").strip()
                check_calendar(specific_week_start=specific_week_start, week=True)
            else:
                speak("I couldn't understand the week query.")
        else:
            check_calendar(date=query)

    elif has_cal and any(w in transcription_lower for w in ["clear", "delete all", "remove all", "empty"]):
        print(cmd(f"Calendar CLEAR matched: has_cal={has_cal}, triggers=['clear','delete all','remove all','empty']"))
        speak("For which date or week would you like to clear?")
        query = whisper_speech_to_text(selected_device, samplerate).strip()

        if "week" in query.lower():
            if "this week" in query.lower():
                clear_calendar(date="this week", week=True)
            elif "next week" in query.lower():
                clear_calendar(date="next week", week=True)
            elif "week of" in query.lower():
                specific_week_start = query.lower().replace("week of", "").strip()
                clear_calendar(date=specific_week_start, week=True)
            else:
                speak("I couldn't understand the week query.")
        else:
            clear_calendar(date=query)

    elif has_cal and any(w in transcription_lower for w in ["remove", "delete", "cancel"]):
        print(cmd(f"Calendar REMOVE matched: has_cal={has_cal}, triggers=['remove','delete','cancel']"))
        speak("What is the name of the event to remove?")
        event_name = whisper_speech_to_text(selected_device, samplerate).strip()

        speak("What date is this event on?")
        event_date = whisper_speech_to_text(selected_device, samplerate).strip()

        remove_event(event_name, event_date)

    # Note: Dictation and Terminal checks moved to PRIORITY section above (before calendar)

    else:
        # Default: send to Ollama as a question
        if LOG_CMD_LENGTH > 0 and len(transcription) > LOG_CMD_LENGTH:
            print(cmd(f"OLLAMA fallback: '{transcription[:LOG_CMD_LENGTH]}...'"))
        else:
            print(cmd(f"OLLAMA fallback: '{transcription}'"))
        interrupted = ask_ollama(transcription)  # Use original case for Ollama
        if interrupted:
            print(cmd("Response interrupted - returning to listen"))
            speak("Okay.", interruptable=False)

def main():
    parser = argparse.ArgumentParser(description='Assistmint Voice Assistant')
    parser.add_argument('--model', '-m', help='Ollama model to use (skip selection)')
    parser.add_argument('--device', '-d', type=int, help='Audio device index (skip selection)')
    parser.add_argument('--voice', '-v', action='store_true', help='Start directly in voice mode')
    parser.add_argument('--type', '-t', action='store_true', help='Start directly in type mode')
    parser.add_argument('--no-commands', '-nc', action='store_true', help='Skip TTS for help command (print only)')
    args = parser.parse_args()

    global quiet_help
    quiet_help = args.no_commands

    # Select or set Ollama model
    if args.model:
        set_model(args.model)
    else:
        select_ollama_model()

    # Load session history
    load_session()

    # Select microphone
    input_devices = list_microphones()
    if args.device is not None:
        selected_device = input_devices[args.device]
        samplerate = selected_device['default_samplerate']
        print(f"Using device {args.device}: {selected_device['name']} ({samplerate} Hz)")
    else:
        selected_device, samplerate = select_microphone_and_samplerate(input_devices)

    # Determine mode
    if args.voice:
        mode = 'voice'
    elif args.type:
        mode = 'type'
    else:
        mode = None

    while True:
        if mode is None:
            mode = input("Type 'voice' to speak or 'type' to enter commands: ").strip().lower()

        if mode == "voice":
            # Initialize wake word detection
            init_wake_word()
            print("\n" + "="*50)
            print("Voice mode active - say 'Hey Jarvis' to wake me up!")
            print("="*50 + "\n")

            voice_loop = True
            while voice_loop:
                # Low-power wake word listening
                detected = listen_for_wake_word(selected_device, samplerate)
                if detected is None:
                    # Error occurred, wait before retry
                    time.sleep(1)
                    continue
                if detected:
                    # Wake word detected! Now listen for command
                    speak("Yes?")

                    # Get the actual command with extended listening
                    print("Listening for your command...")
                    transcription = whisper_speech_to_text(selected_device, samplerate, extended_listen=True)

                    if transcription:
                        print(f"Command: {transcription}")
                        process_voice_command(transcription, selected_device, samplerate)

                    # Check if calendar confirmation is pending - keep listening without wake word
                    while has_pending_calendar():
                        print(cmd("Waiting for calendar confirmation... (say ja/yes or nee/no)"))
                        confirm_transcription = whisper_speech_to_text(selected_device, samplerate)
                        if confirm_transcription:
                            confirm_lower = confirm_transcription.lower().strip().rstrip('.,!?')
                            # Check for cancel words
                            cancel_words = ["no", "nee", "cancel", "annuleer", "stop", "never mind", "laat maar"]
                            if any(w in confirm_lower for w in cancel_words):
                                clear_pending_calendar()
                                speak("Okay, cancelled.")
                                break
                            # Otherwise send to Ollama (it will detect [CALENDAR_CONFIRM] in response)
                            print(f"Confirmation: {confirm_transcription}")
                            ask_ollama(confirm_transcription)

                    print("\n💤 Back to sleep... say 'Hey Jarvis' to wake me up\n")

        elif mode == "type":
            print("\n" + "="*50)
            print("Type mode - enter commands directly")
            print("'voice' or 'v' = switch to voice | 'quit' = exit")
            print("="*50 + "\n")

            while True:
                command = input("You: ").strip()
                if command.lower() == 'quit':
                    print("Goodbye!")
                    return
                if command.lower() in ('voice', 'v'):
                    mode = 'voice'
                    break
                if command:
                    # Use same processing logic as voice
                    process_voice_command(command, selected_device, samplerate)

                    # Check if calendar confirmation is pending
                    while has_pending_calendar():
                        confirm = input("Confirm (ja/yes or nee/no): ").strip()
                        if not confirm:
                            continue
                        confirm_lower = confirm.lower().rstrip('.,!?')
                        cancel_words = ["no", "nee", "cancel", "annuleer", "stop", "never mind", "laat maar"]
                        if any(w in confirm_lower for w in cancel_words):
                            clear_pending_calendar()
                            speak("Okay, cancelled.")
                            break
                        ask_ollama(confirm)

        else:
            print("Invalid mode selected. Please choose 'voice' or 'type'.")

if __name__ == "__main__":
    main()

import requests
import json
import os
from datetime import datetime
from text_to_speech import speak, get_language
from colors import session
from config import SYSTEM_PROMPT, SYSTEM_PROMPT_NL, MAX_TOKENS, MAX_MESSAGES, TEMPERATURE, TOP_P, FREQUENCY_PENALTY, PRESENCE_PENALTY, VERBOSE_SESSION, DEBUG_API, SESSION_ENABLED, DEFAULT_MODEL

# Global variable to store selected model (uses config default)
selected_model = DEFAULT_MODEL

# Session memory
SESSION_FILE = os.path.expanduser("~/.assistmint_session.json")
messages = []

# Pending calendar event (waiting for confirmation)
_pending_calendar_event = None


def has_pending_calendar():
    """Check if there's a calendar event waiting for confirmation."""
    return _pending_calendar_event is not None


def clear_pending_calendar():
    """Clear pending calendar event (e.g., on timeout or cancel)."""
    global _pending_calendar_event
    _pending_calendar_event = None

def load_session():
    """Load session from JSON file."""
    global messages
    if not SESSION_ENABLED:
        messages = []
        return messages
    if os.path.exists(SESSION_FILE):
        try:
            with open(SESSION_FILE, 'r') as f:
                messages = json.load(f)
            if VERBOSE_SESSION:
                print(session(f"Loaded {len(messages)} messages"))
        except:
            messages = []
    return messages

def save_session():
    """Save session to JSON file."""
    if not SESSION_ENABLED:
        return
    with open(SESSION_FILE, 'w') as f:
        json.dump(messages, f, indent=2)

def clear_session():
    """Clear session history."""
    global messages
    messages = []
    save_session()
    if VERBOSE_SESSION:
        print(session("Cleared"))

def list_ollama_models():
    """Fetch available models from Ollama."""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [m["name"] for m in models]
    except requests.ConnectionError:
        print("Could not connect to Ollama. Make sure it's running.")
    return []


def unload_ollama_model():
    """Tell Ollama to unload ALL loaded models from VRAM immediately."""
    try:
        # Get list of currently loaded models
        ps_response = requests.get("http://localhost:11434/api/ps", timeout=5)
        if ps_response.status_code == 200:
            loaded = ps_response.json().get("models", [])
            if not loaded:
                return
            # Unload each loaded model
            for model_info in loaded:
                model_name = model_info.get("name", "")
                if model_name:
                    requests.post(
                        "http://localhost:11434/api/generate",
                        json={"model": model_name, "keep_alive": 0, "prompt": ""},
                        timeout=5
                    )
                    print(session(f"[VRAM] Unloaded {model_name}"))
    except Exception as e:
        print(session(f"[VRAM] Unload failed: {e}"))

def set_model(model_name):
    """Set model directly without prompting."""
    global selected_model
    selected_model = model_name
    print(f"Using model: {selected_model}")
    return selected_model

def select_ollama_model():
    """Let user select a model from available Ollama models."""
    global selected_model
    models = list_ollama_models()

    if not models:
        print("No models found. Using default: mistral")
        return "mistral"

    print("\n=== Available Ollama Models ===")
    for i, model in enumerate(models):
        print(f"  {i}: {model}")

    try:
        choice = input("Select model number (or press Enter for first): ").strip()
        if choice == "":
            selected_model = models[0]
        else:
            selected_model = models[int(choice)]
    except (ValueError, IndexError):
        selected_model = models[0]

    print(f"Selected model: {selected_model}\n")
    return selected_model

def _get_system_prompt():
    """Get system prompt based on current language setting."""
    lang = get_language()

    # Add today's date to the prompt
    today = datetime.now()
    date_info = f"\n\nToday is {today.strftime('%A, %B %d, %Y')}."

    if lang == "nl":
        print(session(f"[LLM] Using Dutch system prompt"))
        return SYSTEM_PROMPT_NL + f"\n\nVandaag is {today.strftime('%A %d %B %Y')}."
    print(session(f"[LLM] Using English system prompt (lang={lang})"))
    return SYSTEM_PROMPT + date_info


def _execute_calendar_action(response_text):
    """
    Check if LLM response contains a calendar action and execute it.
    Supports two-step flow: PENDING (store) -> CONFIRM (add)
    Returns True if action was handled, False otherwise.
    """
    global _pending_calendar_event
    import re

    print(session(f"[CALENDAR] Checking response for calendar action..."))

    # Check for CONFIRM (user said yes to pending event)
    if "[CALENDAR_CONFIRM]" in response_text and _pending_calendar_event:
        print(session(f"[CALENDAR] Confirmed! Adding: {_pending_calendar_event.get('event')}"))
        _add_pending_event()
        return True

    # Check for PENDING (new event waiting for confirmation)
    pattern = r'\[CALENDAR_PENDING\]\s*(\{.*?\})\s*\[/CALENDAR_PENDING\]'
    match = re.search(pattern, response_text, re.DOTALL)

    print(session(f"[CALENDAR] Looking for PENDING block... found={match is not None}"))
    if not match:
        # Debug: show first 200 chars of response
        print(session(f"[CALENDAR] Response preview: {response_text[:200]}..."))

    if match:
        try:
            data = json.loads(match.group(1))
            print(session(f"[CALENDAR] Pending: {data}"))

            # Store for later confirmation
            _pending_calendar_event = data

            # Check for conflicts
            from assistmint.calendar_manager import check_calendar_conflicts
            conflicts = check_calendar_conflicts(data.get("date"), data.get("start"), data.get("end"))
            if conflicts:
                print(session(f"[CALENDAR] Conflict detected: {conflicts}"))
                # The LLM will mention this in its response

            return True

        except json.JSONDecodeError as e:
            print(session(f"[CALENDAR] JSON parse error: {e}"))
            return False

    # Legacy: direct [CALENDAR_ADD] (no confirmation needed)
    pattern_add = r'\[CALENDAR_ADD\]\s*(\{.*?\})\s*\[/CALENDAR_ADD\]'
    match_add = re.search(pattern_add, response_text, re.DOTALL)

    if match_add:
        try:
            data = json.loads(match_add.group(1))
            print(session(f"[CALENDAR] Direct add: {data}"))
            _pending_calendar_event = data
            _add_pending_event()
            return True
        except Exception as e:
            print(session(f"[CALENDAR] Error: {e}"))
            return False

    return False


def _add_pending_event():
    """Add the pending calendar event."""
    global _pending_calendar_event

    if not _pending_calendar_event:
        return False

    try:
        from assistmint.calendar_manager import add_event_to_calendar_extended

        data = _pending_calendar_event
        event_name = data.get("event", "Untitled")
        date = data.get("date")
        start_time = data.get("start")
        end_time = data.get("end")
        location = data.get("location")
        description = data.get("description")
        reminder = data.get("reminder", 30)

        if not date or not start_time:
            print(session("[CALENDAR] Missing date or start time"))
            return False

        # Default end time: 1 hour after start
        if not end_time:
            h, m = map(int, start_time.split(":"))
            h = (h + 1) % 24
            end_time = f"{h:02d}:{m:02d}"

        # Convert 24h to 12h for calendar function
        def to_12h(t):
            h, m = map(int, t.split(":"))
            period = "AM" if h < 12 else "PM"
            h = h % 12 or 12
            return f"{h}:{m:02d} {period}"

        # Add to calendar
        add_event_to_calendar_extended(
            event_name=event_name,
            start_time=to_12h(start_time),
            end_time=to_12h(end_time),
            date=date,
            location=location,
            description=description,
            reminder_minutes=reminder
        )

        # Clear pending
        _pending_calendar_event = None
        return True

    except Exception as e:
        print(session(f"[CALENDAR] Error adding: {e}"))
        return False


def _clean_calendar_response(response_text):
    """Remove calendar blocks from response for cleaner TTS."""
    import re
    # Remove all calendar-related blocks
    clean = re.sub(r'\[CALENDAR_ADD\].*?\[/CALENDAR_ADD\]', '', response_text, flags=re.DOTALL)
    clean = re.sub(r'\[CALENDAR_PENDING\].*?\[/CALENDAR_PENDING\]', '', clean, flags=re.DOTALL)
    clean = re.sub(r'\[CALENDAR_CONFIRM\].*?\[/CALENDAR_CONFIRM\]', '', clean, flags=re.DOTALL)
    # Clean up extra whitespace
    clean = re.sub(r'\n\s*\n', '\n', clean).strip()
    return clean


def ask_ollama(question):
    """Send a question to Ollama and return the answer."""
    global messages

    # Add user message to history
    messages.append({"role": "user", "content": question})

    # Trim if over limit (0 = unlimited)
    if MAX_MESSAGES > 0 and len(messages) > MAX_MESSAGES:
        messages = messages[-MAX_MESSAGES:]

    url = "http://localhost:11434/v1/chat/completions"  # Ollama API
    payload = {
        "model": selected_model,  # Use selected model
        "messages": [
            {"role": "system", "content": _get_system_prompt()},
        ] + messages,
        "stream": False,
        "max_tokens": MAX_TOKENS,
        "stop": None,
        "frequency_penalty": FREQUENCY_PENALTY,
        "presence_penalty": PRESENCE_PENALTY,
        "temperature": TEMPERATURE,
        "top_p": TOP_P
    }
    headers = {"Content-Type": "application/json"}

    if DEBUG_API:
        print("API Call Information:")
        print(f"URL: {url}")
        print(f"Headers: {headers}")
        print(f"Payload: {json.dumps(payload, indent=4)}")

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        if DEBUG_API:
            print(f"Response Status Code: {response.status_code}")
            print(f"Response Content: {response.content.decode('utf-8')}")

        if response.status_code == 200:
            answer = response.json().get("choices", [{}])[0].get("message", {}).get("content", "Sorry, I couldn't get a response.")
            # Add assistant response to history
            messages.append({"role": "assistant", "content": answer})
            save_session()

            # Check for calendar action in response
            calendar_result = _execute_calendar_action(answer)
            if calendar_result:
                # Remove the JSON block from spoken response
                clean_answer = _clean_calendar_response(answer)
                return speak(clean_answer)

            return speak(answer)
        else:
            speak("An error occurred while trying to communicate with Ollama.", interruptable=False)
            return False
    except requests.ConnectionError:
        print("Connection error: Unable to reach the Ollama API.")
        speak("Please make sure Ollama is running.", interruptable=False)
        return False
    except requests.RequestException as e:
        print(f"API request failed: {e}")
        speak("An error occurred while trying to communicate with Ollama.", interruptable=False)
        return False


def parse_calendar_event(text):
    """
    Use LLM to parse natural language into calendar event data.
    Fast, single-shot extraction - no conversation context.

    Args:
        text: Natural language like "meeting with Jan tomorrow at half 3"

    Returns:
        dict with keys: event_name, date, start_time, end_time (or None if parsing failed)
    """
    today = datetime.now()
    today_str = today.strftime("%A, %B %d, %Y")  # e.g., "Friday, January 17, 2026"

    prompt = f"""Extract calendar event details from the user's text. Today is {today_str}.

User said: "{text}"

Return ONLY a JSON object with these fields (no explanation, no markdown):
- event_name: the FULL event title/description (include names, details - e.g., "meeting met Piet" not just "meeting")
- date: the date in YYYY-MM-DD format
- start_time: start time in HH:MM format (24-hour)
- end_time: end time in HH:MM format (24-hour), or null if not specified

Examples:
- "meeting tomorrow at 2pm" -> {{"event_name": "meeting", "date": "2026-01-18", "start_time": "14:00", "end_time": null}}
- "dentist op 20 januari om half 3 tot 4 uur" -> {{"event_name": "dentist", "date": "2026-01-20", "start_time": "14:30", "end_time": "16:00"}}
- "call with Bob next Monday 10am" -> {{"event_name": "call with Bob", "date": "2026-01-20", "start_time": "10:00", "end_time": null}}

IMPORTANT - Dutch time conventions:
- "half X" means 30 minutes BEFORE X (half 3 = 14:30, half 4 = 15:30, half 12 = 11:30)
- "kwart over X" means X:15 (kwart over 2 = 14:15)
- "kwart voor X" means 15 minutes BEFORE X (kwart voor 3 = 14:45)
- "X uur" means X:00 (10 uur = 10:00, 3 uur = 15:00 in afternoon context)

JSON response:"""

    url = "http://localhost:11434/v1/chat/completions"
    # Use a capable model for parsing (qwen2.5:7b is better at Dutch time conventions)
    parse_model = "qwen2.5:7b"
    payload = {
        "model": parse_model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": 200,
        "temperature": 0.1,  # Low temperature for consistent parsing
    }

    try:
        # Longer timeout for first request (model loading)
        response = requests.post(url, json=payload, timeout=60)
        if response.status_code == 200:
            answer = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")

            # Try to parse JSON from response
            answer = answer.strip()
            # Remove markdown code blocks if present
            if answer.startswith("```"):
                answer = answer.split("```")[1]
                if answer.startswith("json"):
                    answer = answer[4:]
            answer = answer.strip()

            try:
                data = json.loads(answer)
                print(session(f"[LLM Parse] {data}"))
                return data
            except json.JSONDecodeError:
                print(session(f"[LLM Parse] Failed to parse JSON: {answer}"))
                return None
        else:
            print(session(f"[LLM Parse] API error: {response.status_code}"))
            return None
    except Exception as e:
        print(session(f"[LLM Parse] Error: {e}"))
        return None


def smart_add_calendar(text):
    """
    Smart calendar add - parse natural language and add event.

    Args:
        text: Natural language like "meeting with Jan tomorrow at half 3 to 4"

    Returns:
        True if event was added, False otherwise
    """
    from assistmint.calendar_manager import add_event_to_calendar

    data = parse_calendar_event(text)
    if not data:
        speak("Sorry, I couldn't understand the event details.")
        return False

    event_name = data.get("event_name")
    date = data.get("date")
    start_time = data.get("start_time")
    end_time = data.get("end_time")

    if not event_name or not date or not start_time:
        speak("I need at least an event name, date, and start time.")
        return False

    # Default end time: 1 hour after start
    if not end_time:
        hour, minute = map(int, start_time.split(":"))
        hour = (hour + 1) % 24
        end_time = f"{hour:02d}:{minute:02d}"

    # Convert 24h time to 12h for add_event_to_calendar
    def to_12h(time_str):
        h, m = map(int, time_str.split(":"))
        period = "AM" if h < 12 else "PM"
        h = h % 12 or 12
        return f"{h}:{m:02d} {period}"

    start_12h = to_12h(start_time)
    end_12h = to_12h(end_time)

    add_event_to_calendar(event_name, start_12h, end_12h, date)
    return True


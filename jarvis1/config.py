# Assistmint Configuration
# Adjust these values for your environment

# === OLLAMA / LLM ===
# API endpoint - change if running Ollama on different host/port
OLLAMA_API_URL = "http://localhost:11434"

# Timeouts (seconds) - increase on slower systems
OLLAMA_CHECK_TIMEOUT = 2        # Fast check if Ollama is running
OLLAMA_LIST_TIMEOUT = 5         # List available models
OLLAMA_COMPLETION_TIMEOUT = 180 # Main chat completion (longer for complex responses)
OLLAMA_PARSE_TIMEOUT = 15       # Calendar/extraction parsing (simpler tasks)

# Default system prompt (English)
SYSTEM_PROMPT = """You are Jarvis, a helpful voice assistant. Keep responses short and conversational.

CAPABILITIES:
- Answer questions on any topic
- Help with coding and technical problems
- Provide information and explanations
- Manage calendar events (when explicitly requested)

RESPONSE STYLE:
- Be concise - user is listening, not reading
- Think step by step for complex questions
- Use natural speech, avoid bullet points

CALENDAR (only when user explicitly asks to add/schedule an event):
When user says things like "add to calendar", "schedule", "remind me about [event] on [date]":
[CALENDAR_PENDING]
{"event": "title", "date": "YYYY-MM-DD", "start": "HH:MM", "end": "HH:MM", "location": null, "description": null, "reminder": 30}
[/CALENDAR_PENDING]
[Confirm message]. Say yes to confirm.

DO NOT use calendar tags for:
- Questions about dates/history ("when was X invented")
- General knowledge questions
- Anything not explicitly about scheduling"""

# Dutch system prompt
SYSTEM_PROMPT_NL = """Je bent Jarvis, een behulpzame Nederlandse spraakassistent. Houd antwoorden kort en conversationeel.

⚠️ TAALREGEL: Antwoord ALTIJD in het Nederlands. Vertaal Engelse termen (meeting = vergadering).

MOGELIJKHEDEN:
- Beantwoord vragen over elk onderwerp
- Help met programmeren en technische problemen
- Geef informatie en uitleg
- Beheer agenda-afspraken (alleen op verzoek)

ANTWOORDSTIJL:
- Wees beknopt - gebruiker luistert, leest niet
- Denk stap voor stap bij complexe vragen
- Gebruik natuurlijke spraak

AGENDA (alleen als gebruiker expliciet vraagt om toevoegen/inplannen):
Bij zinnen als "voeg toe aan agenda", "plan in", "herinner me aan [event] op [datum]":
[CALENDAR_PENDING]
{"event": "titel", "date": "YYYY-MM-DD", "start": "HH:MM", "end": "HH:MM", "location": null, "description": null, "reminder": 30}
[/CALENDAR_PENDING]
[Bevestigingsbericht]. Zeg ja om te bevestigen.

Nederlandse tijd: "half 3" = 14:30, "kwart over 2" = 14:15

GEEN agenda-tags voor:
- Vragen over data/geschiedenis ("wanneer was X uitgevonden")
- Algemene kennisvragen
- Alles wat niet expliciet over inplannen gaat"""

# === MODEL CONFIGURATION ===
# Default models
DEFAULT_MODEL = "qwen2.5:3b"                      # English/fallback (1.9GB)
DEFAULT_MODEL_NL = "bramvanroy/fietje-2b-chat:q8_0" #"bramvanroy/fietje-2b-chat:q4_K_M"  # Dutch (1.7GB - smallest!)
MODEL_AUTO_SWITCH = True           # Auto-switch based on detected language

# Per-model settings (override defaults)
# Keys: model name (or partial match like "qwen", "fietje")
MODEL_SETTINGS = {
    # Qwen models - good multilingual, needs moderate creativity
    "qwen2.5:3b": {
        "max_tokens": 1750,
        "temperature": 0.77,
        "top_p": 0.91,
        "frequency_penalty": 0.42,
        "presence_penalty": 0.38,
    },
    "bramvanroy/fietje-2b-chat:q8_0": {
        "max_tokens": 600,
        "temperature": 0.1,
        "top_p": 0.87,
        "frequency_penalty": 0.75,
        "presence_penalty": 0.75,
    },

    "qwen2.5:7b": {
        "max_tokens": 1750,
        "temperature": 0.75,
        "top_p": 0.90,
        "frequency_penalty": 0.40,
        "presence_penalty": 0.35,
    },
    "saul:latest": {
        "max_tokens": 600,           # Korter = minder kans op degeneratie
        "temperature": 0.45,          # Lager = minder hallucinatie
        "top_p": 0.87,               # Strikter sampling
        "frequency_penalty": 0.7,    # Hoger = voorkomt repetitie
        "presence_penalty": 0.7,     # Hoger = meer variatie
    },

    # Fietje - Dutch model, needs stricter settings to reduce hallucination
    "fietje": {
        "max_tokens": 500,          # Shorter = less hallucination
        "temperature": 0.5,         # Lower = more deterministic
        "top_p": 0.85,              # Stricter sampling
        "frequency_penalty": 0.6,   # Higher = less repetition
        "presence_penalty": 0.5,    # Higher = more focused
    },
    # DeepSeek - reasoning model
    "deepseek": {
        "max_tokens": 1000,
        "temperature": 0.6,
        "top_p": 0.90,
        "frequency_penalty": 0.3,
        "presence_penalty": 0.3,
    },
}

# Default settings (used if model not in MODEL_SETTINGS)
MAX_TOKENS = 750
TEMPERATURE = 0.77
TOP_P = 0.91
FREQUENCY_PENALTY = 0.42
PRESENCE_PENALTY = 0.38

# Extraction settings (calendar, parsing - always deterministic)
EXTRACTION_MAX_TOKENS = 300
EXTRACTION_TEMPERATURE = 0.1


def get_model_settings(model_name: str) -> dict:
    """Get settings for a specific model, with fallback to defaults."""
    # Check exact match first
    if model_name in MODEL_SETTINGS:
        return MODEL_SETTINGS[model_name]

    # Check partial match (e.g., "fietje:latest" matches "fietje")
    for key in MODEL_SETTINGS:
        if key in model_name or model_name.startswith(key):
            return MODEL_SETTINGS[key]

    # Return defaults
    return {
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "frequency_penalty": FREQUENCY_PENALTY,
        "presence_penalty": PRESENCE_PENALTY,
    }

# === SESSION ===
SESSION_ENABLED = True      # Enable session persistence (save/load conversation history)
MAX_MESSAGES = 50           # Rolling window size (25 exchanges: user + assistant pairs)

# === AUDIO SETTINGS ===
AUDIO_SAMPLE_RATE = 16000   # Sample rate for mic monitoring (16kHz = speech optimal)
AUDIO_BLOCKSIZE = 1600      # Audio buffer size (100ms at 16kHz)

# Speech recognition
SILENCE_SKIP_DB = -45       # Skip transcription if quieter than this
SPEECH_START_DB = -40       # Consider speech started above this
SILENCE_DROP_DB = 19        # dB drop from peak = end of speech
SILENCE_DURATION = 1.2      # Seconden stilte voordat opname stopt
SILENCE_DURATION_EXT = 2.0  # Seconden stilte voor extended listen (langere vragen)

# Noise reduction
NOISE_REDUCE = True          # AI noise suppression voor headphones/ruisige omgevingen
NOISE_REDUCE_STRENGTH = 0.65 # 0.0-1.0: How aggressive (0.8 = strong, 0.5 = mild)

# Whisper STT
# Single model (used when no per-language models configured)
WHISPER_MODEL = "medium"       # Options: tiny, base, small, medium, large (RTX 3070: use small/base)

# Per-language models (optional - set to None to use WHISPER_MODEL for all)
# Can be model names ("small", "medium") or paths to local models
WHISPER_MODEL_EN = None       # English-optimized model (None = use WHISPER_MODEL)
WHISPER_MODEL_NL = None       # Dutch-optimized model (None = use WHISPER_MODEL)
# Example with local models:
# WHISPER_MODEL_EN = "/path/to/whisper-en-model"
# WHISPER_MODEL_NL = "/path/to/whisper-nl-model"

WHISPER_BEAM_SIZE = 5         #was 5 Higher = better quality, slower (1-10)
WHISPER_SAMPLE_RATE = 16000   # Whisper vereist 16kHz - niet aanpassen!
STT_BLOCKSIZE = 4096          # Audio buffer voor spraakopname
STT_QUEUE_TIMEOUT = 0.35     # was 0.3 Audio queue timeout (seconds) - lower = more responsive

# Whisper anti-hallucination settings
# These help prevent Whisper from generating fake text on silence/noise
WHISPER_NO_SPEECH_THRESHOLD = 0.6       # 0.0-1.0: Probability threshold for "no speech"
WHISPER_LOG_PROB_THRESHOLD = -1.0       # Log probability threshold for valid speech
WHISPER_HALLUCINATION_SILENCE = 0.4     # Silence duration to trigger hallucination filter

# === GPU SETTINGS ===
USE_GPU = True              # Probeer GPU te gebruiken (met CPU fallback)
GPU_DEVICE_ID = 0        # CUDA device ID (None = auto-select beste GPU, 0/1/2 = specifieke GPU)
WHISPER_COMPUTE_TYPE = "float16"  # GPU: float16, CPU: int8 (auto-detect)

# === TTS SETTINGS (Piper) ===
# Voice models: ~/.local/share/piper/voices/
#   English: en_US-lessac-medium.onnx
#   Dutch:   nl_BE-nathalie-medium.onnx

# --- ENGLISH VOICE SETTINGS ---
TTS_SPEED_EN = 0.90          # Speech rate: 0.5=slow, 1.0=normal, 1.5=fast
TTS_PITCH_EN = 0.95          # Pitch: 0.8=lower, 1.0=normal, 1.2=higher
TTS_VOLUME_EN = 1.0          # Volume multiplier: 0.5=quiet, 1.0=normal, 2.0=loud

# --- DUTCH VOICE SETTINGS ---
TTS_SPEED_NL = 1.14           # Speech rate: 0.5=slow, 1.0=normal, 1.5=fast
TTS_PITCH_NL = 1.39           # Pitch: 0.8=lower, 1.0=normal, 1.2=higher
TTS_VOLUME_NL = 0.90         # Volume multiplier: 0.5=quiet, 1.0=normal, 2.0=loud

# --- LANGUAGE DETECTION ---
TTS_LANG_THRESHOLD = 0.15    # Dutch word ratio to trigger NL voice (0.15 = 15%)
FORCE_LANGUAGE = None        # None=auto-detect, "en"=always English, "nl"=always Dutch

# Language switch commands (voice triggers)
LANG_SWITCH_EN = [
    # English commands
    "speak english", "switch to english", "english please", "in english", "english",
    # Dutch commands to switch to English
    "schakel naar engels", "spreek engels", "naar het engels", "in het engels", "engels",
]
LANG_SWITCH_NL = [
    # Dutch
    "spreek nederlands", "schakel naar nederlands", "in het nederlands",
    "nederlands alsjeblieft", "nederlands", "praat nederlands",
    # English commands for Dutch
    "speak dutch", "switch to dutch", "dutch please", "in dutch", "dutch",
    "go dutch", "use dutch", "change to dutch",
]
LANG_SWITCH_AUTO = ["auto language", "automatisch", "automatic language", "auto detect"]

# --- TTS BEHAVIOR ---
TTS_GRACE_PERIOD = 0.3       # Seconds to ignore mic after TTS starts (prevent self-interrupt)
TTS_LOG_LENGTH = 0           # Max chars in TTS log (0 = unlimited, shows full response)

# --- TTS INTERRUPT ---
INTERRUPT_DB = -28           # Volume threshold to trigger interrupt (higher = less sensitive)
INTERRUPT_DURATION = 0.3     # Seconds of sustained volume before TTS stops

# === WAKE WORD ===
WAKE_WORD = "hey_jarvis"    # Options: hey_jarvis, alexa, hey_mycroft, timer, weather
WAKE_THRESHOLD = 0.4       # 0.0-1.0: sensitivity (hoger = minder vals positief)
                            # 0.5 = gevoelig, 0.7 = strenger, 0.8 = heel streng
WAKE_WORD_WARMUP_DELAY = 1.0  # Seconds to wait for audio system warmup

# === STAY AWAKE MODE ===
# After a command, keep listening without requiring wake word
STAY_AWAKE_ENABLED = True       # True = blijf luisteren na commando, False = slaap meteen
STAY_AWAKE_TIMEOUT = 30.0       # 30 Seconden stilte voordat hij alsnog gaat slapen (0 = nooit auto-sleep)
SLEEP_COMMANDS = [ "ga slapen", "go to sleep", "welterusten"]  # Expliciete slaap-commando's

# === CALENDAR ===
CALENDAR_BACKEND = "evolution"  # "evolution" = GNOME/Evolution (syncs with Google), "google" = gcalcli, "local" = ~/.reminders
CALENDAR_ID = "dutchpatriot@gmail.com"  # Calendar name/email or "primary" for default
CALENDAR_DEFAULT_DURATION = 60  # Default event duration in minutes

CALENDAR_MAX_RETRIES = 5        # How many times to ask again if input not understood (0 = no retry)
CALENDAR_CANCEL_WORDS = ["cancel", "stop", "never mind", "annuleer", "stop maar", "laat maar"]
CALENDAR_ASK_LANGUAGE = True    # Ask "English or Dutch?" at start of calendar actions

# Language detection keywords (for calendar language prompt)
CALENDAR_LANG_EN = ["english", "engels", "en"]
CALENDAR_LANG_NL = ["dutch", "nederlands", "nl", "holland", "hollands"]

# --- CALENDAR TRIGGER WORDS ---
# These determine how voice commands are routed to calendar actions

# Words that indicate we're talking about calendar/appointments
CALENDAR_WORDS = [
    # English
    "calendar", "calander", "agenda", "schedule", "event", "meeting", "appointment", "alarm",
    # Dutch
    "afspraak", "afspraken", "afsprake", "vergadering", "bijeenkomst"
]

# Words that start a REMOVE action (checked at start of sentence)
CALENDAR_REMOVE_PREFIXES = ("remove ", "delete ", "verwijder ", "wis ")

# Trigger phrases for CHECK calendar
CALENDAR_CHECK_WORDS = [
    # English
    "what", "check", "show", "list", "today", "tomorrow", "this week", "next week",
    "what's on", "what do i have",
    # Dutch
    "wat", "bekijk", "toon", "welke", "vandaag", "morgen", "deze week", "volgende week",
    "staat er op", "heb ik"
]

# Trigger phrases for CLEAR calendar (delete ALL events on a date)
CALENDAR_CLEAR_WORDS = [
    # English
    "clear", "delete all", "remove all", "empty",
    # Dutch
    "leeg", "wis alles", "verwijder alles", "leeg maken"
]

# Trigger phrases for REMOVE specific event (interactive numbered selection)
CALENDAR_REMOVE_WORDS = [
    # English
    "remove event", "delete event", "cancel event", "remove appointment",
    "remove meeting", "delete meeting", "cancel meeting",
    # Dutch
    "verwijder afspraak", "wis afspraak", "annuleer afspraak", "afspraak verwijderen",
    "verwijder een afspraak", "afspraak wissen", "vergadering verwijderen"
]

# Trigger phrases for ADD event
CALENDAR_ADD_WORDS = [
    # English
    "add", "put", "create", "new", "schedule", "set", "plan", "book",
    # Dutch
    "voeg toe", "toevoegen", "nieuwe", "maak", "zet", "plaats", "inplannen"
]

# Phrases that trigger "remove ALL" in the numbered selection (not just "all" alone!)
CALENDAR_REMOVE_ALL_PHRASES = [
    # English
    "remove all", "delete all", "clear all", "all of them",
    # Dutch
    "alles verwijderen", "verwijder alles", "allemaal verwijderen",
    "alles wissen", "wis alles", "allemaal"
]

# Bilingual calendar prompts (en, nl)
CALENDAR_PROMPTS = {
    # Language selection
    "which_language": ("English or Dutch?", "Engels of Nederlands?"),
    # Add event
    "what_event": ("What is the event?", "Wat is de afspraak?"),
    "what_start_time": ("What time does the event start?", "Hoe laat begint de afspraak?"),
    "what_end_time": ("What time does the event end?", "Hoe laat eindigt de afspraak?"),
    "what_date": ("What date is the event on?", "Op welke datum is de afspraak?"),
    # Check/clear calendar
    "which_date_or_week": ("For which date or week?", "Voor welke datum of week?"),
    "which_date_to_clear": ("For which date or week would you like to clear?", "Welke datum of week wil je wissen?"),
    # Remove event
    "event_name_to_remove": ("What is the name of the event to remove?", "Wat is de naam van de afspraak om te verwijderen?"),
    "event_date_to_remove": ("What date is this event on?", "Op welke datum is deze afspraak?"),
    # Retry prompts
    "retry_time": ("Please say the time again, like '9 AM' or 'three thirty PM'.", "Zeg de tijd opnieuw, zoals '9 uur' of 'half vier'."),
    "retry_date": ("Please say the date again, like 'tomorrow' or 'January 15th'.", "Zeg de datum opnieuw, zoals 'morgen' of '15 januari'."),
    "retry_date_or_week": ("Please say a date like 'today', 'this week', or 'next Friday'.", "Zeg een datum zoals 'vandaag', 'deze week', of 'volgende vrijdag'."),
    # Errors
    "couldnt_understand_time": ("I couldn't understand the time.", "Ik begreep de tijd niet."),
    "couldnt_understand_date": ("I couldn't understand the date.", "Ik begreep de datum niet."),
    "couldnt_understand_week": ("I didn't understand the week query.", "Ik begreep de week vraag niet."),
    "cancelled": ("Okay, cancelled.", "Oké, geannuleerd."),
    "didnt_catch": ("I didn't catch that.", "Ik heb dat niet verstaan."),
    "lets_start_over": ("Let's start over.", "Laten we opnieuw beginnen."),
    "ask_again": ("Let me ask again.", "Ik vraag het nog een keer."),
}

# === DICTATION SLEEP MODE ===
# Words that trigger sleep (mic silenced, no transcription)
# Each phrase must be a SEPARATE item in the list!
DICTATE_SLEEP_WORDS = [
     "ga slapen", "welterusten", "go to sleep"
]
# Wake uses WAKE_WORD above (e.g., "Hey Jarvis") - no transcription while sleeping

# === INLINE CAPS TRIGGERS ===
# Phrases that trigger inline UPPERCASE: "this is a test in all caps" → "THIS IS A TEST"
# Trigger can be at START or END of phrase
INLINE_CAPS_TRIGGERS = [
    # English + Whisper mishearings
    "write in caps", "right in caps", "ride in caps",
    "write in all caps", "right in all caps", "ride in all caps",
    "in all caps", "in caps", "all caps", "all capitals",
    # Dutch
    "schrijf in caps", "schrijf in hoofdletters",
    "in alle caps", "in hoofdletters", "met hoofdletters",
    "alles in hoofdletters", "alle caps"
]

# === TOGGLE CAPS LOCK TRIGGERS ===
# Phrases that toggle the real Caps Lock key (LED lights up!)
# Add Whisper mishearings as you discover them
TOGGLE_CAPS_TRIGGERS = [
    "toggle caps lock", "toggle caps", "toggle hoofdletters",
    "togglecapslock", "togel caps lock", "togel caps", "togelcapslock",
    "doggel caps lock", "doggelcapslock", "tokulkepslogon", "togelkepslog"
]

# === SCROLL SETTINGS ===
# Reverse scroll direction: "scroll up" = content moves UP (like pressing Page Up key)
# False = natural scrolling (content moves in direction of scroll wheel)
# True = traditional scrolling (content moves opposite to scroll wheel - like arrow keys)
SCROLL_REVERSE = True  # Set to False if you prefer natural scrolling

# === DISPLAY ===
USE_EMOJIS = False          # Emojis in terminal output (zet uit bij compatibiliteitsproblemen)
DICTATE_EMOJIS = False      # Emoji replacements in dictation ("heart" → ❤️)

# Log truncation (0 = unlimited, shows full text)
LOG_CMD_LENGTH = 750        # Max chars for command log (e.g., "OLLAMA fallback: ...")
LOG_OUTPUT_LENGTH = 6       # Max chars for terminal command output

# Terminal command execution
TERMINAL_TIMEOUT = 600      # Timeout in seconds (600 = 10 minutes, 0 = no timeout)
                            # Long commands like rsync need more time
TERMINAL_NEW_WINDOW = True  # True = open commands in new terminal window, False = show in same terminal

# === VRAM OPTIMIZATION ===
AUTO_UNLOAD_ENABLED = True      # Auto-unload models after inactivity to free VRAM
AUTO_UNLOAD_TIMEOUT = 60.0      # Seconds of inactivity before unloading (default: 60s)
AUTO_UNLOAD_CHECK_INTERVAL = 10.0  # How often to check for inactive models (seconds)
VRAM_LOW_MEMORY_MODE = False    # Aggressive VRAM saving (unload immediately after use)

# === DEBUG / VERBOSE ===
VERBOSE_SESSION = True     # Print session load/clear messages ("Loaded X messages")
DEBUG_API = True           # Print full API call details (URL, headers, payload, response)

# === INTENT ROUTER ===
# Controls how voice commands are routed to modules
INTENT_CONFIDENCE_THRESHOLD = 0.5   # Minimum confidence to use detected intent (0.0-1.0)
INTENT_FALLBACK_CONFIDENCE = 0.8    # Confidence score for fallback/unknown intents

# === VOICE2JSON (Optional - for intent recognition) ===
V2J_DOCKER_IMAGE = "synesthesiam/voice2json"  # Docker image for voice2json
V2J_DOCKER_CHECK_TIMEOUT = 5    # Timeout for Docker image check (seconds)
V2J_COMMAND_TIMEOUT = 10        # Timeout for voice2json commands (seconds)

# === MODULE-SPECIFIC SETTINGS ===

# -- Coding Module --
CODING_MAX_TOKENS = 2000            # Max tokens for code generation (longer than chat)
CODING_SPEAK_TRUNCATE_LENGTH = 2000  # Truncate spoken output after this many chars

# -- Terminal Module --
TERMINAL_OUTPUT_TRUNCATE_LENGTH = 2000  # Truncate terminal output for TTS
TERMINAL_LARGE_FILE_SIZE = "200M"      #100M  Size threshold for "find large files" command

# -- Dictation Module --
DICTATION_POST_TYPE_DELAY = 1.5     # Seconds to wait after typing (keyboard settle time)
DICTATION_SUBPROCESS_TIMEOUT = 5    # Timeout for dictation subprocess commands

# === SUBPROCESS / EXTERNAL COMMANDS ===
SUBPROCESS_TIMEOUT = 30             # Default timeout for external commands (seconds)
EVOLUTION_TIMEOUT = 30              # Timeout for Evolution calendar client

# === MODEL MANAGER ===
MODEL_MANAGER_HISTORY_WINDOW = 6    # Keep last N messages per module (3 exchanges)
MODEL_MANAGER_HISTORY_MAX = 12      # Trim history if exceeds this

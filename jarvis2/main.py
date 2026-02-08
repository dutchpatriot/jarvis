#!/usr/bin/env python3
"""
Assistmint Voice Assistant - Modular Architecture Entry Point

This is the new thin main.py (~150 lines) that uses the modular Core + Modules architecture.

Usage:
    python main.py [--model MODEL] [--device N] [--voice] [--type]

The modular design:
- Core (always stable): STT, TTS, Wake word, Resource management
- Modules (stackable): Chat, Calendar, Dictation, Terminal
"""

import argparse
import time
import sys

# Core imports
from assistmint.core.logger import cmd, set_use_emojis
from assistmint.core.resources import get_resource_manager
from assistmint.core.modules import get_module_loader, ModuleContext
from assistmint.core.audio import (
    list_microphones,
    get_default_microphone,
    select_microphone_and_samplerate,
    whisper_speech_to_text,
    speak,
    init_wake_word,
    listen_for_wake_word,
    get_stt_engine  # For unloading Whisper when sleeping
)
from assistmint.core.nlp import apply_corrections, is_hallucination, get_intent_router
from assistmint.core.actions import execute_action, is_system_action

# Import config
try:
    from config import USE_EMOJIS, WAKE_WORD, STAY_AWAKE_ENABLED, STAY_AWAKE_TIMEOUT, SLEEP_COMMANDS
    from config import CALENDAR_BACKEND, CALENDAR_ID, CALENDAR_DEFAULT_DURATION
except ImportError:
    USE_EMOJIS = False
    WAKE_WORD = "hey_jarvis"
    STAY_AWAKE_ENABLED = True
    STAY_AWAKE_TIMEOUT = 30.0
    SLEEP_COMMANDS = ["sleep", "slaap", "ga slapen", "go to sleep"]
    CALENDAR_BACKEND = "evolution"
    CALENDAR_ID = "primary"
    CALENDAR_DEFAULT_DURATION = 60

# Initialize calendar with V2's TTS and config
from assistmint.calendar_manager import set_speak_func, set_calendar_config
set_speak_func(speak)
set_calendar_config(
    backend=CALENDAR_BACKEND,
    calendar_id=CALENDAR_ID,
    default_duration=CALENDAR_DEFAULT_DURATION,
)

# Global flags
_quiet_help = False


def init_assistant():
    """Initialize the modular assistant."""
    # Colors for terminal output
    R = "\033[0m"
    B = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    MAGENTA = "\033[95m"
    WHITE = "\033[97m"
    BG_BLUE = "\033[44m"

    # Configure logging
    set_use_emojis(USE_EMOJIS)

    print(f"\n{B}{WHITE}{BG_BLUE}╔══════════════════════════════════════════════════════════════╗")
    print(f"║          {GREEN}★ ASSISTMINT v2.0 ★{WHITE}  Modular Voice Assistant        ║")
    print(f"╚══════════════════════════════════════════════════════════════╝{R}\n")

    # Initialize resource manager (detects GPU)
    print(f"{B}{CYAN}[1/5] Resources{R}")
    rm = get_resource_manager()
    if rm.gpu_available:
        print(f"  {GREEN}✓{R} GPU: {rm.gpu_name}")
        print(f"  {DIM}  VRAM: {rm.gpu_memory_gb:.1f} GB{R}")
    else:
        print(f"  {YELLOW}⚠{R} CPU mode (no GPU detected)")

    # Import config for display
    try:
        from config import WHISPER_MODEL, DEFAULT_MODEL, WAKE_WORD as CFG_WAKE
        print(f"\n{B}{CYAN}[2/5] Models{R}")
        print(f"  {GREEN}✓{R} Whisper STT: {WHISPER_MODEL}")
        print(f"  {GREEN}✓{R} Ollama LLM: {DEFAULT_MODEL}")
        print(f"  {GREEN}✓{R} Wake Word: {CFG_WAKE.replace('_', ' ').title()}")
    except ImportError:
        pass

    # Discover and load modules
    print(f"\n{B}{CYAN}[3/5] Modules{R}")
    loader = get_module_loader()

    # Register built-in modules
    from modules.chat.module import ChatModule
    from modules.calendar.module import CalendarModule
    from modules.dictation.module import DictationModule
    from modules.terminal.module import TerminalModule
    from modules.coding.module import CodingModule
    from modules.project.module import ProjectModule

    loader.register_module(ChatModule)
    loader.register_module(CalendarModule)
    loader.register_module(DictationModule)
    loader.register_module(TerminalModule)
    loader.register_module(CodingModule)
    loader.register_module(ProjectModule)

    # Load all modules
    loaded = loader.load_all_modules()
    for m in loader.get_loaded_modules():
        print(f"  {GREEN}✓{R} {m.name}: {m.description}")

    # TTS voices
    print(f"\n{B}{CYAN}[4/5] TTS Voices{R}")
    try:
        import os
        voice_dir = os.path.expanduser("~/.local/share/piper/voices")
        if os.path.exists(voice_dir):
            voices = [f for f in os.listdir(voice_dir) if f.endswith('.onnx')]
            for v in voices:
                lang = "🇳🇱 Dutch" if "nl_" in v else "🇬🇧 English"
                print(f"  {GREEN}✓{R} {lang}: {v.replace('.onnx', '')}")
    except Exception:
        print(f"  {DIM}(voice detection skipped){R}")

    # Setup VRAM auto-unload
    try:
        from config import AUTO_UNLOAD_ENABLED, AUTO_UNLOAD_TIMEOUT
        from assistmint.core.audio.stt import get_stt_engine
        from assistmint.core.audio.tts import get_tts_engine
        from assistmint.core.resources.manager import ResourceType

        if AUTO_UNLOAD_ENABLED and rm.gpu_available:
            print(f"\n{B}{CYAN}[5/5] VRAM Optimization{R}")

            # Get engine instances
            stt = get_stt_engine()
            tts = get_tts_engine()

            # Register unload callbacks
            rm.register_unload_callback(ResourceType.STT, stt.unload_model)
            rm.register_unload_callback(ResourceType.TTS, tts.unload_voices)

            # Configure and enable auto-unload
            rm.set_unload_timeout(AUTO_UNLOAD_TIMEOUT)
            rm.enable_auto_unload(True)

            print(f"  {GREEN}✓{R} Auto-unload: {AUTO_UNLOAD_TIMEOUT}s timeout")
            vram = rm.get_vram_usage()
            print(f"  {DIM}  Current VRAM: {vram['reserved_mb']}MB / {vram['total_mb']}MB{R}")
    except ImportError:
        pass
    except Exception as e:
        print(f"  {DIM}(VRAM optimization skipped: {e}){R}")

    print(f"\n{DIM}{'─' * 64}{R}\n")

    return loader


def process_command(text: str, context: ModuleContext, loader) -> bool:
    """
    Process a voice command using the modular router.

    Returns True if should continue listening, False to exit.
    """
    # Apply corrections
    text = apply_corrections(text)
    text_lower = text.lower().strip().rstrip('.,!?')

    # Filter hallucinations
    if is_hallucination(text, strict=True):
        print(cmd(f"Skipped hallucination: '{text}'"))
        return True

    # Skip very short noise
    if len(text_lower) <= 2 and text_lower not in ["ok", "hi"]:
        print(cmd(f"Skipped short noise: '{text}'"))
        return True

    print(cmd(f"Processing: {text_lower}"))

    # Check for shutdown command
    shutdown_triggers = [
        "kill yourself", "shut down", "shutdown", "exit", "quit",
        "goodbye jarvis", "bye jarvis", "stop jarvis",
        "sluit af", "afsluiten", "stop jezelf", "doei jarvis"
    ]
    if any(trigger in text_lower for trigger in shutdown_triggers):
        print(cmd("Shutdown requested"))
        speak("Goodbye!", interruptable=False)
        return False  # Exit

    # Check for help command (NOT "commands" - that goes to terminal module)
    help_triggers = ["help me", "what can you do"]
    if any(t in text_lower for t in help_triggers) or text_lower == "help":
        _show_help()
        return True

    # PRIORITY: Clear session (before any other processing)
    clear_session_triggers = ["clear session", "forget everything", "vergeet alles", "wis sessie"]
    if any(t in text_lower for t in clear_session_triggers):
        try:
            from modules.chat.module import clear_session
            clear_session()
            speak("Sessie gewist." if "vergeet" in text_lower or "wis" in text_lower else "Session cleared.", interruptable=False)
        except ImportError:
            speak("Could not clear session.")
        return True

    # PRIORITY: Language switching (before any other processing)
    try:
        from config import LANG_SWITCH_EN, LANG_SWITCH_NL, LANG_SWITCH_AUTO
        from assistmint.core.audio.tts import set_language
        from modules.chat.module import clear_pending_calendar

        if any(t in text_lower for t in LANG_SWITCH_EN):
            set_language("en")
            clear_pending_calendar()  # Clear any pending calendar state
            speak("Switched to English.", interruptable=False)
            return True
        if any(t in text_lower for t in LANG_SWITCH_NL):
            set_language("nl")
            clear_pending_calendar()
            speak("Nederlands.", interruptable=False)
            return True
        if any(t in text_lower for t in LANG_SWITCH_AUTO):
            set_language(None)
            clear_pending_calendar()
            speak("Auto.", interruptable=False)
            return True
    except ImportError:
        pass

    # Update context with text
    context.text = text
    context.text_lower = text_lower

    # Try intent recognition first
    router = get_intent_router()
    intent_result = router.recognize_intent(text_lower, language="auto")
    action = intent_result.get("action", "ollama")

    # Handle system actions (clipboard, browser, keys) directly
    if is_system_action(action):
        response = execute_action(action)
        if response:
            speak(response, interruptable=False)
        return True

    # Route to appropriate module
    result = loader.route(text, context, intent_result.get("intent"))

    if result:
        if result.text:
            # Print full response before speaking
            print(f"\n\033[1m\033[92mJarvis:\033[0m {result.text}\n")
            speak(result.text)
        return True
    else:
        # Fallback to chat module
        chat_module = loader.get_module("chat")
        if chat_module:
            result = chat_module.execute(context)
            if result and result.text:
                # Print full response before speaking
                print(f"\n\033[1m\033[92mJarvis:\033[0m {result.text}\n")
                speak(result.text)

    return True


def _show_help():
    """Display help menu."""
    R = "\033[0m"
    B = "\033[1m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    MAGENTA = "\033[95m"
    BLUE = "\033[94m"
    DIM = "\033[2m"
    WHITE = "\033[97m"
    BG_BLUE = "\033[44m"

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

{YELLOW}  🖥️  PROGRAMMING MODE{R}                   {YELLOW}🖥️  PROGRAMMEER MODUS{R}
     {WHITE}"Let's program"{R}                        {WHITE}"Laten we programmeren"{R}
     {WHITE}"Coding mode"{R}                          {WHITE}"Codeermodus"{R}
     {WHITE}"Save code"{R}                            {WHITE}"Bewaar code"{R}
     {DIM}Opens terminal with tail -f on code file. Code auto-saves as you talk!{R}

{DIM}  ═══════════════════════════════════════════════════════════════════════{R}
""")
    if not _quiet_help:
        speak("Calendar: add to calendar, check my agenda. Dictation: say dictate to type. Programming: say let's program to code together. Or just ask me anything.")


def main():
    """Main entry point."""
    global _quiet_help

    parser = argparse.ArgumentParser(description='Assistmint Voice Assistant')
    parser.add_argument('--model', '-m', help='Ollama model for English')
    parser.add_argument('--model-nl', help='Ollama model for Dutch (enables auto-switch)')
    parser.add_argument('--no-auto-switch', action='store_true', help='Disable auto model switching')
    parser.add_argument('--device', '-d', type=int, help='Audio device index')
    parser.add_argument('--voice', '-v', action='store_true', help='Start in voice mode')
    parser.add_argument('--type', '-t', action='store_true', help='Start in type mode')
    parser.add_argument('--no-commands', '-nc', action='store_true', help='Skip TTS for help command')
    args = parser.parse_args()

    # Set global flags
    _quiet_help = args.no_commands

    # Initialize modular assistant
    loader = init_assistant()

    # Configure models from CLI
    import config
    if args.model:
        config.DEFAULT_MODEL = args.model
        print(f"📗 English model: {args.model}")
    if args.model_nl:
        config.DEFAULT_MODEL_NL = args.model_nl
        config.MODEL_AUTO_SWITCH = True
        print(f"📙 Dutch model: {args.model_nl}")
    if args.no_auto_switch:
        config.MODEL_AUTO_SWITCH = False
        print("🔒 Auto model switching: disabled")
    elif config.MODEL_AUTO_SWITCH:
        print(f"🔄 Auto-switch: EN={config.DEFAULT_MODEL} / NL={config.DEFAULT_MODEL_NL}")

    # Set model on chat module
    chat_module = loader.get_module("chat")
    if chat_module and args.model:
        chat_module.set_model(args.model)

    # Select microphone
    input_devices = list_microphones()
    if args.device is not None:
        selected_device = input_devices[args.device]
        samplerate = int(selected_device['default_samplerate'])
        print(f"Using device {args.device}: {selected_device['name']}")
    else:
        selected_device, samplerate = get_default_microphone(input_devices)

    # Create context
    context = ModuleContext(
        text="",
        text_lower="",
        selected_device=selected_device,
        samplerate=samplerate
    )

    # Determine mode
    mode = 'voice' if args.voice else ('type' if args.type else None)

    while True:
        if mode is None:
            mode = input("Mode [voice/type]: ").strip().lower()

        if mode == "voice":
            context.session_data["input_mode"] = "voice"
            init_wake_word()

            # Show VRAM status
            import subprocess
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    used, total = map(int, result.stdout.strip().split(','))
                    pct = int(used / total * 100)
                    status = "⚠ HIGH" if pct > 80 else "✓"
                    print(f"{status} VRAM: {used}MB/{total}MB ({pct}%)")
            except Exception:
                pass

            print(f"\nVoice mode - say '{WAKE_WORD.replace('_', ' ').title()}' to wake\n")

            while True:
                detected = listen_for_wake_word(selected_device, samplerate)
                if detected is None:
                    time.sleep(1)
                    continue
                if detected:
                    speak("Yes?")

                    # === STAY AWAKE LOOP ===
                    # Keep listening until sleep command or timeout
                    last_activity = time.time()
                    awake = True

                    while awake:
                        transcription = whisper_speech_to_text(selected_device, samplerate, extended_listen=True)

                        if transcription:
                            last_activity = time.time()
                            text_lower = transcription.lower().strip()

                            # Check for explicit sleep command
                            if any(cmd in text_lower for cmd in SLEEP_COMMANDS):
                                speak("Going to sleep.", interruptable=False)
                                awake = False
                                break

                            # Process command (NOT strict hallucination filter in stay-awake)
                            if not process_command(transcription, context, loader):
                                return  # Exit requested

                            # Stay awake for follow-ups
                            if not STAY_AWAKE_ENABLED:
                                awake = False
                                break
                        else:
                            # No transcription - check timeout
                            if STAY_AWAKE_ENABLED and STAY_AWAKE_TIMEOUT > 0:
                                if time.time() - last_activity > STAY_AWAKE_TIMEOUT:
                                    print(f"\n⏰ Timeout ({STAY_AWAKE_TIMEOUT}s) - going to sleep\n")
                                    awake = False
                                    break

                    # Unload Whisper to free GPU memory while sleeping
                    get_stt_engine().unload_model()
                    print(f"\n💤 Back to sleep... say '{WAKE_WORD.replace('_', ' ').title()}' to wake me up\n")

        elif mode == "type":
            context.session_data["input_mode"] = "type"
            print("\nType mode - 'voice' to switch, 'quit' to exit\n")

            while True:
                command = input("You: ").strip()
                if command.lower() == 'quit':
                    speak("Goodbye!", interruptable=False)
                    return
                if command.lower() in ('voice', 'v'):
                    mode = 'voice'
                    break
                if command:
                    if not process_command(command, context, loader):
                        return

        else:
            print("Invalid mode. Choose 'voice' or 'type'.")
            mode = None


if __name__ == "__main__":
    main()

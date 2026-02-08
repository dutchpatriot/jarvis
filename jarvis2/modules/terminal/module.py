"""
Terminal Module - Voice-activated shell commands with aliases.

Features:
- Custom voice aliases (like bash aliases but spoken)
- Show command before execution
- Confirmation required
- Persistent storage of learned commands
- Nice terminal UI
"""

import subprocess
import json
import os
import re
from typing import Optional, List, Dict

from assistmint.core.modules.base import BaseModule, ModuleResult, ModuleContext, ModuleCapability
from assistmint.core.audio.stt import whisper_speech_to_text
from assistmint.core.audio.tts import speak
from assistmint.core.nlp.filters import is_hallucination
from assistmint.core.logger import cmd as log_cmd
from assistmint.core.constants import NATO_ALPHABET, NUMBER_WORDS, TERMINAL_SYMBOLS

# Import config
try:
    from config import LOG_OUTPUT_LENGTH, TERMINAL_TIMEOUT
except ImportError:
    LOG_OUTPUT_LENGTH = 500
    TERMINAL_TIMEOUT = 600  # 10 minutes default

# Aliases file
ALIASES_FILE = os.path.expanduser("~/.assistmint_aliases.json")
COMMANDS_FILE = os.path.expanduser("~/.assistmint/commands.txt")

# Built-in command aliases
BUILTIN_ALIASES = {
    # System info
    "system info": "neofetch || uname -a",
    "disk space": "df -h",
    "disk usage": "du -sh *",
    "memory": "free -h",
    "processes": "ps aux --sort=-%mem | head -20",
    "cpu usage": "top -bn1 | head -10",
    "uptime": "uptime",
    # Network
    "ip address": "ip addr show | grep 'inet '",
    "my ip": "curl -s ifconfig.me",
    "network status": "nmcli general status",
    "ping google": "ping -c 4 google.com",
    "ports": "ss -tuln",
    # Files
    "list files": "ls -la",
    "list": "ls -la",
    "current directory": "pwd",
    "home directory": "ls -la ~",
    "find large files": "find . -type f -size +100M 2>/dev/null",
    # Git
    "git status": "git status",
    "git log": "git log --oneline -10",
    "git branches": "git branch -a",
    "git pull": "git pull",
    # Docker
    "docker containers": "docker ps -a",
    "docker images": "docker images",
    # Updates
    "update system": "sudo apt update && sudo apt upgrade -y",
    "check updates": "apt list --upgradable 2>/dev/null | head -20",
    # Misc
    "weather": "curl -s 'wttr.in?format=3'",
    "date": "date",
    "calendar": "cal",
}


class TerminalModule(BaseModule):
    """
    Terminal module for voice-activated shell commands.

    Features:
    - Built-in command aliases
    - User-defined voice aliases
    - Command preview + confirmation
    - Learn new commands
    """

    def __init__(self):
        super().__init__()
        self._device = None
        self._samplerate = 16000
        self._user_aliases: Dict[str, str] = {}
        self._file_commands: Dict[str, str] = {}  # From commands.txt
        self._numbered_commands: List[tuple] = []  # [(alias, command), ...] for number selection
        self._load_aliases()
        self._load_commands_file()

    @property
    def name(self) -> str:
        return "terminal"

    @property
    def capabilities(self) -> ModuleCapability:
        return (
            ModuleCapability.TEXT_INPUT |
            ModuleCapability.TEXT_OUTPUT |
            ModuleCapability.SYSTEM_ACCESS |
            ModuleCapability.LEARNING
        )

    @property
    def description(self) -> str:
        return "Voice-activated terminal commands with aliases"

    @property
    def triggers(self) -> List[str]:
        return [
            "run command", "terminal", "execute", "shell",
            "run terminal", "command line", "show commands",
            "add command", "learn command", "remove command"
        ]

    @property
    def priority(self) -> int:
        return 75

    def can_handle(self, text: str, intent: Optional[str] = None) -> float:
        """Check if this is a terminal request."""
        text_lower = text.lower()

        if intent == "terminal":
            return 1.0

        # Check for command management
        if any(t in text_lower for t in ["show commands", "list commands", "add command", "learn command", "remove command"]):
            return 0.95

        # Check for general terminal triggers
        if any(t in text_lower for t in ["run command", "terminal", "execute", "shell"]):
            return 0.9

        # Check if it matches a known alias (built-in, user, or file)
        all_aliases = (
            list(BUILTIN_ALIASES.keys()) +
            list(self._user_aliases.keys()) +
            list(self._file_commands.keys())
        )
        for alias in all_aliases:
            if alias in text_lower:
                return 0.85

        return 0.0

    def execute(self, context: ModuleContext) -> ModuleResult:
        """Execute terminal action."""
        self._device = context.selected_device
        self._samplerate = context.samplerate
        text_lower = context.text_lower
        original_text = context.text

        # Check for command management
        if "show commands" in text_lower or "list commands" in text_lower:
            return self._show_commands()

        if "add command" in text_lower or "learn command" in text_lower:
            return self._add_alias()

        if "remove command" in text_lower:
            return self._remove_alias()

        # Try to extract command from original utterance (e.g., "execute ls" -> "ls")
        extracted_cmd = self._extract_command_from_utterance(original_text)

        # Regular command execution
        return self._execute_command_flow(extracted_cmd)

    def _show_commands(self) -> ModuleResult:
        """Show all available commands in a nice format."""
        R = "\033[0m"
        B = "\033[1m"
        CYAN = "\033[96m"
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        MAGENTA = "\033[95m"
        WHITE = "\033[97m"
        DIM = "\033[2m"
        BG_BLUE = "\033[44m"

        print(f"""
{B}{WHITE}{BG_BLUE}╔══════════════════════════════════════════════════════════════════════════════════╗
║                         💻 TERMINAL COMMANDS                                     ║
╚══════════════════════════════════════════════════════════════════════════════════╝{R}
""")
        # Built-in commands
        print(f"{B}{CYAN}  BUILT-IN COMMANDS{R}")
        print(f"{DIM}  {'─'*70}{R}")

        categories = {
            "System": ["system info", "disk space", "memory", "processes", "cpu usage", "uptime"],
            "Network": ["ip address", "my ip", "network status", "ping google", "ports"],
            "Files": ["list files", "current directory", "home directory", "find large files"],
            "Git": ["git status", "git log", "git branches", "git pull"],
            "Docker": ["docker containers", "docker images"],
            "Updates": ["update system", "check updates"],
            "Misc": ["weather", "date", "calendar"],
        }

        for cat, aliases in categories.items():
            print(f"\n  {B}{YELLOW}{cat}:{R}")
            for alias in aliases:
                if alias in BUILTIN_ALIASES:
                    cmd = BUILTIN_ALIASES[alias]
                    if len(cmd) > 40:
                        cmd = cmd[:37] + "..."
                    print(f"    {GREEN}\"{alias}\"{R} → {DIM}{cmd}{R}")

        # File commands (from commands.txt)
        if self._file_commands:
            print(f"\n{B}{MAGENTA}  FILE COMMANDS (~/.assistmint/commands.txt){R}")
            print(f"{DIM}  {'─'*70}{R}")
            for alias, cmd in self._file_commands.items():
                if len(cmd) > 40:
                    cmd_display = cmd[:37] + "..."
                else:
                    cmd_display = cmd
                print(f"    {GREEN}\"{alias}\"{R} → {DIM}{cmd_display}{R}")

        # User commands (learned via voice)
        if self._user_aliases:
            print(f"\n{B}{CYAN}  LEARNED COMMANDS (voice-added){R}")
            print(f"{DIM}  {'─'*70}{R}")
            for alias, cmd in self._user_aliases.items():
                if len(cmd) > 40:
                    cmd = cmd[:37] + "..."
                print(f"    {GREEN}\"{alias}\"{R} → {DIM}{cmd}{R}")

        if not self._user_aliases and not self._file_commands:
            print(f"\n{DIM}  No custom commands yet. Say \"add command\" or edit commands.txt{R}")

        print(f"""
{DIM}  {'─'*70}{R}
  {B}Usage:{R} Say the alias name or "run command" then speak any command.
  {B}Add:{R}   "add command" or "learn command"
  {B}Remove:{R} "remove command"
{DIM}  {'─'*70}{R}
""")

        total_custom = len(self._user_aliases) + len(self._file_commands)
        speak(f"You have {len(BUILTIN_ALIASES)} built-in commands and {total_custom} custom commands.")
        return ModuleResult(text="", success=True)

    def _extract_command_from_utterance(self, text: str) -> Optional[str]:
        """Extract command from utterance like 'execute ls' -> 'ls'."""
        text_lower = text.lower().strip()

        # Trigger words to strip from beginning
        triggers = ["execute", "run command", "run", "terminal", "shell", "command"]

        for trigger in triggers:
            if text_lower.startswith(trigger):
                # Get the rest after the trigger
                remainder = text[len(trigger):].strip()
                if remainder:
                    return remainder

        return None

    def _show_quick_commands(self):
        """Show numbered list of available commands."""
        R = "\033[0m"
        B = "\033[1m"
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        DIM = "\033[2m"
        CYAN = "\033[96m"

        # Build numbered list: file commands, then user aliases, then some built-ins
        self._numbered_commands = []

        # Add file commands
        for alias, cmd in self._file_commands.items():
            self._numbered_commands.append((alias, cmd))

        # Add user aliases
        for alias, cmd in self._user_aliases.items():
            self._numbered_commands.append((alias, cmd))

        # Add a few useful built-ins
        quick_builtins = ["git status", "git pull", "disk space", "memory", "docker containers"]
        for alias in quick_builtins:
            if alias in BUILTIN_ALIASES:
                self._numbered_commands.append((alias, BUILTIN_ALIASES[alias]))

        print(f"\n{B}{CYAN}📋 Commands (say number or name):{R}")
        print(f"{DIM}{'─'*55}{R}")

        for i, (alias, cmd) in enumerate(self._numbered_commands, 1):
            cmd_short = cmd[:35] + "..." if len(cmd) > 35 else cmd
            print(f"  {YELLOW}{i:2}{R}. {GREEN}{alias:<20}{R} {DIM}{cmd_short}{R}")

        print(f"{DIM}{'─'*55}{R}")
        print(f"{DIM}📁 ~/.assistmint/commands.txt  |  Say 'show commands' for full list{R}\n")

    def _resolve_number(self, text: str) -> Optional[str]:
        """Check if text is a command number reference and return the command."""
        text_lower = text.lower().strip()

        # Number words to digits
        word_to_num = {
            "one": 1, "een": 1, "two": 2, "twee": 2, "three": 3, "drie": 3,
            "four": 4, "vier": 4, "five": 5, "vijf": 5, "six": 6, "zes": 6,
            "seven": 7, "zeven": 7, "eight": 8, "acht": 8, "nine": 9, "negen": 9,
            "ten": 10, "tien": 10
        }

        num = None

        # Only match if text IS a number (not contains a number)
        # e.g., "3" or "command 3" or "nummer 3" but NOT "syncvps1"

        # Pure number: "3", "10"
        if text_lower.isdigit():
            num = int(text_lower)

        # Command phrase: "command 3", "run 3", "nummer 3", "execute 3"
        if num is None:
            match = re.match(r'^(?:command|run|execute|nummer|commando)\s+(\d+)$', text_lower)
            if match:
                num = int(match.group(1))

        # Number word as standalone or in phrase: "three", "command three", "drie"
        if num is None:
            for word, n in word_to_num.items():
                # Word must be standalone, not part of another word
                if re.search(rf'\b{word}\b', text_lower):
                    num = n
                    break

        # Resolve to command
        if num and 1 <= num <= len(self._numbered_commands):
            alias, cmd = self._numbered_commands[num - 1]
            print(log_cmd(f"Selected #{num}: {alias}"))
            return cmd

        return None

    def _execute_command_flow(self, pre_extracted_cmd: Optional[str] = None) -> ModuleResult:
        """Main command execution flow with confirmation."""

        # If command already extracted from utterance, use it directly
        if pre_extracted_cmd:
            # Check if it's a number reference first
            num_cmd = self._resolve_number(pre_extracted_cmd)
            if num_cmd:
                text = pre_extracted_cmd
                command = num_cmd
                # Skip to confirmation
                print(f"\n{log_cmd(f'Command (number): {command[:60]}...' if len(command) > 60 else f'Command (number): {command}')}")
                speak(f"Run this command?")
                confirm = whisper_speech_to_text(self._device, self._samplerate).strip().lower().rstrip('.,!?')
                background = any(w in confirm for w in ["background", "achtergrond"])
                if any(w in confirm for w in ["yes", "ja", "yep", "do it", "go ahead", "run it", "okay", "ok"]) or background:
                    if background:
                        speak("Running in background.")
                    return self._run_command(command, background=background)
                else:
                    return ModuleResult(text="Cancelled.", success=True)

            text = pre_extracted_cmd
            print(log_cmd(f"Extracted command: {text}"))
        else:
            print(log_cmd("Terminal command mode"))

            # Show available commands for reference
            self._show_quick_commands()

            speak("What command? Say name or number.")

            # Get command from voice
            text = whisper_speech_to_text(self._device, self._samplerate, extended_listen=True)

            # Check if user said a number
            if text:
                num_cmd = self._resolve_number(text)
                if num_cmd:
                    command = num_cmd
                    print(f"\n{log_cmd(f'Command (number): {command[:60]}...' if len(command) > 60 else f'Command (number): {command}')}")
                    speak(f"Run this command?")
                    confirm = whisper_speech_to_text(self._device, self._samplerate).strip().lower().rstrip('.,!?')
                    background = any(w in confirm for w in ["background", "achtergrond"])
                    if any(w in confirm for w in ["yes", "ja", "yep", "do it", "go ahead", "run it", "okay", "ok"]) or background:
                        if background:
                            speak("Running in background.")
                        return self._run_command(command, background=background)
                    else:
                        return ModuleResult(text="Cancelled.", success=True)

        if not text:
            return ModuleResult(text="I didn't hear a command.", success=False)

        # Skip hallucinations
        if is_hallucination(text.lower().strip().rstrip('.'), strict=False):
            print(log_cmd(f"Skipped hallucination: '{text}'"))
            return ModuleResult(text="I didn't catch that.", success=False)

        # Try to resolve as alias FIRST (before symbol processing)
        # This prevents "disk space" becoming "disk " before alias lookup
        command = self._resolve_alias(text)

        if command:
            source = "alias"
        else:
            # Only process symbols if not an alias
            processed_text = self._process_command_text(text)
            command = processed_text
            source = "direct"

        # Show the command to user
        print(f"\n{log_cmd(f'Command ({source}): {command}')}")

        # Ask for confirmation
        speak(f"Run {self._make_speakable(command)}?")

        confirm = whisper_speech_to_text(self._device, self._samplerate).strip().lower()
        confirm = confirm.rstrip('.,!?')

        # Check for background execution request
        background = any(w in confirm for w in ["background", "achtergrond", "in background", "op achtergrond"])

        # Check confirmation
        if any(w in confirm for w in ["yes", "ja", "yep", "do it", "go ahead", "run it", "okay", "ok"]) or background:
            if background:
                speak("Running in background.")
            return self._run_command(command, background=background)
        else:
            return ModuleResult(text="Cancelled.", success=True)

    def _resolve_alias(self, text: str) -> Optional[str]:
        """Try to resolve text as a command alias."""
        text_lower = text.lower().strip()

        # Check user aliases first (highest priority)
        for alias, cmd in self._user_aliases.items():
            if alias.lower() in text_lower or text_lower in alias.lower():
                return cmd

        # Check file commands (from commands.txt)
        for alias, cmd in self._file_commands.items():
            if alias.lower() in text_lower or text_lower in alias.lower():
                print(log_cmd(f"Matched file command: {alias}"))
                return cmd

        # Check built-in aliases
        for alias, cmd in BUILTIN_ALIASES.items():
            if alias.lower() in text_lower or text_lower in alias.lower():
                return cmd

        return None

    def _process_command_text(self, text: str) -> str:
        """Process command text with NATO spelling and symbols."""
        # Case instructions
        text = re.sub(r'\b(capital|uppercase)\s+(\w+)', lambda m: m.group(2).upper(), text, flags=re.IGNORECASE)
        text = re.sub(r'\b(lowercase)\s+(\w+)', lambda m: m.group(2).lower(), text, flags=re.IGNORECASE)

        # NATO alphabet spelling
        for word, letter in NATO_ALPHABET.items():
            text = re.sub(r'\b(upper|capital)\s+' + word + r'\b', lambda m, l=letter: l.upper(), text, flags=re.IGNORECASE)
            text = re.sub(r'\b(lower)\s+' + word + r'\b', lambda m, l=letter: l, text, flags=re.IGNORECASE)
            text = re.sub(r'\bletter\s+' + word + r'\b', lambda m, l=letter: l, text, flags=re.IGNORECASE)

        # Direct letter spelling
        text = re.sub(r'\b(upper|capital)\s+([a-z])\b', lambda m: m.group(2).upper(), text, flags=re.IGNORECASE)
        text = re.sub(r'\b(lower)\s+([a-z])\b', lambda m: m.group(2).lower(), text, flags=re.IGNORECASE)

        # Numbers
        for word, digit in NUMBER_WORDS.items():
            text = re.sub(r'\b(number|digit)\s+' + word + r'\b', lambda m, d=digit: d, text, flags=re.IGNORECASE)

        # Symbols
        for word, symbol in TERMINAL_SYMBOLS.items():
            text = re.sub(r'\b' + re.escape(word) + r'\b', lambda m, s=symbol: s, text, flags=re.IGNORECASE)

        return text.strip()

    def _make_speakable(self, command: str) -> str:
        """Make command speakable (remove special chars, truncate)."""
        # Remove special chars that TTS can't handle well
        safe = ''.join(c for c in command if ord(c) < 128 and c not in '.?!,;:|<>{}[]')
        safe = safe.strip()[:50]  # Truncate for speech
        return safe if safe else "this command"

    def _run_command(self, command: str, background: bool = False) -> ModuleResult:
        """Execute the command and return result."""
        print(log_cmd(f"Executing{'(background)' if background else ''}: {command}"))

        if background:
            # Run in background with output to log file + popup tail window
            try:
                import time
                timestamp = int(time.time())
                log_file = f"/tmp/assistmint_bg_{timestamp}.log"

                # Start the command in background, output to log file
                subprocess.Popen(
                    f"nohup {command} > {log_file} 2>&1 &",
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True
                )

                # Open a terminal window with tail -f on the log
                # Try different terminal emulators
                terminal_cmd = None
                for term in ["gnome-terminal", "xfce4-terminal", "konsole", "xterm"]:
                    if subprocess.run(["which", term], capture_output=True).returncode == 0:
                        if term == "gnome-terminal":
                            terminal_cmd = f"{term} --title='Background: {command[:30]}' -- tail -f {log_file}"
                        elif term == "xfce4-terminal":
                            terminal_cmd = f"{term} --title='Background: {command[:30]}' -e 'tail -f {log_file}'"
                        elif term == "konsole":
                            terminal_cmd = f"{term} --title 'Background: {command[:30]}' -e tail -f {log_file}"
                        else:  # xterm
                            terminal_cmd = f"{term} -title 'Background: {command[:30]}' -e tail -f {log_file}"
                        break

                if terminal_cmd:
                    subprocess.Popen(terminal_cmd, shell=True, start_new_session=True)
                    print(log_cmd(f"Background started with tail window: {log_file}"))
                else:
                    print(log_cmd(f"Background started (no terminal found): {log_file}"))

                return ModuleResult(
                    text=f"Running in background. Log: {log_file}",
                    success=True
                )
            except Exception as e:
                return ModuleResult(text=f"Failed to start background: {e}", success=False)

        try:
            # timeout=None means no timeout, 0 in config means no timeout
            timeout_val = TERMINAL_TIMEOUT if TERMINAL_TIMEOUT > 0 else None
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout_val
            )

            output = result.stdout.strip() or result.stderr.strip()

            if result.returncode == 0:
                # Show output
                if output:
                    if LOG_OUTPUT_LENGTH > 0 and len(output) > LOG_OUTPUT_LENGTH:
                        print(f"\n{output[:LOG_OUTPUT_LENGTH]}...\n[truncated]")
                    else:
                        print(f"\n{output}\n")

                # Speak summary
                if len(output) > 200:
                    speech = f"Done. Output: {output[:150]}... and more."
                else:
                    speech = f"Done. {output}" if output else "Command completed."

                return ModuleResult(text=speech, success=True)
            else:
                print(f"\n[ERROR] {output}\n")
                return ModuleResult(
                    text=f"Command failed: {output[:100]}" if output else "Command failed.",
                    success=False
                )

        except subprocess.TimeoutExpired:
            mins = TERMINAL_TIMEOUT // 60
            return ModuleResult(text=f"Command timed out after {mins} minutes. Use 'background' for long tasks.", success=False)
        except Exception as e:
            return ModuleResult(text=f"Error: {str(e)}", success=False)

    # === Alias Management ===

    def _load_aliases(self):
        """Load user aliases from file."""
        if os.path.exists(ALIASES_FILE):
            try:
                with open(ALIASES_FILE, 'r') as f:
                    self._user_aliases = json.load(f)
            except json.JSONDecodeError:
                self._user_aliases = {}

    def _save_aliases(self):
        """Save user aliases to file."""
        with open(ALIASES_FILE, 'w') as f:
            json.dump(self._user_aliases, f, indent=2)

    def _load_commands_file(self):
        """Load commands from ~/.assistmint/commands.txt file."""
        if not os.path.exists(COMMANDS_FILE):
            return

        try:
            with open(COMMANDS_FILE, 'r') as f:
                lines = f.readlines()

            current_command = []
            current_trigger = None

            for line in lines:
                line = line.rstrip('\n')

                # Skip comments and empty lines
                if not line.strip() or line.strip().startswith('#'):
                    continue

                # Check for line continuation (ends with \)
                if line.rstrip().endswith('\\'):
                    if '|' in line and current_trigger is None:
                        # First line with trigger
                        parts = line.split('|', 1)
                        current_trigger = parts[0].strip().lower()
                        current_command.append(parts[1].rstrip('\\').strip())
                    else:
                        # Continuation line
                        current_command.append(line.rstrip('\\').strip())
                    continue

                # Complete line (no continuation)
                if current_trigger is not None:
                    # Finish multi-line command
                    current_command.append(line.strip())
                    self._file_commands[current_trigger] = ' '.join(current_command)
                    current_trigger = None
                    current_command = []
                elif '|' in line:
                    # Single line command
                    parts = line.split('|', 1)
                    trigger = parts[0].strip().lower()
                    command = parts[1].strip()
                    self._file_commands[trigger] = command

            # Handle any remaining multi-line command
            if current_trigger and current_command:
                self._file_commands[current_trigger] = ' '.join(current_command)

            if self._file_commands:
                print(log_cmd(f"Loaded {len(self._file_commands)} commands from commands.txt"))

        except Exception as e:
            print(log_cmd(f"Error loading commands.txt: {e}"))

    def reload_commands(self):
        """Reload commands from file (useful after editing)."""
        self._file_commands = {}
        self._load_commands_file()
        return len(self._file_commands)

    def _add_alias(self) -> ModuleResult:
        """Add a new voice command alias."""
        print(log_cmd("Adding new command alias"))

        # Get alias name
        speak("What do you want to call this command?")
        alias = whisper_speech_to_text(self._device, self._samplerate).strip()

        if not alias or is_hallucination(alias, strict=False):
            return ModuleResult(text="I didn't catch the name.", success=False)

        alias = alias.lower().rstrip('.,!?')
        print(log_cmd(f"Alias name: {alias}"))

        # Get the command
        speak("What command should it run? You can spell with NATO alphabet.")
        command = whisper_speech_to_text(self._device, self._samplerate, extended_listen=True).strip()

        if not command:
            return ModuleResult(text="I didn't catch the command.", success=False)

        command = self._process_command_text(command)
        print(log_cmd(f"Command: {command}"))

        # Confirm
        speak(f"So '{alias}' will run '{self._make_speakable(command)}'. Correct?")
        confirm = whisper_speech_to_text(self._device, self._samplerate).strip().lower()

        if any(w in confirm for w in ["yes", "ja", "yep", "correct", "right", "okay"]):
            self._user_aliases[alias] = command
            self._save_aliases()
            print(log_cmd(f"Saved: '{alias}' → '{command}'"))
            return ModuleResult(text=f"Saved. Say '{alias}' to run it.", success=True)
        else:
            return ModuleResult(text="Cancelled.", success=True)

    def _remove_alias(self) -> ModuleResult:
        """Remove a voice command alias."""
        if not self._user_aliases:
            return ModuleResult(text="You don't have any custom commands yet.", success=True)

        print(log_cmd("Removing command alias"))
        speak("Which command do you want to remove?")

        alias = whisper_speech_to_text(self._device, self._samplerate).strip().lower()
        alias = alias.rstrip('.,!?')

        if alias in self._user_aliases:
            del self._user_aliases[alias]
            self._save_aliases()
            return ModuleResult(text=f"Removed '{alias}'.", success=True)
        else:
            # Try fuzzy match
            for stored_alias in self._user_aliases:
                if alias in stored_alias or stored_alias in alias:
                    speak(f"Did you mean '{stored_alias}'?")
                    confirm = whisper_speech_to_text(self._device, self._samplerate).strip().lower()
                    if any(w in confirm for w in ["yes", "ja", "yep"]):
                        del self._user_aliases[stored_alias]
                        self._save_aliases()
                        return ModuleResult(text=f"Removed '{stored_alias}'.", success=True)

            return ModuleResult(text=f"Command '{alias}' not found.", success=False)

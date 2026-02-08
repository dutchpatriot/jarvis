"""
Coding Module - Voice pair programming with specialized code model.

Triggers: "join me", "help me code", "code with me", "pair program"

Features:
- Uses qwen2.5-coder model (or configured coding model)
- Continuous conversation mode (no wake word needed)
- File context awareness - load files and discuss them
- Code generation with diff preview
- Apply changes with confirmation

Commands while in coding mode:
- "open [file]" - Load file into context
- "explain this" / "what does this do" - Explain loaded code
- "write function [description]" - Generate code
- "fix this" / "suggest fix" - Get fix suggestions
- "apply changes" / "do it" - Apply suggested changes
- "done" / "stop" / "exit" - Exit coding mode
"""

import os
import re
import difflib
from typing import Optional, List, Dict, Tuple

from assistmint.core.modules.base import BaseModule, ModuleResult, ModuleContext, ModuleCapability
from assistmint.core.audio.stt import whisper_speech_to_text
from assistmint.core.audio.tts import speak
from assistmint.core.models.manager import get_model_manager
from assistmint.core.nlp.filters import is_hallucination


# ANSI colors for terminal output
R = "\033[0m"      # Reset
B = "\033[1m"      # Bold
DIM = "\033[2m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
WHITE = "\033[97m"
BG_BLUE = "\033[44m"
BG_GREEN = "\033[42m"


# System prompt for coding assistant
CODING_SYSTEM_PROMPT = """You are a skilled programming assistant helping with voice pair programming.

RULES:
1. Be concise - the user is speaking, not typing. Short, clear responses.
2. When showing code, use markdown code blocks with language tags.
3. When suggesting changes, clearly explain what you'll change and why.
4. If asked to generate code, provide working, production-ready code.
5. For file edits, describe the change first, then provide the new code.

CONTEXT FORMAT:
When files are loaded, they appear as:
[FILE: path/to/file.py]
```python
# code content
```

Use line numbers when referencing code (e.g., "line 42").

RESPONSE STYLE:
- Explain briefly WHAT you're doing
- Show the code
- Ask if user wants to apply/continue
"""


class CodingModule(BaseModule):
    """
    Interactive coding assistant for voice pair programming.

    Enters continuous mode - no wake word needed during session.
    Uses code-optimized model (qwen2.5-coder or configured).
    Can read files, explain code, generate solutions, and apply changes.
    """

    def __init__(self):
        super().__init__()
        self._device = None
        self._samplerate = 16000
        self._loaded_files: Dict[str, str] = {}  # path -> content
        self._current_file: Optional[str] = None
        self._pending_changes: Optional[Dict] = None  # {file, old, new}
        self._context: List[Dict] = []  # Conversation context

    @property
    def name(self) -> str:
        return "coding"

    @property
    def capabilities(self) -> ModuleCapability:
        return (
            ModuleCapability.TEXT_INPUT |
            ModuleCapability.TEXT_OUTPUT |
            ModuleCapability.EXTERNAL_API |
            ModuleCapability.MULTI_TURN |
            ModuleCapability.CONTINUOUS |  # Skip wake word in session
            ModuleCapability.SYSTEM_ACCESS  # Can read/write files
        )

    @property
    def description(self) -> str:
        return "Voice pair programming assistant"

    @property
    def triggers(self) -> List[str]:
        return [
            "join me", "code with me", "help me code",
            "pair program", "coding mode", "programming mode",
            "let's code", "lets code", "start coding"
        ]

    @property
    def priority(self) -> int:
        return 85  # High priority for explicit triggers

    def can_handle(self, text: str, intent: Optional[str] = None) -> float:
        """Check if this is a coding mode trigger."""
        text_lower = text.lower()

        if intent == "coding":
            return 1.0

        for trigger in self.triggers:
            if trigger in text_lower:
                return 0.95

        return 0.0

    def execute(self, context: ModuleContext) -> ModuleResult:
        """Start coding session."""
        self._device = context.selected_device
        self._samplerate = context.samplerate

        # Clear previous session state
        self._loaded_files.clear()
        self._current_file = None
        self._pending_changes = None
        self._context.clear()

        # Check if Ollama is available
        manager = get_model_manager()
        if not manager.is_ollama_available():
            speak("Ollama is not running. Please start it first.")
            return ModuleResult(text="Ollama not available", success=False)

        # Show welcome and enter coding loop
        self._show_welcome()
        speak("Coding mode activated. What are you working on?")

        return self._coding_loop()

    def _show_welcome(self):
        """Display coding mode welcome screen."""
        model = get_model_manager().get_model_for_module("coding")
        print(f"""
{B}{WHITE}{BG_BLUE}{'=' * 70}{R}
{B}{WHITE}{BG_BLUE}  CODE MODE - Voice Pair Programming                                  {R}
{B}{WHITE}{BG_BLUE}{'=' * 70}{R}

  {B}{CYAN}Model:{R} {model}

  {B}{YELLOW}Commands:{R}
    {GREEN}"open [file]"{R}      - Load a file for context
    {GREEN}"explain this"{R}     - Explain the loaded code
    {GREEN}"write [desc]"{R}     - Generate code from description
    {GREEN}"fix this"{R}         - Suggest fixes for issues
    {GREEN}"apply" / "do it"{R}  - Apply pending changes
    {GREEN}"type"{R}             - Type input (flags, vars, commands)
    {GREEN}"done" / "stop"{R}    - Exit coding mode

  {DIM}Speak naturally - no wake word needed in this mode.{R}
  {DIM}Say "type" to switch to keyboard for technical input.{R}
{DIM}{'─' * 70}{R}
""")

    def _coding_loop(self) -> ModuleResult:
        """Main coding conversation loop."""
        manager = get_model_manager()

        while True:
            # Listen (continuous mode - no wake word)
            print(f"\n{DIM}[CODING] Listening...{R}")
            text = whisper_speech_to_text(
                self._device,
                self._samplerate,
                extended_listen=True
            )

            if not text:
                continue

            text = text.strip()
            text_lower = text.lower().rstrip('.,!?')

            # Skip hallucinations
            if is_hallucination(text_lower, strict=False):
                print(f"{DIM}[CODING] Skipped hallucination: {text[:50]}{R}")
                continue

            print(f"{CYAN}You:{R} {text}")

            # Check for exit commands
            if text_lower in ["done", "stop", "exit", "quit", "klaar", "stop coding", "end coding"]:
                speak("Ending coding session. Good work!")
                self._cleanup()
                return ModuleResult(text="Coding session ended.", success=True)

            # Check for type mode (keyboard input for technical stuff)
            if text_lower in ["type", "type mode", "keyboard", "type input", "typ", "typen"]:
                typed_text = self._get_typed_input()
                if typed_text:
                    text = typed_text
                    text_lower = text.lower().rstrip('.,!?')
                    print(f"{CYAN}You (typed):{R} {text}")
                else:
                    continue

            # Check for file operations
            if text_lower.startswith("open "):
                filepath = text[5:].strip()
                self._load_file(filepath)
                continue

            # Check for apply changes
            if text_lower in ["apply", "apply it", "do it", "apply changes", "yes apply", "save it"]:
                if self._pending_changes:
                    self._apply_pending_changes()
                else:
                    speak("No pending changes to apply.")
                continue

            # Check for cancel pending changes
            if text_lower in ["cancel", "no", "nevermind", "never mind", "don't apply"]:
                if self._pending_changes:
                    self._pending_changes = None
                    speak("Changes cancelled.")
                continue

            # Check for show diff
            if text_lower in ["show diff", "show changes", "what changed"]:
                if self._pending_changes:
                    self._show_diff(
                        self._pending_changes["old"],
                        self._pending_changes["new"],
                        self._pending_changes["file"]
                    )
                else:
                    speak("No pending changes.")
                continue

            # Build context and ask LLM
            prompt = self._build_prompt(text)

            print(f"{DIM}[CODING] Thinking...{R}")
            response = manager.ask(
                question=prompt,
                module_name="coding",
                system_prompt=CODING_SYSTEM_PROMPT,
                temperature=0.3,  # Lower temperature for code
                max_tokens=2000
            )

            if response:
                # Check if response contains code blocks
                self._process_response(response, text)
            else:
                speak("Sorry, I couldn't process that.")

        return ModuleResult(text="Coding session ended.", success=True)

    def _load_file(self, filepath: str):
        """Load a file into context."""
        # Handle relative paths
        if not filepath.startswith('/') and not filepath.startswith('~'):
            filepath = os.path.join(os.getcwd(), filepath)

        expanded = os.path.expanduser(filepath)

        if not os.path.exists(expanded):
            speak(f"File not found: {filepath}")
            print(f"{RED}File not found: {expanded}{R}")
            return

        try:
            with open(expanded, 'r', encoding='utf-8') as f:
                content = f.read()

            self._loaded_files[expanded] = content
            self._current_file = expanded

            # Get file stats
            lines = content.count('\n') + 1
            ext = os.path.splitext(expanded)[1]
            filename = os.path.basename(expanded)

            # Show file preview
            print(f"\n{GREEN}Loaded: {filename}{R} ({lines} lines)")
            preview_lines = content.split('\n')[:15]
            for i, line in enumerate(preview_lines, 1):
                print(f"{DIM}{i:4}{R} {line[:100]}")
            if lines > 15:
                print(f"{DIM}  ... ({lines - 15} more lines){R}")

            speak(f"Loaded {filename}. {lines} lines. What would you like to do?")

        except Exception as e:
            speak(f"Error reading file: {str(e)}")
            print(f"{RED}Error: {e}{R}")

    def _build_prompt(self, user_input: str) -> str:
        """Build prompt with file context."""
        parts = []

        # Add loaded files as context
        if self._loaded_files:
            for path, content in self._loaded_files.items():
                filename = os.path.basename(path)
                ext = os.path.splitext(path)[1].lstrip('.')
                # Truncate very long files
                if len(content) > 8000:
                    content = content[:8000] + "\n... (truncated)"
                parts.append(f"[FILE: {filename}]\n```{ext}\n{content}\n```")

        # Add user input
        parts.append(f"\nUser request: {user_input}")

        return "\n\n".join(parts)

    def _process_response(self, response: str, original_request: str):
        """Process LLM response, extract code blocks, handle changes."""
        # Display the response
        print(f"\n{MAGENTA}Assistant:{R}")
        print(response)

        # Extract code blocks for potential application
        code_blocks = self._extract_code_blocks(response)

        if code_blocks and self._current_file:
            # Check if this looks like a replacement/edit suggestion
            edit_keywords = ["change", "replace", "modify", "update", "fix", "edit", "add"]
            is_edit = any(kw in original_request.lower() for kw in edit_keywords)

            if is_edit and len(code_blocks) == 1:
                # Store as pending change
                self._pending_changes = {
                    "file": self._current_file,
                    "old": self._loaded_files[self._current_file],
                    "new": code_blocks[0]["code"]
                }
                speak("I've prepared changes. Say 'apply' to save, or 'show diff' to preview.")
            else:
                # Just explain
                speak_text = self._make_speakable(response)
                if len(speak_text) > 300:
                    speak(speak_text[:280] + "... See the full response above.")
                else:
                    speak(speak_text)
        else:
            # No code blocks, just speak the response
            speak_text = self._make_speakable(response)
            if len(speak_text) > 300:
                speak(speak_text[:280] + "... See the full response above.")
            else:
                speak(speak_text)

    def _extract_code_blocks(self, text: str) -> List[Dict]:
        """Extract code blocks from markdown response."""
        blocks = []
        pattern = r'```(\w*)\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)

        for lang, code in matches:
            blocks.append({
                "language": lang or "text",
                "code": code.strip()
            })

        return blocks

    def _apply_pending_changes(self):
        """Apply pending changes to file."""
        if not self._pending_changes:
            speak("No pending changes.")
            return

        filepath = self._pending_changes["file"]
        new_content = self._pending_changes["new"]

        try:
            # Create backup
            backup_path = filepath + ".bak"
            with open(filepath, 'r') as f:
                original = f.read()
            with open(backup_path, 'w') as f:
                f.write(original)

            # Write new content
            with open(filepath, 'w') as f:
                f.write(new_content)

            # Update loaded file
            self._loaded_files[filepath] = new_content
            self._pending_changes = None

            speak("Changes applied. Backup saved.")
            print(f"{GREEN}Changes saved to {filepath}{R}")
            print(f"{DIM}Backup at {backup_path}{R}")

        except Exception as e:
            speak(f"Error saving file: {str(e)}")
            print(f"{RED}Error: {e}{R}")

    def _show_diff(self, old: str, new: str, filepath: str):
        """Show unified diff between old and new content."""
        filename = os.path.basename(filepath)
        old_lines = old.splitlines(keepends=True)
        new_lines = new.splitlines(keepends=True)

        diff = difflib.unified_diff(
            old_lines, new_lines,
            fromfile=f"{filename} (original)",
            tofile=f"{filename} (modified)"
        )

        print(f"\n{B}Diff for {filename}:{R}")
        for line in diff:
            if line.startswith('+') and not line.startswith('+++'):
                print(f"{GREEN}{line.rstrip()}{R}")
            elif line.startswith('-') and not line.startswith('---'):
                print(f"{RED}{line.rstrip()}{R}")
            elif line.startswith('@@'):
                print(f"{CYAN}{line.rstrip()}{R}")
            else:
                print(line.rstrip())

    def _make_speakable(self, text: str) -> str:
        """Clean text for TTS - remove code blocks and markdown."""
        # Remove code blocks
        text = re.sub(r'```[\s\S]*?```', '[code block]', text)
        # Remove inline code
        text = re.sub(r'`[^`]+`', '', text)
        # Remove markdown headers
        text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
        # Clean up whitespace
        text = re.sub(r'\n\s*\n', '. ', text)
        text = re.sub(r'\n', ' ', text)
        return text.strip()

    def _cleanup(self):
        """Cleanup when exiting coding mode."""
        self._loaded_files.clear()
        self._current_file = None
        self._pending_changes = None
        self._context.clear()
        # Clear model history for this module
        get_model_manager().clear_history("coding")

    def _get_typed_input(self) -> Optional[str]:
        """Get typed input from keyboard for technical stuff like flags, vars, commands."""
        print(f"\n{B}{YELLOW}TYPE MODE{R} - Enter your text (flags, commands, code snippets):")
        print(f"{DIM}Press Enter when done, or type 'cancel' to abort.{R}")
        print(f"{GREEN}>{R} ", end="", flush=True)

        try:
            typed = input().strip()
            if typed.lower() in ["cancel", "abort", "nevermind"]:
                print(f"{DIM}Cancelled.{R}")
                return None
            if not typed:
                print(f"{DIM}Empty input, continuing with voice.{R}")
                return None
            return typed
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}Input cancelled.{R}")
            return None

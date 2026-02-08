"""
Project Module - Voice-driven project exploration with multi-file LLM context.

Triggers: "project mode", "scan project", "explore project"

Features:
- Scan project directory, index files/classes/functions
- Fuzzy-find files by partial name or symbol
- Load multiple files into LLM context
- ASCII project tree display
- Git status, diff, log integration
- Continuous conversation mode (no wake word)

Commands while in project mode:
- "show project" / "project tree"   - Display project structure
- "find [name]"                     - Search for class/function/file
- "open [file]"                     - Fuzzy-match and load file
- "also open [file]" / "add [file]" - Load additional file
- "close [file]" / "close all"      - Remove file(s) from context
- "what's loaded"                   - List loaded files
- "git status" / "what changed"     - Show git status
- "git diff" / "show changes"       - Show git diff
- "git log" / "recent commits"      - Show recent commits
- "rescan"                          - Re-index project
- "type"                            - Switch to keyboard input
- "done" / "stop" / "exit"          - Exit project mode
"""

import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict

from assistmint.core.modules.base import BaseModule, ModuleResult, ModuleContext, ModuleCapability
from assistmint.core.audio.stt import whisper_speech_to_text
from assistmint.core.audio.tts import speak
from assistmint.core.models.manager import get_model_manager
from assistmint.core.nlp.filters import is_hallucination


# ANSI colors
R = "\033[0m"
B = "\033[1m"
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


# System prompt for project-aware assistant
PROJECT_SYSTEM_PROMPT = """You are a skilled programming assistant with full project awareness.

RULES:
1. Be concise - the user is speaking, not typing. Short, clear responses.
2. When showing code, use markdown code blocks with language tags.
3. You have access to the project structure and loaded files.
4. Reference files by name and line numbers when discussing code.
5. When asked about the project, use the provided context.

CONTEXT FORMAT:
Project info appears as [PROJECT: ...] with file counts and structure.
Loaded files appear as [FILE: path] with their content.

RESPONSE STYLE:
- Answer questions about the project structure and code
- Explain relationships between files
- Suggest improvements with specific file references
- Be direct and factual
"""


@dataclass
class FileInfo:
    """Indexed file information."""
    rel_path: str
    abs_path: str
    size_bytes: int
    lines: int
    extension: str
    classes: List[str] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)


@dataclass
class ProjectIndex:
    """Project-wide file index."""
    root: str
    files: Dict[str, FileInfo] = field(default_factory=dict)
    scanned_at: float = 0.0
    git_available: bool = False


class ProjectModule(BaseModule):
    """
    Voice-driven project exploration and multi-file LLM context.

    Enters continuous mode - no wake word needed during session.
    Scans project, indexes symbols, supports fuzzy file finding,
    git integration, and multi-file LLM context.
    """

    def __init__(self):
        super().__init__()
        self._device = None
        self._samplerate = 16000
        self._input_mode = "type"  # "type" or "voice"
        self._index: Optional[ProjectIndex] = None
        self._loaded_files: Dict[str, str] = {}  # abs_path -> content
        self._config = self._load_config()

    @property
    def name(self) -> str:
        return "project"

    @property
    def capabilities(self) -> ModuleCapability:
        return (
            ModuleCapability.TEXT_INPUT |
            ModuleCapability.TEXT_OUTPUT |
            ModuleCapability.EXTERNAL_API |
            ModuleCapability.MULTI_TURN |
            ModuleCapability.CONTINUOUS |
            ModuleCapability.SYSTEM_ACCESS
        )

    @property
    def description(self) -> str:
        return "Project exploration and multi-file context"

    @property
    def triggers(self) -> List[str]:
        return [
            "project mode", "scan project", "explore project",
            "project explorer", "open project", "show project",
        ]

    @property
    def priority(self) -> int:
        return 86

    def can_handle(self, text: str, intent: Optional[str] = None) -> float:
        text_lower = text.lower()
        if intent == "project":
            return 1.0
        for trigger in self.triggers:
            if trigger in text_lower:
                return 0.95
        return 0.0

    def execute(self, context: ModuleContext) -> ModuleResult:
        self._device = context.selected_device
        self._samplerate = context.samplerate

        # Detect input mode from session data (set by main.py)
        self._input_mode = context.session_data.get("input_mode", "type")

        # Clear previous session
        self._loaded_files.clear()
        self._index = None

        # Check Ollama
        manager = get_model_manager()
        if not manager.is_ollama_available():
            speak("Ollama is not running. Please start it first.")
            return ModuleResult(text="Ollama not available", success=False)

        # Determine project root
        root = self._config["project_root"] or os.getcwd()

        # Auto-scan
        if self._config["auto_scan"]:
            self._scan_project(root)

        self._show_welcome()

        if self._index and self._index.files:
            count = len(self._index.files)
            msg = f"Project mode activated. Scanned {count} files."
            print(f"\n{GREEN}{msg}{R}")
            speak(msg + " What would you like to explore?")
        else:
            msg = "Project mode activated. No files indexed yet."
            print(f"\n{YELLOW}{msg}{R}")
            speak(msg + " Say scan project to scan.")

        return self._project_loop()

    # ── Config ───────────────────────────────────────────────────────

    def _load_config(self) -> dict:
        """Load config with defaults."""
        defaults = {
            "model": "qwen2.5:3b",
            "max_tokens": 4000,
            "context_max_chars": 32000,
            "project_root": None,
            "auto_scan": True,
            "max_file_size": 100_000,
            "scan_extensions": [
                ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java",
                ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php",
                ".yaml", ".yml", ".toml", ".json", ".md", ".txt", ".cfg", ".ini",
                ".sh", ".bash", ".html", ".css", ".sql",
            ],
            "ignore_dirs": [
                ".git", "__pycache__", "node_modules", "venv", ".venv", "env",
                ".mypy_cache", ".pytest_cache", ".tox", "dist", "build",
                ".eggs", "*.egg-info", ".idea", ".vscode",
            ],
            "git_timeout": 10,
        }
        try:
            from config import DEFAULT_MODEL
            defaults["model"] = DEFAULT_MODEL
        except ImportError:
            pass
        try:
            from config import (
                PROJECT_MAX_TOKENS, PROJECT_CONTEXT_MAX_CHARS,
                PROJECT_ROOT, PROJECT_AUTO_SCAN,
                PROJECT_MAX_FILE_SIZE, PROJECT_SCAN_EXTENSIONS,
                PROJECT_IGNORE_DIRS, PROJECT_GIT_TIMEOUT,
            )
            defaults["max_tokens"] = PROJECT_MAX_TOKENS
            defaults["context_max_chars"] = PROJECT_CONTEXT_MAX_CHARS
            defaults["project_root"] = PROJECT_ROOT
            defaults["auto_scan"] = PROJECT_AUTO_SCAN
            defaults["max_file_size"] = PROJECT_MAX_FILE_SIZE
            defaults["scan_extensions"] = PROJECT_SCAN_EXTENSIONS
            defaults["ignore_dirs"] = PROJECT_IGNORE_DIRS
            defaults["git_timeout"] = PROJECT_GIT_TIMEOUT
        except ImportError:
            pass
        return defaults

    # ── Welcome ──────────────────────────────────────────────────────

    def _show_welcome(self):
        model = self._config["model"]
        root = self._index.root if self._index else os.getcwd()
        file_count = len(self._index.files) if self._index else 0

        print(f"""
{B}{WHITE}{BG_BLUE}{'=' * 70}{R}
{B}{WHITE}{BG_BLUE}  PROJECT MODE - Voice Project Explorer                                {R}
{B}{WHITE}{BG_BLUE}{'=' * 70}{R}

  {B}{CYAN}Model:{R} {model}
  {B}{CYAN}Root:{R}  {root}
  {B}{CYAN}Files:{R} {file_count} indexed

  {B}{YELLOW}Commands:{R}
    {GREEN}"show project"{R}       - Display project tree
    {GREEN}"find [name]"{R}        - Search for class/function/file
    {GREEN}"open [file]"{R}        - Load file into context (fuzzy match)
    {GREEN}"also open [file]"{R}   - Add another file to context
    {GREEN}"close [file]"{R}       - Remove file from context
    {GREEN}"what's loaded"{R}      - List loaded files
    {GREEN}"git status"{R}         - Show git status
    {GREEN}"git diff"{R}           - Show changes
    {GREEN}"git log"{R}            - Recent commits
    {GREEN}"rescan"{R}             - Re-index project
    {GREEN}"type"{R}               - Keyboard input mode
    {GREEN}"done" / "stop"{R}      - Exit project mode

  {DIM}Speak naturally - no wake word needed in this mode.{R}
{DIM}{'─' * 70}{R}
""")

    # ── Main Loop ────────────────────────────────────────────────────

    def _get_user_input(self) -> Optional[str]:
        """Get input from voice or keyboard based on mode."""
        if self._input_mode == "voice":
            print(f"\n{DIM}[PROJECT] Listening...{R}")
            text = whisper_speech_to_text(
                self._device,
                self._samplerate,
                extended_listen=True,
            )
            if not text:
                return None
            text = text.strip()
            if is_hallucination(text.lower(), strict=False):
                print(f"{DIM}[PROJECT] Skipped hallucination: {text[:50]}{R}")
                return None
            return text
        else:
            # Type mode
            try:
                print(f"{GREEN}project>{R} ", end="", flush=True)
                text = input().strip()
                if not text:
                    return None
                return text
            except (EOFError, KeyboardInterrupt):
                return None

    def _project_loop(self) -> ModuleResult:
        manager = get_model_manager()

        while True:
            text = self._get_user_input()
            if text is None:
                continue

            text_lower = text.lower().rstrip('.,!?')
            print(f"{CYAN}You:{R} {text}")

            # ── Exit ──
            if text_lower in ["done", "stop", "exit", "quit", "klaar",
                               "stop project", "end project"]:
                speak("Ending project session.")
                self._cleanup()
                return ModuleResult(text="Project session ended.", success=True)

            # ── Switch input mode ──
            if text_lower in ["type", "type mode", "keyboard", "typ", "typen"]:
                self._input_mode = "type"
                print(f"{YELLOW}Switched to keyboard input.{R}")
                continue
            if text_lower in ["voice", "voice mode", "speak", "spraak"]:
                self._input_mode = "voice"
                print(f"{YELLOW}Switched to voice input.{R}")
                speak("Voice mode.")
                continue

            # ── Dispatch commands ──
            handled = self._dispatch_command(text, text_lower)
            if handled:
                continue

            # ── Fallback: ask LLM ──
            prompt = self._build_prompt(text)
            print(f"{DIM}[PROJECT] Thinking...{R}")

            response = manager.ask(
                question=prompt,
                module_name="project",
                system_prompt=PROJECT_SYSTEM_PROMPT,
                model_override=self._config["model"],
                temperature=0.3,
                max_tokens=self._config["max_tokens"],
            )

            if response:
                print(f"\n{MAGENTA}Assistant:{R}")
                print(response)
                speak_text = self._make_speakable(response)
                if len(speak_text) > 300:
                    speak(speak_text[:280] + "... See full response above.")
                else:
                    speak(speak_text)
            else:
                print(f"{RED}No response from LLM.{R}")
                speak("Sorry, I couldn't process that.")

        return ModuleResult(text="Project session ended.", success=True)

    def _dispatch_command(self, text: str, text_lower: str) -> bool:
        """Dispatch voice commands. Returns True if handled."""

        # ── Show project ──
        if text_lower in ["show project", "project tree", "show tree",
                          "what files are here", "list files"]:
            self._show_project_tree()
            return True

        # ── Rescan ──
        if text_lower in ["rescan", "scan again", "scan project", "re-scan",
                          "refresh", "index again"]:
            root = self._index.root if self._index else os.getcwd()
            self._scan_project(root)
            count = len(self._index.files) if self._index else 0
            self._speak(f"Rescanned. Found {count} files.")
            return True

        # ── Find symbol/file ──
        if text_lower.startswith(("find ", "where is ", "search ", "locate ")):
            for prefix in ["find ", "where is ", "search ", "locate "]:
                if text_lower.startswith(prefix):
                    query = text[len(prefix):].strip()
                    break
            self._find_symbol(query)
            return True

        # ── Open file (fuzzy) ──
        if text_lower.startswith("open "):
            query = text[5:].strip()
            resolved = self._resolve_file(query)
            if resolved:
                self._load_file(resolved)
            else:
                self._speak(f"Couldn't find a file matching {query}. Try 'show project' to see available files.")
            return True

        # ── Also open / add file ──
        if text_lower.startswith(("also open ", "add ")):
            for prefix in ["also open ", "add "]:
                if text_lower.startswith(prefix):
                    query = text[len(prefix):].strip()
                    break
            resolved = self._resolve_file(query)
            if resolved:
                self._load_file(resolved)
            else:
                self._speak(f"Couldn't find a file matching {query}.")
            return True

        # ── Close file ──
        if text_lower in ["close all", "clear all", "unload all"]:
            self._loaded_files.clear()
            self._speak("All files closed.")
            return True

        if text_lower.startswith("close "):
            name = text[6:].strip()
            self._close_file(name)
            return True

        # ── What's loaded ──
        if text_lower in ["what's loaded", "whats loaded", "loaded files",
                          "show loaded", "list loaded", "loaded"]:
            self._show_loaded_files()
            return True

        # ── Git status ──
        if text_lower in ["git status", "what changed", "status",
                          "show status", "changes"]:
            self._git_status()
            return True

        # ── Git diff ──
        if text_lower in ["git diff", "show diff", "show changes",
                          "review changes", "diff"]:
            self._git_diff()
            return True

        # ── Git log ──
        if text_lower in ["git log", "recent commits", "show log",
                          "commit history", "log"]:
            self._git_log()
            return True

        return False

    # ── Project Scanning ─────────────────────────────────────────────

    def _scan_project(self, root: str):
        """Walk directory, build index, extract symbols."""
        root = os.path.expanduser(root)
        if not os.path.isdir(root):
            self._speak(f"Directory not found: {root}")
            return

        ignore_dirs = set(self._config["ignore_dirs"])
        extensions = set(self._config["scan_extensions"])
        max_size = self._config["max_file_size"]

        index = ProjectIndex(root=root, scanned_at=time.time())

        # Check git availability
        index.git_available = self._check_git(root)

        file_count = 0
        skipped = 0

        for dirpath, dirnames, filenames in os.walk(root):
            # Filter ignored directories in-place
            dirnames[:] = [
                d for d in dirnames
                if d not in ignore_dirs and not d.endswith(".egg-info")
            ]

            for filename in filenames:
                ext = os.path.splitext(filename)[1].lower()
                if ext not in extensions:
                    continue

                abs_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(abs_path, root)

                # Skip large files
                try:
                    size = os.path.getsize(abs_path)
                except OSError:
                    continue
                if size > max_size:
                    skipped += 1
                    continue

                # Skip binary files
                if self._is_binary(abs_path):
                    continue

                # Read and extract symbols
                classes = []
                functions = []
                lines = 0
                try:
                    with open(abs_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    lines = content.count('\n') + 1
                    classes, functions = self._extract_symbols(content, ext)
                except (OSError, UnicodeDecodeError):
                    continue

                index.files[rel_path] = FileInfo(
                    rel_path=rel_path,
                    abs_path=abs_path,
                    size_bytes=size,
                    lines=lines,
                    extension=ext,
                    classes=classes,
                    functions=functions,
                )
                file_count += 1

        self._index = index

        print(f"{GREEN}Scanned:{R} {file_count} files in {root}")
        if skipped:
            print(f"{DIM}  ({skipped} files skipped - too large){R}")
        if index.git_available:
            print(f"{DIM}  Git: available{R}")

    def _extract_symbols(self, content: str, ext: str) -> tuple:
        """Extract class and function names via regex."""
        classes = []
        functions = []

        if ext == ".py":
            classes = re.findall(r'^class\s+(\w+)', content, re.MULTILINE)
            functions = re.findall(r'^def\s+(\w+)', content, re.MULTILINE)
        elif ext in (".js", ".ts", ".jsx", ".tsx"):
            classes = re.findall(r'\bclass\s+(\w+)', content)
            functions = re.findall(
                r'(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\()',
                content,
            )
            # Flatten tuples from alternation groups
            functions = [name for group in functions for name in group if name]
        elif ext in (".go",):
            functions = re.findall(r'^func\s+(?:\([^)]*\)\s+)?(\w+)', content, re.MULTILINE)
            classes = re.findall(r'^type\s+(\w+)\s+struct', content, re.MULTILINE)
        elif ext in (".rs",):
            classes = re.findall(r'^(?:pub\s+)?struct\s+(\w+)', content, re.MULTILINE)
            functions = re.findall(r'^(?:pub\s+)?fn\s+(\w+)', content, re.MULTILINE)
        elif ext in (".java", ".cs"):
            classes = re.findall(r'\bclass\s+(\w+)', content)
            functions = re.findall(
                r'(?:public|private|protected|static|\s)+[\w<>\[\]]+\s+(\w+)\s*\(',
                content,
            )
        elif ext in (".c", ".cpp", ".h", ".hpp"):
            classes = re.findall(r'\b(?:class|struct)\s+(\w+)', content)
            functions = re.findall(r'^[\w*]+\s+(\w+)\s*\([^)]*\)\s*\{', content, re.MULTILINE)

        return classes, functions

    def _is_binary(self, filepath: str) -> bool:
        """Check if file is binary by looking for null bytes in first 8KB."""
        try:
            with open(filepath, 'rb') as f:
                chunk = f.read(8192)
            return b'\x00' in chunk
        except OSError:
            return True

    def _check_git(self, root: str) -> bool:
        """Check if git is available and root is a git repo."""
        if not shutil.which("git"):
            return False
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            return False

    # ── File Operations ──────────────────────────────────────────────

    def _resolve_file(self, query: str) -> Optional[str]:
        """Fuzzy-match query against project index.

        4-tier matching:
        1. Exact basename match
        2. Partial basename match (substring)
        3. Symbol match (class/function name)
        4. Literal path
        """
        if not self._index or not self._index.files:
            # No index - try literal path
            expanded = os.path.expanduser(query)
            if not os.path.isabs(expanded):
                expanded = os.path.join(os.getcwd(), expanded)
            if os.path.isfile(expanded):
                return expanded
            return None

        query_lower = query.lower().strip()

        # Tier 1: Exact basename match
        for rel_path, info in self._index.files.items():
            basename = os.path.basename(rel_path).lower()
            if basename == query_lower or os.path.splitext(basename)[0] == query_lower:
                return info.abs_path

        # Tier 2: Partial basename match
        matches = []
        for rel_path, info in self._index.files.items():
            basename = os.path.basename(rel_path).lower()
            if query_lower in basename:
                matches.append(info)
        if len(matches) == 1:
            return matches[0].abs_path
        if matches:
            # Pick shortest path (most specific)
            matches.sort(key=lambda f: len(f.rel_path))
            print(f"{YELLOW}Multiple matches for '{query}':{R}")
            for m in matches[:5]:
                print(f"  {m.rel_path}")
            print(f"{DIM}Opening first match: {matches[0].rel_path}{R}")
            return matches[0].abs_path

        # Tier 3: Symbol match (class/function name)
        for rel_path, info in self._index.files.items():
            all_symbols = [s.lower() for s in info.classes + info.functions]
            if query_lower in all_symbols:
                return info.abs_path

        # Tier 4: Partial path match
        for rel_path, info in self._index.files.items():
            if query_lower in rel_path.lower():
                return info.abs_path

        # Tier 5: Literal filesystem path
        expanded = os.path.expanduser(query)
        if not os.path.isabs(expanded):
            expanded = os.path.join(
                self._index.root if self._index else os.getcwd(),
                expanded,
            )
        if os.path.isfile(expanded):
            return expanded

        return None

    def _load_file(self, filepath: str):
        """Load a file into context."""
        filepath = os.path.expanduser(filepath)

        if not os.path.exists(filepath):
            self._speak(f"File not found: {filepath}")
            return

        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except OSError as e:
            self._speak(f"Error reading file: {e}")
            return

        # Warn about large files
        if len(content) > self._config["max_file_size"]:
            self._speak("Warning: this file is very large. Loading it may use a lot of context.")

        self._loaded_files[filepath] = content
        lines = content.count('\n') + 1
        filename = os.path.basename(filepath)

        # Show preview
        print(f"\n{GREEN}Loaded: {filename}{R} ({lines} lines)")
        preview = content.split('\n')[:15]
        for i, line in enumerate(preview, 1):
            print(f"{DIM}{i:4}{R} {line[:100]}")
        if lines > 15:
            print(f"{DIM}  ... ({lines - 15} more lines){R}")

        self._speak(f"Loaded {filename}. {lines} lines.")

    def _close_file(self, name: str):
        """Remove a file from loaded context by partial name match."""
        name_lower = name.lower().strip()

        for path in list(self._loaded_files.keys()):
            basename = os.path.basename(path).lower()
            if name_lower in basename or name_lower in path.lower():
                del self._loaded_files[path]
                self._speak(f"Closed {os.path.basename(path)}.")
                return

        self._speak(f"No loaded file matching '{name}'.")

    def _show_loaded_files(self):
        """List currently loaded files."""
        if not self._loaded_files:
            self._speak("No files loaded.")
            print(f"{DIM}No files loaded. Say 'open [file]' to load one.{R}")
            return

        print(f"\n{B}{CYAN}Loaded files:{R}")
        total_lines = 0
        for path, content in self._loaded_files.items():
            lines = content.count('\n') + 1
            total_lines += lines
            filename = os.path.basename(path)
            rel = os.path.relpath(path, self._index.root) if self._index else path
            print(f"  {GREEN}{filename}{R}  ({lines} lines)  {DIM}{rel}{R}")

        self._speak(f"{len(self._loaded_files)} files loaded, {total_lines} total lines.")

    # ── Project Display ──────────────────────────────────────────────

    def _show_project_tree(self):
        """Display ASCII tree of indexed files."""
        if not self._index or not self._index.files:
            self._speak("No project indexed. Say 'scan project' first.")
            return

        root = self._index.root
        root_name = os.path.basename(root) or root

        # Group files by directory
        tree: Dict[str, list] = {}
        for rel_path in sorted(self._index.files.keys()):
            dirname = os.path.dirname(rel_path) or "."
            if dirname not in tree:
                tree[dirname] = []
            tree[dirname].append(os.path.basename(rel_path))

        # Print tree
        print(f"\n{B}{CYAN}{root_name}/{R}")
        sorted_dirs = sorted(tree.keys())
        for i, dirname in enumerate(sorted_dirs):
            is_last_dir = (i == len(sorted_dirs) - 1)
            prefix = "└── " if is_last_dir else "├── "
            child_prefix = "    " if is_last_dir else "│   "

            if dirname == ".":
                # Root-level files
                for j, fname in enumerate(tree[dirname]):
                    fprefix = "└── " if (j == len(tree[dirname]) - 1 and is_last_dir) else "├── "
                    info = self._index.files.get(fname)
                    extra = ""
                    if info and info.classes:
                        extra = f"  {DIM}[{', '.join(info.classes[:3])}]{R}"
                    print(f"  {fprefix}{fname}{extra}")
            else:
                print(f"  {prefix}{BLUE}{dirname}/{R}")
                files_in_dir = tree[dirname]
                for j, fname in enumerate(files_in_dir):
                    fprefix = "└── " if j == len(files_in_dir) - 1 else "├── "
                    rel = os.path.join(dirname, fname)
                    info = self._index.files.get(rel)
                    extra = ""
                    if info and info.classes:
                        extra = f"  {DIM}[{', '.join(info.classes[:3])}]{R}"
                    print(f"  {child_prefix}{fprefix}{fname}{extra}")

        # Summary
        total_files = len(self._index.files)
        extensions = {}
        total_lines = 0
        for info in self._index.files.values():
            ext = info.extension
            extensions[ext] = extensions.get(ext, 0) + 1
            total_lines += info.lines

        ext_summary = ", ".join(
            f"{count} {ext}" for ext, count in
            sorted(extensions.items(), key=lambda x: -x[1])[:5]
        )
        print(f"\n{DIM}{total_files} files, {total_lines} lines ({ext_summary}){R}")

        self._speak(f"Project has {total_files} files and {total_lines} lines. {ext_summary}.")

    def _find_symbol(self, query: str):
        """Search for a class, function, or file name across the index."""
        if not self._index or not self._index.files:
            self._speak("No project indexed. Say 'scan project' first.")
            return

        query_lower = query.lower().strip()
        results = []

        for rel_path, info in self._index.files.items():
            # Check filename
            basename = os.path.basename(rel_path).lower()
            if query_lower in basename:
                results.append(("file", rel_path, basename))

            # Check classes
            for cls in info.classes:
                if query_lower in cls.lower():
                    results.append(("class", rel_path, cls))

            # Check functions
            for func in info.functions:
                if query_lower in func.lower():
                    results.append(("function", rel_path, func))

        if not results:
            self._speak(f"Nothing found matching '{query}'.")
            print(f"{YELLOW}No matches for '{query}'{R}")
            return

        print(f"\n{B}{CYAN}Found {len(results)} match(es) for '{query}':{R}")
        for kind, path, name in results[:15]:
            icon = {"file": "F", "class": "C", "function": "f"}[kind]
            color = {"file": WHITE, "class": YELLOW, "function": GREEN}[kind]
            print(f"  {DIM}[{icon}]{R} {color}{name}{R}  {DIM}in {path}{R}")

        if len(results) > 15:
            print(f"{DIM}  ... and {len(results) - 15} more{R}")

        self._speak(f"Found {len(results)} matches for {query}.")

    # ── Git Integration ──────────────────────────────────────────────

    def _run_git(self, args: List[str]) -> Optional[str]:
        """Run a git command and return stdout."""
        if not self._index or not self._index.git_available:
            self._speak("Git is not available in this project.")
            return None

        try:
            result = subprocess.run(
                ["git"] + args,
                cwd=self._index.root,
                capture_output=True,
                text=True,
                timeout=self._config["git_timeout"],
            )
            if result.returncode != 0:
                print(f"{RED}Git error: {result.stderr.strip()}{R}")
                return None
            return result.stdout
        except subprocess.TimeoutExpired:
            self._speak("Git command timed out.")
            return None
        except OSError as e:
            print(f"{RED}Git error: {e}{R}")
            return None

    def _git_status(self):
        """Show git status summary."""
        output = self._run_git(["status", "--porcelain"])
        if output is None:
            return

        if not output.strip():
            self._speak("Working tree is clean. No changes.")
            print(f"{GREEN}Clean - no changes{R}")
            return

        modified = []
        added = []
        deleted = []
        untracked = []

        for line in output.strip().split('\n'):
            if len(line) < 3:
                continue
            status = line[:2]
            filepath = line[3:]

            if 'M' in status:
                modified.append(filepath)
            elif 'A' in status:
                added.append(filepath)
            elif 'D' in status:
                deleted.append(filepath)
            elif '?' in status:
                untracked.append(filepath)

        print(f"\n{B}{CYAN}Git Status:{R}")
        if modified:
            print(f"  {YELLOW}Modified ({len(modified)}):{R}")
            for f in modified:
                print(f"    {YELLOW}M{R} {f}")
        if added:
            print(f"  {GREEN}Added ({len(added)}):{R}")
            for f in added:
                print(f"    {GREEN}A{R} {f}")
        if deleted:
            print(f"  {RED}Deleted ({len(deleted)}):{R}")
            for f in deleted:
                print(f"    {RED}D{R} {f}")
        if untracked:
            print(f"  {DIM}Untracked ({len(untracked)}):{R}")
            for f in untracked[:10]:
                print(f"    {DIM}?{R} {f}")
            if len(untracked) > 10:
                print(f"    {DIM}... and {len(untracked) - 10} more{R}")

        parts = []
        if modified:
            parts.append(f"{len(modified)} modified")
        if added:
            parts.append(f"{len(added)} added")
        if deleted:
            parts.append(f"{len(deleted)} deleted")
        if untracked:
            parts.append(f"{len(untracked)} untracked")

        self._speak(". ".join(parts) + ".")

    def _git_diff(self, filepath: str = None):
        """Show git diff."""
        args = ["diff", "--stat"]
        if filepath:
            args.append(filepath)

        output = self._run_git(args)
        if output is None:
            return

        if not output.strip():
            self._speak("No differences found.")
            return

        print(f"\n{B}{CYAN}Git Diff:{R}")
        print(output)

        # Count changes
        lines = output.strip().split('\n')
        if lines:
            summary_line = lines[-1]
            self._speak(f"Changes: {summary_line.strip()}")

    def _git_log(self, count: int = 10):
        """Show recent git commits."""
        output = self._run_git(["log", f"--oneline", f"-{count}"])
        if output is None:
            return

        if not output.strip():
            self._speak("No commits found.")
            return

        print(f"\n{B}{CYAN}Recent Commits:{R}")
        for line in output.strip().split('\n'):
            parts = line.split(' ', 1)
            if len(parts) == 2:
                hash_str, msg = parts
                print(f"  {YELLOW}{hash_str}{R} {msg}")
            else:
                print(f"  {line}")

        commit_count = len(output.strip().split('\n'))
        self._speak(f"Showing {commit_count} recent commits.")

    # ── LLM Context ──────────────────────────────────────────────────

    def _build_prompt(self, user_input: str) -> str:
        """Build prompt with project context and loaded files."""
        parts = []
        budget = self._config["context_max_chars"]

        # Project summary
        summary = self._project_summary()
        if summary:
            parts.append(summary)
            budget -= len(summary)

        # Loaded files (newest first, truncate oldest if over budget)
        file_parts = []
        for path, content in self._loaded_files.items():
            filename = os.path.basename(path)
            ext = os.path.splitext(path)[1].lstrip('.')
            rel = path
            if self._index:
                rel = os.path.relpath(path, self._index.root)

            file_block = f"[FILE: {rel}]\n```{ext}\n{content}\n```"

            if len(file_block) > budget:
                # Truncate this file's content
                available = budget - 200  # Reserve space for markers
                if available > 500:
                    truncated = content[:available]
                    file_block = f"[FILE: {rel}]\n```{ext}\n{truncated}\n... (truncated)\n```"
                else:
                    file_block = f"[FILE: {rel}] (too large to include, {len(content)} chars)"

            file_parts.append(file_block)
            budget -= len(file_block)

            if budget < 200:
                file_parts.append(f"[... {len(self._loaded_files) - len(file_parts)} more files omitted]")
                break

        parts.extend(file_parts)
        parts.append(f"\nUser request: {user_input}")

        return "\n\n".join(parts)

    def _project_summary(self) -> str:
        """Generate a project summary string for LLM context."""
        if not self._index or not self._index.files:
            return ""

        total_files = len(self._index.files)
        extensions = {}
        total_lines = 0
        all_classes = []
        all_functions = []

        for info in self._index.files.values():
            extensions[info.extension] = extensions.get(info.extension, 0) + 1
            total_lines += info.lines
            all_classes.extend(info.classes)
            all_functions.extend(info.functions)

        ext_summary = ", ".join(
            f"{count}{ext}" for ext, count in
            sorted(extensions.items(), key=lambda x: -x[1])[:5]
        )

        summary = (
            f"[PROJECT: {os.path.basename(self._index.root)}]\n"
            f"Root: {self._index.root}\n"
            f"Files: {total_files} ({ext_summary})\n"
            f"Lines: {total_lines}\n"
            f"Classes: {len(all_classes)} | Functions: {len(all_functions)}"
        )

        if self._index.git_available:
            summary += "\nGit: available"

        return summary

    # ── Utilities ────────────────────────────────────────────────────

    def _make_speakable(self, text: str) -> str:
        """Clean text for TTS - remove code blocks and markdown."""
        text = re.sub(r'```[\s\S]*?```', '[code block]', text)
        text = re.sub(r'`[^`]+`', '', text)
        text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'\n\s*\n', '. ', text)
        text = re.sub(r'\n', ' ', text)
        return text.strip()

    def _speak(self, text: str):
        """Speak text and print it."""
        print(f"  {DIM}{text}{R}")
        speak(text)

    def _cleanup(self):
        """Cleanup when exiting project mode."""
        self._loaded_files.clear()
        self._index = None
        get_model_manager().clear_history("project")

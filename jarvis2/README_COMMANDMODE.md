# Command Mode & Project Mode Documentation

## Changelog
- **2026-02-08 v1.0** - Initial documentation: Project Module created

---

## Modules Overview

| Module | Trigger | Priority | Mode |
|--------|---------|----------|------|
| Chat | (fallback) | 10 | Single-turn |
| Calendar | "add to calendar", "check agenda" | 70 | Multi-turn |
| Terminal | "run command", "terminal" | 75 | Confirmation |
| Dictation | "dictate", "dicteer" | 80 | Continuous |
| Coding | "join me", "code with me" | 85 | Continuous |
| **Project** | **"project mode", "scan project"** | **86** | **Continuous** |

---

## Project Module

### Entry Triggers
- "project mode"
- "scan project"
- "explore project"
- "project explorer"
- "open project"
- "show project"

### Voice Commands (inside project mode)

#### Project Navigation
| Command | Description |
|---------|-------------|
| "show project" / "project tree" | Display ASCII file tree with class annotations |
| "find [name]" | Search for class, function, or file by name |
| "rescan" / "scan again" | Re-index the project directory |

#### File Management
| Command | Description |
|---------|-------------|
| "open [name]" | Fuzzy-match and load file into LLM context |
| "also open [name]" / "add [name]" | Load additional file (keeps existing) |
| "close [name]" | Remove specific file from context |
| "close all" | Remove all files from context |
| "what's loaded" | List all loaded files with line counts |

#### Git Integration
| Command | Description |
|---------|-------------|
| "git status" / "what changed" | Show modified/added/deleted/untracked files |
| "git diff" / "show changes" | Show diff stats |
| "git log" / "recent commits" | Show last 10 commits |

#### LLM Interaction
| Command | Description |
|---------|-------------|
| (any question) | Sent to LLM with project context + loaded files |

#### Control
| Command | Description |
|---------|-------------|
| "type" | Switch to keyboard input mode |
| "done" / "stop" / "exit" | Exit project mode |

### Fuzzy File Resolution
When you say "open [name]", the module searches in this order:
1. **Exact basename** - "module.py" matches `modules/project/module.py`
2. **Partial basename** - "project" matches `modules/project/module.py`
3. **Symbol match** - "ProjectModule" matches the file containing that class
4. **Partial path** - "project/module" matches `modules/project/module.py`
5. **Literal filesystem path** - Falls back to exact path if nothing else matches

### Configuration (`config.py`)
```python
PROJECT_MAX_TOKENS = 4000          # LLM response token limit
PROJECT_CONTEXT_MAX_CHARS = 32000  # Max prompt context size
PROJECT_ROOT = None                # None = CWD, or absolute path
PROJECT_AUTO_SCAN = True           # Auto-scan on entering project mode
PROJECT_MAX_FILE_SIZE = 100_000    # Skip files > 100KB in scan
PROJECT_SCAN_EXTENSIONS = [...]    # File extensions to index
PROJECT_IGNORE_DIRS = [...]        # Directories to skip
PROJECT_GIT_TIMEOUT = 10           # Git command timeout (seconds)
```

### Architecture
```
jarvis2/modules/project/
├── __init__.py     # Package init, exports ProjectModule
└── module.py       # ProjectModule class
    ├── FileInfo        # Dataclass: indexed file metadata
    ├── ProjectIndex    # Dataclass: project-wide index
    └── ProjectModule   # BaseModule subclass
        ├── execute()           # Entry point → scan + loop
        ├── _project_loop()     # Continuous listen-dispatch
        ├── _dispatch_command() # Voice command router
        ├── _scan_project()     # os.walk + symbol extraction
        ├── _extract_symbols()  # Regex class/function extraction
        ├── _resolve_file()     # 5-tier fuzzy file matching
        ├── _load_file()        # Read file into context
        ├── _show_project_tree()# ASCII tree display
        ├── _find_symbol()      # Cross-project symbol search
        ├── _git_status()       # Git status integration
        ├── _git_diff()         # Git diff integration
        ├── _git_log()          # Git log integration
        └── _build_prompt()     # LLM context assembly
```

### Troubleshooting

| Issue | Solution |
|-------|----------|
| "Git not available" | Install git: `sudo apt install git` |
| No files found after scan | Check `PROJECT_SCAN_EXTENSIONS` includes your file types |
| Large project slow to scan | Increase `PROJECT_MAX_FILE_SIZE` or add dirs to `PROJECT_IGNORE_DIRS` |
| Fuzzy match finds wrong file | Use more specific name or say "show project" to see available files |
| LLM response truncated | Reduce number of loaded files or increase `PROJECT_MAX_TOKENS` |

---

## Coding Module (existing)

### Entry Triggers
- "join me", "code with me", "help me code"
- "pair program", "coding mode", "let's code"

### Voice Commands
| Command | Description |
|---------|-------------|
| "open [file]" | Load file (exact path) |
| "explain this" | Explain loaded code |
| "write [desc]" | Generate code |
| "fix this" | Suggest fixes |
| "apply" / "do it" | Apply pending changes |
| "show diff" | Preview changes |
| "cancel" | Cancel pending changes |
| "type" | Keyboard input |
| "done" / "stop" | Exit coding mode |

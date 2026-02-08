import json
import os
from colors import learn

CORRECTIONS_FILE = os.path.expanduser("~/.assistmint_corrections.json")

def load_corrections():
    """Load corrections dictionary from file."""
    if os.path.exists(CORRECTIONS_FILE):
        try:
            with open(CORRECTIONS_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

def save_corrections(corrections):
    """Save corrections dictionary to file."""
    with open(CORRECTIONS_FILE, 'w') as f:
        json.dump(corrections, f, indent=2)

def apply_corrections(text):
    """Apply stored corrections to transcribed text."""
    corrections = load_corrections()
    original = text.lower()

    for wrong, right in corrections.items():
        if wrong.lower() in original:
            text = text.lower().replace(wrong.lower(), right)
            print(learn(f"Auto-corrected: '{wrong}' → '{right}'"))

    return text

def add_correction(wrong, right):
    """Add a new correction to the dictionary."""
    corrections = load_corrections()
    corrections[wrong.lower()] = right
    save_corrections(corrections)
    print(learn(f"Saved: '{wrong}' → '{right}'"))

def list_corrections():
    """List all stored corrections."""
    corrections = load_corrections()
    if corrections:
        print("\n=== Stored Corrections ===")
        for wrong, right in corrections.items():
            print(f"  '{wrong}' → '{right}'")
        print()
    else:
        print(learn("No corrections stored yet."))
    return corrections

def remove_correction(wrong):
    """Remove a correction from the dictionary."""
    corrections = load_corrections()
    if wrong.lower() in corrections:
        del corrections[wrong.lower()]
        save_corrections(corrections)
        print(learn(f"Removed correction for '{wrong}'"))
        return True
    return False

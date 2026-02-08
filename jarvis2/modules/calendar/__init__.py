"""
Calendar Module - Calendar management for Assistmint.

Supports:
- Evolution (GNOME) - syncs with Google Calendar
- Google Calendar (via gcalcli)
- Local .reminders file
"""

from modules.calendar.module import CalendarModule

__all__ = ["CalendarModule"]

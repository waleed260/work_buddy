"""
Tools for Remote Work Buddy AI Agent.
Provides calendar, email, task management, and wellness integrations.
Compatible with OpenAI Agents SDK.
"""

from datetime import datetime, timedelta
from typing import Optional
from pydantic import BaseModel
from agents.tool import function_tool


class CalendarEvent(BaseModel):
    """Represents a calendar event."""
    title: str
    start_time: str
    end_time: str
    description: str = ""
    attendees: list[str] = []


class Task(BaseModel):
    """Represents a task."""
    title: str
    priority: str = "medium"
    due_date: Optional[str] = None
    completed: bool = False


class EmailDraft(BaseModel):
    """Represents an email draft."""
    to: str
    subject: str
    body: str


class SlackMessage(BaseModel):
    """Represents a Slack message draft."""
    channel: str
    message: str


# Shared state for tools
_calendar_events: list[CalendarEvent] = []
_email_drafts: list[EmailDraft] = []
_slack_messages: list[SlackMessage] = []
_tasks: list[Task] = []
_breaks: list[dict] = []
_habits: dict[str, list[str]] = {}
_transcripts: list[dict] = []


# ============ Calendar Tools ============

def _check_calendar_events(date: str) -> list[str]:
    """Check events for a specific date. Returns list of event summaries."""
    global _calendar_events
    events = [e for e in _calendar_events if e.start_time.startswith(date)]
    return [f"{e.title}: {e.start_time} - {e.end_time}" for e in events]


def _add_calendar_event(title: str, start_time: str, end_time: str, description: str = "") -> str:
    """Add a new event to calendar. Returns confirmation."""
    global _calendar_events
    event = CalendarEvent(
        title=title,
        start_time=start_time,
        end_time=end_time,
        description=description
    )
    _calendar_events.append(event)
    return f"✅ Event '{title}' scheduled from {start_time} to {end_time}"


def _get_calendar_free_slots(date: str, duration_minutes: int = 60) -> list[str]:
    """Get available time slots for a given date."""
    global _calendar_events
    work_start = 9  # 9 AM
    work_end = 20   # 8 PM (guardrail: no tasks after 8 PM)
    
    slots = []
    current_hour = work_start
    while current_hour < work_end:
        slot = f"{date}T{current_hour:02d}:00:00"
        if not any(e.start_time == slot for e in _calendar_events):
            slots.append(f"{current_hour:02d}:00")
        current_hour += 1
    return slots


def _get_todays_schedule() -> str:
    """Get today's schedule as a formatted string."""
    global _calendar_events
    today = datetime.now().strftime("%Y-%m-%d")
    events = [e for e in _calendar_events if e.start_time.startswith(today)]
    
    if not events:
        return "📅 Your schedule is clear today!"
    
    schedule = "📅 **Today's Schedule**\n\n"
    schedule += "| Time | Event |\n"
    schedule += "|------|-------|\n"
    
    for event in events:
        start = event.start_time.split('T')[-1][:5]
        end = event.end_time.split('T')[-1][:5]
        schedule += f"| {start} - {end} | {event.title} |\n"
    
    return schedule


# Export both raw functions and function_tool wrapped versions
check_calendar_events = function_tool(_check_calendar_events)
add_calendar_event = function_tool(_add_calendar_event)
get_calendar_free_slots = function_tool(_get_calendar_free_slots)
get_todays_schedule = function_tool(_get_todays_schedule)


# ============ Task Management Tools ============

def _add_task(title: str, priority: str = "medium", due_date: Optional[str] = None) -> str:
    """Add a new task. Returns confirmation."""
    global _tasks
    task = Task(title=title, priority=priority, due_date=due_date)
    _tasks.append(task)
    return f"✅ Task added: '{title}' (Priority: {priority})"


def _get_tasks(completed: Optional[bool] = None) -> str:
    """Get tasks as a formatted string."""
    global _tasks
    if completed is None:
        filtered = _tasks
    else:
        filtered = [t for t in _tasks if t.completed == completed]
    
    if not filtered:
        return "📋 No tasks found."
    
    result = "📋 **Tasks**\n\n"
    for task in filtered:
        status = "✅" if task.completed else "🔄"
        result += f"{status} {task.title} ({task.priority})\n"
    
    return result


def _complete_task(title: str) -> str:
    """Mark a task as completed."""
    global _tasks
    for task in _tasks:
        if task.title == title:
            task.completed = True
            return f"✅ Marked '{title}' as completed"
    return f"Task '{title}' not found"


def _get_daily_standup() -> str:
    """Generate daily standup summary."""
    global _tasks
    completed = [t for t in _tasks if t.completed]
    pending = [t for t in _tasks if not t.completed]
    
    standup = "📋 **Daily Standup**\n\n"
    standup += "**Completed:**\n"
    for t in completed[-5:]:
        standup += f"  ✅ {t.title}\n"
    if not completed:
        standup += "  (none yet)\n"
    standup += "\n**In Progress:**\n"
    for t in pending[:5]:
        standup += f"  🔄 {t.title} ({t.priority})\n"
    if not pending:
        standup += "  (none)\n"
    return standup


# Export both raw functions and function_tool wrapped versions
add_task = function_tool(_add_task)
get_tasks = function_tool(_get_tasks)
complete_task = function_tool(_complete_task)
get_daily_standup = function_tool(_get_daily_standup)


# ============ Email Tools ============

def _draft_email(to: str, subject: str, body: str) -> str:
    """Draft an email. Returns confirmation."""
    global _email_drafts
    draft = EmailDraft(to=to, subject=subject, body=body)
    _email_drafts.append(draft)
    return f"✅ Email drafted to {to}: '{subject}'"


def _get_email_drafts() -> str:
    """Get all email drafts."""
    global _email_drafts
    if not _email_drafts:
        return "📧 No email drafts."
    
    result = "📧 **Email Drafts**\n\n"
    for draft in _email_drafts:
        result += f"• To: {draft.to}\n  Subject: {draft.subject}\n\n"
    return result


# Export both raw functions and function_tool wrapped versions
draft_email = function_tool(_draft_email)
get_email_drafts = function_tool(_get_email_drafts)


# ============ Slack Tools ============

def _draft_slack_message(channel: str, message: str) -> str:
    """Draft a Slack message. Returns confirmation."""
    global _slack_messages
    msg = SlackMessage(channel=channel, message=message)
    _slack_messages.append(msg)
    return f"✅ Slack message drafted for #{channel}"


def _get_slack_messages() -> str:
    """Get all drafted Slack messages."""
    global _slack_messages
    if not _slack_messages:
        return "💬 No Slack messages drafted."
    
    result = "💬 **Slack Messages**\n\n"
    for msg in _slack_messages:
        result += f"• #{msg.channel}: {msg.message[:50]}...\n"
    return result


# Export both raw functions and function_tool wrapped versions
draft_slack_message = function_tool(_draft_slack_message)
get_slack_messages = function_tool(_get_slack_messages)


# ============ Wellness Tools ============

def _suggest_break(last_break_minutes_ago: int) -> str:
    """Suggest a break if needed (every 60-90 minutes)."""
    if last_break_minutes_ago >= 60:
        return "🧘 Time for a break! Consider a 5-15 minute walk or stretch."
    return "✅ You're good for now. Take a break when you reach 60+ minutes of work."


def _log_break(break_type: str, duration_minutes: int) -> str:
    """Log a break taken."""
    global _breaks
    entry = {
        "type": break_type,
        "duration": duration_minutes,
        "timestamp": datetime.now().isoformat()
    }
    _breaks.append(entry)
    return f"✅ Logged {break_type} break for {duration_minutes} minutes"


def _get_weekly_insights() -> str:
    """Generate weekly wellness insights."""
    global _breaks
    total_breaks = len(_breaks)
    avg_break_duration = sum(b["duration"] for b in _breaks) / max(total_breaks, 1)
    
    insights = "📊 **Weekly Wellness Insights**\n\n"
    insights += f"• Breaks taken: {total_breaks}\n"
    insights += f"• Average break duration: {avg_break_duration:.1f} minutes\n"
    
    if total_breaks < 10:
        insights += "💡 Tip: Try to take more frequent breaks for better productivity.\n"
    else:
        insights += "✅ Great job maintaining regular breaks!\n"
    
    return insights


def _track_habit(habit_name: str, status: str) -> str:
    """Track a habit completion."""
    global _habits
    if habit_name not in _habits:
        _habits[habit_name] = []
    _habits[habit_name].append(status)
    return f"✅ Tracked habit '{habit_name}': {status}"


# Export both raw functions and function_tool wrapped versions
suggest_break = function_tool(_suggest_break)
log_break = function_tool(_log_break)
get_weekly_insights = function_tool(_get_weekly_insights)
track_habit = function_tool(_track_habit)


# ============ Transcription Tools ============

def _transcribe_meeting(meeting_id: str) -> str:
    """Mock transcription of a meeting."""
    global _transcripts
    transcript = f"""
[Meeting Transcript - {meeting_id}]

Participant 1: Let's discuss the project timeline.
Participant 2: The deadline is next Friday. We need to prioritize features.
Participant 1: Agreed. Let's focus on the core functionality first.
Participant 3: I'll prepare the technical specs by tomorrow.
Participant 2: Great. Let's schedule a follow-up for Wednesday.

[End of meeting]
    """.strip()
    
    _transcripts.append({
        "meeting_id": meeting_id,
        "transcript": transcript,
        "timestamp": datetime.now().isoformat()
    })
    
    return transcript


def _extract_action_items(transcript: str) -> str:
    """Extract action items from a transcript."""
    action_items = []
    lines = transcript.split("\n")
    for line in lines:
        if "I'll" in line or "will" in line.lower():
            action_items.append(line.strip())
    
    if not action_items:
        return "No action items found."
    
    result = "✅ **Action Items**\n\n"
    for item in action_items:
        result += f"• {item}\n"
    return result


# Export both raw functions and function_tool wrapped versions
transcribe_meeting = function_tool(_transcribe_meeting)
extract_action_items = function_tool(_extract_action_items)


# ============ Guardrails Tools ============

def _validate_time_slot(hour: int) -> str:
    """Validate if a time slot is within acceptable work hours (9 AM - 8 PM PKT)."""
    if hour >= 20:
        return "❌ Invalid: Task scheduled after 20:00. Work-life balance: no tasks after 8 PM PKT."
    if hour < 6:
        return "❌ Invalid: Task scheduled before 6:00 AM. Consider respecting rest hours."
    return f"✅ Valid: {hour:02d}:00 is within work hours (9 AM - 8 PM PKT)"


def _get_current_time_pkt() -> str:
    """Get current time in PKT timezone."""
    from datetime import timezone, timedelta
    pkt = timezone(timedelta(hours=5))
    current = datetime.now(pkt)
    return current.strftime("%Y-%m-%d %H:%M:%S PKT")


def _is_within_work_hours() -> str:
    """Check if current time is within work hours (9 AM - 8 PM PKT)."""
    from datetime import timezone, timedelta
    pkt = timezone(timedelta(hours=5))
    current = datetime.now(pkt)
    if 9 <= current.hour < 20:
        return "✅ Yes, currently within work hours."
    return "❌ No, currently outside work hours (9 AM - 8 PM PKT)."


def _suggest_optimal_focus_time() -> str:
    """Suggest optimal focus session time."""
    from datetime import timezone, timedelta
    pkt = timezone(timedelta(hours=5))
    current = datetime.now(pkt)
    
    if current.hour < 11:
        return "📍 Optimal: 09:00-11:00 (Morning focus block)"
    elif current.hour < 16:
        return "📍 Optimal: 14:00-16:00 (Afternoon focus block)"
    else:
        return "📍 Optimal: 09:00-11:00 tomorrow (Morning focus block)"


# Export both raw functions and function_tool wrapped versions
validate_time_slot = function_tool(_validate_time_slot)
get_current_time_pkt = function_tool(_get_current_time_pkt)
is_within_work_hours = function_tool(_is_within_work_hours)
suggest_optimal_focus_time = function_tool(_suggest_optimal_focus_time)


# Export raw functions for direct testing (access via .raw_function attribute)
# For direct calls in tests, use the underscore-prefixed internal functions

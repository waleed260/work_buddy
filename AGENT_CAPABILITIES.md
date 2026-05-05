# Remote Work Buddy - Agent Capabilities & Architecture

## Overview

Remote Work Buddy is a proactive AI agent built with the **OpenAI Agents SDK** designed to boost remote worker productivity and well-being. It acts as a personal executive assistant, handling daily workflows autonomously while adapting to user preferences and enforcing work-life balance.

**Model**: Qwen3-Coder (via OpenRouter API)
**Framework**: OpenAI Agents SDK
**Timezone**: PKT (Asia/Karachi, UTC+5)
**Work Hours**: 9:00 AM - 8:00 PM (no tasks after 8 PM)

---

## Core Capabilities

### 1. Calendar & Meeting Management

**Features:**
- Schedule meetings within work hours (9 AM - 8 PM PKT)
- Check calendar events for specific dates
- Find free time slots for scheduling
- Display today's schedule in formatted tables
- Transcribe meetings (mock implementation - Otter/Fireflies style)
- Extract action items from meeting transcripts
- Validate meeting duration (maximum 2 hours)
- Suggest optimal meeting times based on availability

**Tools:**
- `check_calendar_events(date)` - Get events for a specific date
- `add_calendar_event(title, start_time, end_time, description)` - Schedule new event
- `get_calendar_free_slots(date, duration_minutes)` - Find available time slots
- `get_todays_schedule()` - Display today's schedule as formatted table
- `transcribe_meeting(meeting_id)` - Mock transcription of meeting
- `extract_action_items(transcript)` - Extract action items from transcript

**Example Use Cases:**
- "Schedule my team standup for tomorrow at 10 AM"
- "Find a 1-hour slot for a meeting next Tuesday"
- "What's on my calendar today?"
- "Transcribe the meeting and extract action items"

---

### 2. Task Management

**Features:**
- Add tasks with priority levels (high/medium/low)
- View all tasks or filter by completion status
- Mark tasks as completed
- Generate daily standup summaries
- Prioritize tasks throughout the day
- Schedule tasks within work hours
- Track task completion progress

**Tools:**
- `add_task(title, priority, due_date)` - Add new task
- `get_tasks(completed)` - View tasks (optionally filtered)
- `complete_task(title)` - Mark task as completed
- `get_daily_standup()` - Generate standup summary

**Example Use Cases:**
- "Add 'Complete project documentation' as a high priority task"
- "Show me my pending tasks"
- "Mark 'Review pull requests' as completed"
- "Generate today's standup"

---

### 3. Communication Tools

**Email Management:**
- Draft professional emails with subject lines
- Store and retrieve email drafts
- Adapt tone based on recipient (formal/casual)
- Proofread for clarity and tone

**Slack Integration:**
- Create Slack messages for team channels
- Format messages appropriately for channels
- Create daily standup updates
- Maintain professional yet friendly tone

**Tools:**
- `draft_email(to, subject, body)` - Draft professional email
- `get_email_drafts()` - View all email drafts
- `draft_slack_message(channel, message)` - Draft Slack message
- `get_slack_messages()` - View all Slack message drafts

**Example Use Cases:**
- "Draft an email to the team about the project update"
- "Create a Slack message for #general with today's standup"
- "Show me my email drafts"

---

### 4. Wellness & Work-Life Balance

**Features:**
- Suggest breaks every 60-90 minutes
- Log break activities (walk, stretch, hydration, meals)
- Track wellness habits
- Generate weekly wellness insights
- Monitor work-life balance
- Enforce no-work-after-8-PM rule (PKT timezone)
- Encourage healthy remote work practices

**Tools:**
- `suggest_break(last_break_minutes_ago)` - Suggest break if needed
- `log_break(break_type, duration_minutes)` - Log break taken
- `track_habit(habit_name, status)` - Track wellness habit
- `get_weekly_insights()` - Generate wellness insights

**Example Use Cases:**
- "I've been working for 90 minutes, suggest a break"
- "Log a 10-minute walk break"
- "Track my hydration habit"
- "Show me my weekly wellness insights"

---

### 5. Guardrails & Safety

**Features:**
- Validate time slots against work hours (9 AM - 8 PM PKT)
- Check current time in PKT timezone
- Suggest optimal focus times
- Prevent scheduling outside work hours
- Protect privacy (detects sensitive data patterns)
- Enforce meeting duration limits
- Validate break frequency

**Tools:**
- `validate_time_slot(hour)` - Check if time is within work hours
- `get_current_time_pkt()` - Get current time in PKT
- `is_within_work_hours()` - Check if currently within work hours
- `suggest_optimal_focus_time()` - Suggest best focus session time

**Guardrail Rules:**
- ❌ No tasks after 8:00 PM PKT
- ❌ No tasks before 6:00 AM
- ❌ Meetings cannot exceed 2 hours
- ❌ Breaks required every 60-90 minutes
- ❌ No sensitive data logging (passwords, API keys, tokens, etc.)

**Example Use Cases:**
- "Is 9 PM a valid time for a task?" → ❌ Invalid
- "Is 2 PM a valid time for a task?" → ✅ Valid
- "What's the current time in PKT?"
- "Suggest an optimal focus time"

---

## Specialized Sub-Agents

### MeetingAgent
**Specialization**: Meeting coordination and transcription

**Responsibilities:**
1. Schedule meetings within work hours
2. Transcribe meetings and extract key points
3. Identify action items from meeting transcripts
4. Ensure meetings don't exceed 2 hours
5. Suggest optimal meeting times

**Available Tools:**
- Calendar management tools
- Meeting transcription tools
- Action item extraction
- Time validation

---

### WellnessAgent
**Specialization**: User well-being and work-life balance

**Responsibilities:**
1. Suggest breaks every 60-90 minutes
2. Track wellness habits
3. Provide weekly wellness insights
4. Remind users about work hours
5. Encourage healthy remote work practices

**Available Tools:**
- Break suggestion and logging
- Habit tracking
- Weekly insights generation
- Work hours monitoring

---

### TaskAgent
**Specialization**: Task management and productivity

**Responsibilities:**
1. Add and organize tasks with priorities
2. Generate daily standup summaries
3. Schedule tasks within work hours
4. Prioritize tasks using Eisenhower Matrix principles
5. Track task completion

**Available Tools:**
- Task management tools
- Daily standup generation
- Calendar integration
- Time validation

---

### EmailAgent
**Specialization**: Professional email communication

**Responsibilities:**
1. Draft clear, professional emails
2. Adapt tone based on recipient
3. Include appropriate subject lines
4. Keep emails concise and action-oriented
5. Proofread for clarity and tone

**Available Tools:**
- Email drafting
- Email draft retrieval

---

### SlackAgent
**Specialization**: Team communication via Slack

**Responsibilities:**
1. Draft concise Slack messages
2. Format messages for channels
3. Create daily standup updates
4. Maintain professional yet friendly tone
5. Use threads and mentions appropriately

**Available Tools:**
- Slack message drafting
- Slack message retrieval

---

## System Instructions

The main agent operates with these core instructions:

```
You are Remote Work Buddy, a proactive AI personal executive assistant 
built with OpenAI Agents SDK.

## Your Core Role
Boost remote worker productivity and well-being by handling daily workflows 
autonomously while adapting to user preferences.

## Key Responsibilities

### 1. Greeting & Goal Setting
- Greet users warmly and confirm their goals
- Ask: "What's your top priority today?"

### 2. Daily Planning
- Create optimized daily schedules
- Block 2-hour deep work sessions
- Include 15-minute breaks between meetings
- Present schedules in clear tables

### 3. Task Management
- Manage tasks with priorities (high/medium/low)
- Generate daily standups
- Track weekly progress

### 4. Meeting Coordination
- Schedule meetings within work hours
- Transcribe meetings (Otter/Fireflies style)
- Extract action items

### 5. Communication
- Draft emails (professional tone)
- Create Slack messages for team updates
- Maintain context across sessions

### 6. Wellness & Balance
- Suggest breaks every 60-90 minutes
- Track wellness habits
- Provide weekly insights
- Enforce work-life balance (no work after 8 PM PKT)

## Guardrails (MUST FOLLOW)
1. NEVER schedule tasks after 8:00 PM PKT
2. ALWAYS respect timezone Asia/Karachi
3. Protect user privacy (no sensitive data logging)
4. Encourage regular breaks
5. Validate all time slots before scheduling

## Response Style
- Be concise and direct
- Use bullet points and tables for clarity
- End with "How else can I assist?"
- Use emojis sparingly but warmly
- Maintain professional yet friendly tone
```

---

## Data Models

### CalendarEvent
```python
{
    "title": str,
    "start_time": str,  # ISO format: YYYY-MM-DDTHH:MM:SS
    "end_time": str,    # ISO format: YYYY-MM-DDTHH:MM:SS
    "description": str,
    "attendees": list[str]
}
```

### Task
```python
{
    "title": str,
    "priority": str,    # "high", "medium", "low"
    "due_date": str,    # Optional, ISO format
    "completed": bool
}
```

### EmailDraft
```python
{
    "to": str,
    "subject": str,
    "body": str
}
```

### SlackMessage
```python
{
    "channel": str,
    "message": str
}
```

---

## Usage Examples

### Example 1: Daily Planning
```
User: "Schedule my day. I have 3 priority tasks: project review, team meeting, and code review."

Agent Response:
📋 **Optimized Schedule**

| Time | Activity | Type |
|------|----------|------|
| 09:00 - 11:00 | Project Review | Deep Work |
| 11:00 - 11:15 | Break | Rest |
| 11:15 - 12:00 | Team Meeting | Meeting |
| 12:00 - 13:00 | Lunch | Break |
| 13:00 - 15:00 | Code Review | Deep Work |
| 15:00 - 15:15 | Break | Rest |

✅ Blocked 2h deep work sessions with breaks.
How else can I assist?
```

### Example 2: Task Management
```
User: "Add 'Complete project documentation' as high priority"

Agent Response:
✅ Task added: 'Complete project documentation' (Priority: high)

User: "Show me my tasks"

Agent Response:
📋 **Tasks**

🔄 Complete project documentation (high)
🔄 Review pull requests (medium)
🔄 Update team wiki (low)
```

### Example 3: Wellness Tracking
```
User: "Log a 10-minute walk break"

Agent Response:
✅ Logged walk break for 10 minutes

User: "Show me my weekly wellness insights"

Agent Response:
📊 **Weekly Wellness Insights**

• Breaks taken: 15
• Average break duration: 8.5 minutes
✅ Great job maintaining regular breaks!
```

### Example 4: Guardrails in Action
```
User: "Schedule a task for 9 PM"

Agent Response:
❌ Invalid: Task scheduled after 20:00. 
Work-life balance: no tasks after 8 PM PKT.

User: "Schedule a task for 2 PM"

Agent Response:
✅ Valid: 14:00 is within work hours (9 AM - 8 PM PKT)
```

---

## Architecture

### Project Structure
```
work_buddy/
├── src/work_buddy/
│   ├── __init__.py              # Package entry point
│   ├── agent.py                 # Main RemoteWorkBuddy agent
│   ├── sub_agents.py            # Specialized sub-agents
│   ├── tools.py                 # Tool functions (OpenAI SDK compatible)
│   ├── guardrails.py            # Guardrails implementation
│   └── test_agent.py            # Test suite
├── pyproject.toml               # Project configuration
├── .env                         # Environment variables (not in git)
├── .gitignore                   # Git ignore rules
├── README.md                    # User documentation
└── AGENT_CAPABILITIES.md        # This file
```

### Dependencies
- `openai-agents>=0.12.4` - OpenAI Agents SDK
- `openai>=2.25.0` - OpenAI Python client
- `python-dotenv>=1.0.0` - Environment variable management
- `pydantic>=2.10.0` - Data validation
- `httpx>=0.28.0` - HTTP client

---

## Configuration

### Environment Variables
```bash
# OpenRouter API Configuration
OPENAI_API_KEY=sk-or-v1-...
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_MODEL=wen/qwen3-coder:free

# Optional
TRACING_ENABLED=false
```

### Timezone & Work Hours
- **Timezone**: Asia/Karachi (PKT, UTC+5)
- **Work Start**: 9:00 AM
- **Work End**: 8:00 PM (20:00)
- **Break Frequency**: Every 60-90 minutes
- **Deep Work Block**: 2 hours
- **Break Duration**: 15 minutes

---

## Key Features

✅ **Timezone-Aware** - All times in PKT (Asia/Karachi, UTC+5)
✅ **Work-Life Balance** - Enforces 9 AM - 8 PM work hours
✅ **Multi-Agent Orchestration** - Can delegate to specialists
✅ **Mock Data Storage** - In-memory storage for testing
✅ **Formatted Output** - Tables, bullet points, emojis for clarity
✅ **Privacy Protection** - Detects sensitive data patterns
✅ **OpenAI Agents SDK** - Built with latest OpenAI framework
✅ **Async/Await** - Full async support for non-blocking operations
✅ **Extensible** - Easy to add new tools and sub-agents

---

## What It Does Perfectly

This agent is ideal for:

1. **Remote Workers** - Daily schedule optimization with wellness focus
2. **Teams** - Meeting coordination and transcription
3. **Productivity Tracking** - Task management with priorities
4. **Communication** - Professional email and Slack drafting
5. **Work-Life Balance** - Break reminders and wellness tracking
6. **Daily Standups** - Automated standup generation
7. **Meeting Management** - Scheduling and action item extraction

---

## Future Enhancements

Potential additions:
- Real calendar integration (Google Calendar, Outlook)
- Actual email sending (Gmail, Outlook API)
- Real Slack integration (Slack API)
- Meeting transcription (Otter.ai, Fireflies.io API)
- Database persistence (PostgreSQL, MongoDB)
- Multi-user support
- Custom guardrails per user
- Habit analytics and trends
- AI-powered task prioritization
- Natural language scheduling

---





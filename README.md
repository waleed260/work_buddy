# 🤖 Remote Work Buddy

**Remote Work Buddy** is a proactive AI agent built with the **OpenAI Agents SDK** to boost remote worker productivity and well-being. It acts as your personal executive assistant, handling daily workflows autonomously while adapting to your preferences.

## ✨ Features

### Core Capabilities
- **Daily Planning**: Optimized schedules with 2-hour deep work blocks and 15-minute breaks
- **Task Management**: Todoist/Notion-style task tracking with priorities
- **Meeting Coordination**: Schedule, transcribe (Otter/Fireflies style), and extract action items
- **Communication**: Draft professional emails and Slack messages
- **Wellness Tracking**: Break reminders, habit tracking, weekly insights
- **Guardrails**: Enforces work-life balance (no tasks after 8 PM PKT)

### Sub-Agents
| Agent | Purpose |
|-------|---------|
| **MeetingAgent** | Schedule/transcribe meetings, extract action items |
| **WellnessAgent** | Break reminders, wellness checks, habit tracking |
| **TaskAgent** | Task management, daily standups, prioritization |
| **EmailAgent** | Professional email drafting |
| **SlackAgent** | Team communication via Slack |

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- UV package manager (recommended) or pip

### Installation

1. **Clone the repository**:
```bash
cd /home/waleed/Documents/work_buddy
```

2. **Install dependencies**:
```bash
# Using uv (recommended)
uv pip install -e .

# Or using pip in a virtual environment
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

3. **Set up environment variables**:
```bash
cp .env.example .env
```

Edit `.env` and add your API key:
```bash
# For OpenAI
OPENAI_API_KEY=sk-...

# For Gemini (requires OpenAI-compatible endpoint)
GEMINI_API_KEY=your_key
GEMINI_BASE_URL=https://generativelanguage.googleapis.com
```

### Running Tests

```bash
# Run the test suite
python -m work_buddy.test_agent

# Or directly
python -c "from work_buddy.test_agent import main; main()"
```

### Interactive Mode

```bash
# Start the interactive CLI
python -c "from work_buddy import main; main()"
```

## 📋 Usage Examples

### 1. Initialize the Agent

```python
from work_buddy import RemoteWorkBuddy

# Create agent with API key
buddy = RemoteWorkBuddy(
    api_key="your-api-key",
    timezone="Asia/Karachi",
    tracing_enabled=False  # Tracing disabled by default
)
```

### 2. Chat with the Agent

```python
import asyncio

async def main():
    response = await buddy.chat("Schedule my day")
    print(response)

asyncio.run(main())
```

### 3. Use Tools Directly (Mock Mode)

```python
from work_buddy.tools import (
    add_task,
    get_daily_standup,
    add_calendar_event,
    log_break,
    get_weekly_insights,
)

# Add tasks
add_task("Complete project documentation", "high")
add_task("Review pull requests", "medium")

# Get daily standup
standup = get_daily_standup()
print(standup)

# Add calendar event
add_calendar_event(
    title="Team Standup",
    start_time="2026-03-19T10:00:00",
    end_time="2026-03-19T10:30:00"
)

# Log breaks
log_break("walk", 10)
log_break("stretch", 5)

# Get wellness insights
insights = get_weekly_insights()
print(insights)
```

### 4. Guardrails Validation

```python
from work_buddy.tools import validate_time_slot, get_current_time_pkt

# Validate time slots
result = validate_time_slot(21)  # 9 PM
# ❌ Invalid: Task scheduled after 20:00. Work-life balance: no tasks after 8 PM PKT.

result = validate_time_slot(14)  # 2 PM
# ✅ Valid: 14:00 is within work hours (9 AM - 8 PM PKT)

# Get current time in PKT
current = get_current_time_pkt()
print(f"Current time (PKT): {current}")
```

## 🛡️ Guardrails

Remote Work Buddy enforces these guardrails by default:

| Guardrail | Description |
|-----------|-------------|
| **Timezone** | All times in PKT (Asia/Karachi, UTC+5) |
| **Work Hours** | 9:00 AM - 8:00 PM (no tasks after 8 PM) |
| **Break Reminders** | Suggested every 60-90 minutes |
| **Privacy** | No sensitive data logging (passwords, API keys, etc.) |
| **Meeting Duration** | Maximum 2 hours per meeting |

## 📁 Project Structure

```
work_buddy/
├── src/work_buddy/
│   ├── __init__.py          # Package entry point
│   ├── agent.py             # Main RemoteWorkBuddy agent
│   ├── sub_agents.py        # Specialized sub-agents
│   ├── tools.py             # Tool functions (OpenAI SDK compatible)
│   ├── guardrails.py        # Guardrails implementation
│   └── test_agent.py        # Test suite
├── pyproject.toml           # Project configuration
├── .env.example             # Environment variables template
└── README.md                # This file
```


```

## 🧪 Testing

The test suite covers all components:

```bash
# Run all tests
python -c "from work_buddy.test_agent import main; main()"

# Expected output:
# ✅ Agent initialization
# ✅ Task management
# ✅ Calendar management
# ✅ Wellness tracking
# ✅ Guardrails validation
# ✅ Email drafting
# ✅ Meeting transcription
```

## 📝 Example Output

### Daily Standup
```
📋 **Daily Standup**

**Completed:**
  ✅ Finished quarterly report

**In Progress:**
  🔄 Complete project documentation (high)
  🔄 Review pull requests (medium)
  🔄 Update team wiki (low)
```

### Wellness Insights
```
📊 **Weekly Wellness Insights**

• Breaks taken: 15
• Average break duration: 8.5 minutes
✅ Great job maintaining regular breaks!
```

### Schedule Table
```
📅 **Today's Schedule**

| Time | Event |
|------|-------|
| 10:00 - 10:30 | Team Standup |
| 14:00 - 16:00 | Deep Work Session |
```

 🚀

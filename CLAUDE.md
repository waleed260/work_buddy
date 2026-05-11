# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Remote Work Buddy is an AI-powered personal executive assistant for remote workers, built with the OpenAI Agents SDK. It uses OpenRouter to access various LLM models and provides task management, calendar scheduling, wellness tracking, and communication tools.

## Development Setup

### Environment Configuration

1. Activate virtual environment:
```bash
source .venv/bin/activate
```

2. Configure OpenRouter API in `.env`:
```bash
OPENAI_API_KEY=sk-or-v1-your_key_here
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_MODEL=openai/gpt-3.5-turbo
```

### Running Tests

```bash
# Comprehensive test suite
python test_final.py

# Quick demo
python quickstart.py

# Simple API test
python test_simple.py

# Interactive CLI
python -m work_buddy
```

## Architecture

### Core Components

**Main Agent** (`src/work_buddy/agent.py`)
- `RemoteWorkBuddy` class orchestrates all functionality
- Initializes 5 sub-agents and 20+ tools
- Uses OpenRouter provider for LLM access
- Enforces guardrails (work hours, timezone)

**Provider System** (`src/work_buddy/providers.py`)
- Simplified to use OpenRouter only
- `create_model()` - Creates OpenAI-compatible model instances
- `create_model_from_env()` - Loads configuration from environment variables
- Disables OpenAI Responses API (`use_responses=False`) for OpenRouter compatibility

**Tools** (`src/work_buddy/tools.py`)
- All tools use `@function_tool` decorator from OpenAI Agents SDK
- Tools maintain in-memory state (global variables: `_tasks`, `_calendar_events`, etc.)
- Each tool has both a wrapped version (for agent use) and raw function (for direct calls)
- Access raw functions with underscore prefix: `_add_task()`, `_get_tasks()`

**Sub-Agents** (`src/work_buddy/sub_agents.py`)
- 5 specialized agents: Meeting, Wellness, Task, Email, Slack
- Each created with `Agent()` class from OpenAI Agents SDK
- Have their own instructions and subset of tools

**Guardrails** (`src/work_buddy/guardrails.py`)
- `Guardrails` class enforces work-life balance
- Timezone: Asia/Karachi (PKT, UTC+5)
- Work hours: 9 AM - 8 PM (hard limit)
- Validates time slots, meeting durations, break frequency

### Key Design Patterns

**Agent Initialization Flow:**
1. Load environment variables (API key, base URL, model)
2. Create OpenRouter provider with `use_responses=False`
3. Get model instance from provider
4. Initialize sub-agents
5. Create main agent with tools and model

**Tool State Management:**
- All state is in-memory (not persisted)
- Global variables at module level in `tools.py`
- State resets when Python process restarts
- For production, replace with database/API calls

**OpenRouter Integration:**
- Must use `use_responses=False` in OpenAIProvider
- Model names must support function calling (e.g., `openai/gpt-3.5-turbo`)
- Models without tool support (like `qwen/qwen-2.5-coder-32b-instruct`) only work for basic chat

## Important Implementation Details

### Adding New Tools

1. Define raw function with underscore prefix in `tools.py`:
```python
def _my_new_tool(param: str) -> str:
    """Tool description."""
    # Implementation
    return result
```

2. Export wrapped version:
```python
my_new_tool = function_tool(_my_new_tool)
```

3. Add to agent's tool list in `agent.py`:
```python
def _get_agent_tools(self) -> list:
    return [
        # ... existing tools
        my_new_tool,
    ]
```

### Modifying Agent Behavior

The main agent's instructions are in `_get_system_instructions()` method. Key sections:
- Core role and responsibilities
- Guardrails (MUST FOLLOW section)
- Response style
- Handoff to sub-agents

### Timezone Handling

All time operations use PKT (Asia/Karachi, UTC+5):
- `get_current_time_pkt()` - Returns current time in PKT
- `validate_time_slot(hour)` - Validates against work hours
- `is_within_work_hours()` - Checks if current time is 9 AM - 8 PM PKT

### Model Compatibility

OpenRouter models with tool support:
- `openai/gpt-3.5-turbo` (current default)
- `openai/gpt-4o-mini`
- `anthropic/claude-3-5-sonnet`
- `google/gemini-pro-1.5`
- `meta-llama/llama-3.1-70b-instruct`

Models without tool support will only work for basic chat, not autonomous tool use.

## Testing Strategy

Tests use direct tool calls (underscore-prefixed functions) to verify functionality without requiring API calls. The agent's chat functionality requires a valid API key and model with tool support.

## Configuration Files

- `.env` - Active configuration (not in git)
- `.env.example` - Template with available models
- `pyproject.toml` - Dependencies and project metadata
- Entry point: `work-buddy` command maps to `work_buddy:main`

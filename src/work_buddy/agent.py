"""
Remote Work Buddy - Main AI Agent
Built with OpenAI Agents SDK.
"""
from dotenv import load_dotenv
import os
from typing import Optional
from agents.agent import Agent
from agents.run import Runner 
from .tools import (
    check_calendar_events,
    add_calendar_event,
    get_calendar_free_slots,
    get_todays_schedule,
    add_task,
    get_tasks,
    complete_task,
    get_daily_standup,
    suggest_break,
    log_break,
    get_weekly_insights,
    track_habit,
    transcribe_meeting,
    extract_action_items,
    validate_time_slot,
    get_current_time_pkt,
    is_within_work_hours,
    suggest_optimal_focus_time,
    draft_email,
    get_email_drafts,
    draft_slack_message,
    get_slack_messages,
)
from .guardrails import Guardrails
from .sub_agents import (
    create_meeting_agent,
    create_wellness_agent,
    create_task_agent,
    create_email_agent,
    create_slack_agent,
)
load_dotenv()

class RemoteWorkBuddy:
    """
    Remote Work Buddy AI Agent.
    
    Acts as a personal executive assistant for remote workers,
    handling daily workflows autonomously while adapting to user preferences.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timezone: str = "Asia/Karachi",
        tracing_enabled: bool = False,
    ):
        """
        Initialize Remote Work Buddy.
        
        Args:
            api_key: API key (OpenAI or Gemini)
            base_url: Custom base URL for API
            timezone: User timezone (default: PKT - Asia/Karachi)
            tracing_enabled: Enable/disable tracing (default: False)
        """
        self.timezone = timezone
        self.tracing_enabled = tracing_enabled
        self.guardrails = Guardrails(timezone=timezone, work_end_hour=20)
        
        # Store API configuration
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4")
        
        # Set environment variables for the SDK
        if self.api_key:
            os.environ["OPENAI_API_KEY"] = self.api_key
        if self.base_url:
            os.environ["OPENAI_BASE_URL"] = self.base_url
        
        # Initialize sub-agents
        self.meeting_agent = create_meeting_agent()
        self.wellness_agent = create_wellness_agent()
        self.task_agent = create_task_agent()
        self.email_agent = create_email_agent()
        self.slack_agent = create_slack_agent()
        
        # Create main agent
        self.main_agent = self._create_main_agent()
    
    def _create_main_agent(self) -> Agent:
        """Create the main Remote Work Buddy agent."""
        if not self.api_key:
            return None

        # Use OpenRouter provider
        from .providers import create_model

        model_obj = create_model(
            api_key=self.api_key,
            base_url=self.base_url,
            model_name=self.model
        )

        return Agent(
            name="RemoteWorkBuddy",
            instructions=self._get_system_instructions(),
            tools=self._get_agent_tools(),
            model=model_obj,
        )
    
    def _get_system_instructions(self) -> str:
        """Get comprehensive system instructions for the main agent."""
        return f"""You are Remote Work Buddy, a proactive AI personal executive assistant built with OpenAI Agents SDK.

## Your Core Role
Boost remote worker productivity and well-being by handling daily workflows autonomously while adapting to user preferences.

## User Context
- Timezone: {self.timezone} (PKT)
- Work Hours: 9:00 AM - 8:00 PM (no tasks after 8 PM)

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
2. ALWAYS respect timezone {self.timezone}
3. Protect user privacy (no sensitive data logging)
4. Encourage regular breaks
5. Validate all time slots before scheduling

## Response Style
- Be concise and direct
- Use bullet points and tables for clarity
- End with "How else can I assist?"
- Use emojis sparingly but warmly
- Maintain professional yet friendly tone

## Handoff to Sub-Agents
When specialized help is needed:
- MeetingAgent: For scheduling/transcribing meetings
- WellnessAgent: For breaks and wellness checks
- TaskAgent: For detailed task management
- EmailAgent: For professional email drafting
- SlackAgent: For team communication

## Proactive Suggestions
- Daily standups (morning)
- Weekly reviews (Friday afternoon)
- Focus sessions (2-hour blocks)
- Break reminders (every 60-90 min)
- Habit tracking insights"""
    
    def _get_agent_tools(self) -> list:
        """Get tools available to the main agent."""
        return [
            # Calendar tools
            check_calendar_events,
            add_calendar_event,
            get_calendar_free_slots,
            get_todays_schedule,
            # Task tools
            add_task,
            get_tasks,
            complete_task,
            get_daily_standup,
            # Wellness tools
            suggest_break,
            log_break,
            get_weekly_insights,
            track_habit,
            # Communication tools
            draft_email,
            get_email_drafts,
            draft_slack_message,
            get_slack_messages,
            # Transcription
            transcribe_meeting,
            extract_action_items,
            # Guardrails
            validate_time_slot,
            is_within_work_hours,
            get_current_time_pkt,
            suggest_optimal_focus_time,
        ]
    
    async def chat(self, user_message: str) -> str:
        """
        Process a user message and return a response.
        
        Args:
            user_message: The user's input message
            
        Returns:
            Agent's response
        """
        if not self.api_key or not self.main_agent:
            return "⚠️ API key not configured. Please set OPENAI_API_KEY environment variable.\n\nIn mock mode, you can still use the direct tool functions."
        
        try:
            result = await Runner.run(
                self.main_agent,
                user_message,
            )
            return result.final_output
        except Exception as e:
            return f"⚠️ API Error: {str(e)}\n\nPlease check your API key and model configuration."
    
    def get_daily_schedule(self) -> str:
        """Get today's schedule as a formatted table."""
        from .tools import _get_todays_schedule
        return _get_todays_schedule()

    def get_daily_standup(self) -> str:
        """Get daily standup summary."""
        from .tools import _get_daily_standup
        return _get_daily_standup()
    
    def suggest_schedule(self, priorities: list[str]) -> str:
        """
        Suggest an optimized daily schedule based on priorities.
        
        Args:
            priorities: List of priority tasks
            
        Returns:
            Formatted schedule suggestion
        """
        schedule = "📋 **Optimized Schedule**\n\n"
        schedule += "| Time | Activity | Type |\n"
        schedule += "|------|----------|------|\n"
        
        current_hour = 9
        for i, priority in enumerate(priorities[:4]):
            end_hour = current_hour + 2
            schedule += f"| {current_hour:02d}:00 - {end_hour:02d}:00 | {priority} | Deep Work |\n"
            current_hour = end_hour + 1  # 1 hour break
        
        schedule += "\n✅ Blocked 2h deep work sessions with breaks.\n"
        schedule += "How else can I assist?"
        return schedule


async def create_remote_work_buddy(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timezone: str = "Asia/Karachi",
    tracing_enabled: bool = False,
) -> RemoteWorkBuddy:
    """
    Factory function to create Remote Work Buddy instance.
    
    Args:
        api_key: API key for the LLM backend
        base_url: Custom base URL for API
        timezone: User timezone
        tracing_enabled: Enable/disable tracing
        
    Returns:
        Configured RemoteWorkBuddy instance
    """
    return RemoteWorkBuddy(
        api_key=api_key,
        base_url=base_url,
        timezone=timezone,
        tracing_enabled=tracing_enabled,
    )

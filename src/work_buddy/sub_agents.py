"""
Sub-agents for Remote Work Buddy AI Agent.
Specialized agents for meetings, wellness, and task management.
"""

from agents import Agent
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


def create_meeting_agent() -> Agent:
    """
    Create MeetingAgent specialized for handling meetings.
    
    Capabilities:
    - Schedule meetings
    - Transcribe meetings (Otter/Fireflies style)
    - Extract action items
    - Validate meeting times against guardrails
    """
    return Agent(
        name="MeetingAgent",
        instructions="""You are MeetingAgent, specialized in handling all meeting-related tasks.

Your responsibilities:
1. Schedule meetings within work hours (9 AM - 8 PM PKT)
2. Transcribe meetings and extract key points
3. Identify action items from meeting transcripts
4. Ensure meetings don't exceed 2 hours
5. Suggest optimal meeting times based on calendar availability

Always validate times against guardrails and respect work-life balance.
Respond concisely with clear action items and next steps.""",
        tools=[
            check_calendar_events,
            add_calendar_event,
            get_calendar_free_slots,
            transcribe_meeting,
            extract_action_items,
            validate_time_slot,
        ]
    )


def create_wellness_agent() -> Agent:
    """
    Create WellnessAgent specialized in user well-being.
    
    Capabilities:
    - Suggest breaks
    - Track wellness habits
    - Provide weekly insights
    - Monitor work-life balance
    """
    return Agent(
        name="WellnessAgent",
        instructions="""You are WellnessAgent, dedicated to user well-being and work-life balance.

Your responsibilities:
1. Suggest breaks every 60-90 minutes
2. Track wellness habits (hydration, movement, meals)
3. Provide weekly wellness insights
4. Remind users about work hours (no work after 8 PM PKT)
5. Encourage healthy remote work practices

Be supportive and encouraging. Prioritize user health over productivity.
Use emojis to make interactions friendly and warm.""",
        tools=[
            suggest_break,
            log_break,
            track_habit,
            get_weekly_insights,
            is_within_work_hours,
            get_current_time_pkt,
        ]
    )


def create_task_agent() -> Agent:
    """
    Create TaskAgent specialized in task management.
    
    Capabilities:
    - Add/manage tasks (Todoist/Notion style)
    - Prioritize daily tasks
    - Generate daily standups
    - Schedule tasks within work hours
    """
    return Agent(
        name="TaskAgent",
        instructions="""You are TaskAgent, specialized in task management and productivity.

Your responsibilities:
1. Add and organize tasks with priorities
2. Generate daily standup summaries
3. Schedule tasks within work hours (9 AM - 8 PM PKT)
4. Prioritize tasks using Eisenhower Matrix principles
5. Track task completion and provide motivation

Always respect guardrails:
- No tasks after 8 PM PKT
- Include breaks between deep work sessions
- Balance high-priority tasks throughout the day

Respond with clear tables and bullet points for task lists.""",
        tools=[
            add_task,
            get_tasks,
            complete_task,
            get_daily_standup,
            get_calendar_free_slots,
            validate_time_slot,
        ]
    )


def create_email_agent() -> Agent:
    """
    Create EmailAgent specialized in email drafting.
    
    Capabilities:
    - Draft professional emails
    - Schedule email sending
    - Manage email templates
    """
    return Agent(
        name="EmailAgent",
        instructions="""You are EmailAgent, specialized in professional email communication.

Your responsibilities:
1. Draft clear, professional emails
2. Adapt tone based on recipient (formal/casual)
3. Include appropriate subject lines
4. Keep emails concise and action-oriented
5. Proofread for clarity and tone

Always maintain professionalism and respect privacy guardrails.""",
        tools=[
            draft_email,
            get_email_drafts,
        ]
    )


def create_slack_agent() -> Agent:
    """
    Create SlackAgent specialized in team communication.
    
    Capabilities:
    - Draft Slack messages
    - Format messages for channels
    - Manage standup updates
    """
    return Agent(
        name="SlackAgent",
        instructions="""You are SlackAgent, specialized in team communication via Slack.

Your responsibilities:
1. Draft concise Slack messages
2. Format messages appropriately for channels
3. Create daily standup updates
4. Maintain professional yet friendly tone
5. Use threads and mentions appropriately

Keep messages brief and actionable. Use emojis appropriately.""",
        tools=[
            draft_slack_message,
            get_slack_messages,
        ]
    )

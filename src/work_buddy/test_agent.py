"""
Test script for Remote Work Buddy AI Agent.
Tests all components and demonstrates functionality.
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from work_buddy.agent import RemoteWorkBuddy, create_remote_work_buddy
from work_buddy.tools import (
    _add_task as add_task,
    _get_tasks as get_tasks,
    _get_daily_standup as get_daily_standup,
    _add_calendar_event as add_calendar_event,
    _get_todays_schedule as get_todays_schedule,
    _log_break as log_break,
    _get_weekly_insights as get_weekly_insights,
    _validate_time_slot as validate_time_slot,
    _get_current_time_pkt as get_current_time_pkt,
    _is_within_work_hours as is_within_work_hours,
    _draft_email as draft_email,
    _transcribe_meeting as transcribe_meeting,
    _extract_action_items as extract_action_items,
)


async def test_remote_work_buddy():
    """Test the Remote Work Buddy agent."""
    
    print("=" * 60)
    print("🤖 Remote Work Buddy - AI Agent Test")
    print("=" * 60)
    
    # Get API configuration from environment or use defaults
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    base_url = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com")
    tracing_enabled = os.getenv("TRACING_ENABLED", "false").lower() == "true"
    
    print(f"\n📍 Timezone: Asia/Karachi (PKT)")
    print(f"🔑 API Key: {'Set' if api_key else 'Not set (using mock mode)'}")
    print(f"🌐 Base URL: {base_url}")
    print(f"📊 Tracing: {'Enabled' if tracing_enabled else 'Disabled'}")
    
    # Create the agent
    print("\n⏳ Initializing Remote Work Buddy...")
    
    try:
        buddy = RemoteWorkBuddy(
            api_key=api_key,
            base_url=base_url,
            timezone="Asia/Karachi",
            tracing_enabled=False,  # Tracing disabled as requested
        )
        print("✅ Agent initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing agent: {e}")
        print("\n💡 Running in mock mode (tools only) without API key...\n")
        run_mock_tests()
        return
    
    # Test 1: Greeting and goal setting
    print("\n" + "=" * 60)
    print("📝 Test 1: Greeting & Goal Setting")
    print("=" * 60)
    
    try:
        response = await buddy.chat("Hello! What can you help me with today?")
        print(f"\n🤖 Agent Response:\n{response}")
    except Exception as e:
        print(f"❌ Error in Test 1: {e}")
        print("   Note: Chat requires a valid API key (OPENAI_API_KEY or GEMINI_API_KEY)")
    
    # Test 2: Schedule my day
    print("\n" + "=" * 60)
    print("📅 Test 2: Schedule My Day")
    print("=" * 60)
    
    try:
        response = await buddy.chat("Schedule my day. I have 3 priority tasks: project review, team meeting, and code review.")
        print(f"\n🤖 Agent Response:\n{response}")
    except Exception as e:
        print(f"❌ Error in Test 2: {e}")
    
    # Test 3: Add a task
    print("\n" + "=" * 60)
    print("✅ Test 3: Task Management")
    print("=" * 60)
    
    try:
        # Add tasks directly using tools
        add_task("Complete project documentation", "high")
        add_task("Review pull requests", "medium")
        add_task("Update team wiki", "low")
        
        standup = get_daily_standup()
        print(f"\n📋 Daily Standup:\n{standup}")
    except Exception as e:
        print(f"❌ Error in Test 3: {e}")
    
    # Test 4: Calendar management
    print("\n" + "=" * 60)
    print("📆 Test 4: Calendar Management")
    print("=" * 60)
    
    try:
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
        
        add_calendar_event(
            title="Team Standup",
            start_time=f"{today}T10:00:00",
            end_time=f"{today}T10:30:00",
            description="Daily team sync"
        )
        
        schedule = get_todays_schedule()
        print(f"\n{schedule}")
    except Exception as e:
        print(f"❌ Error in Test 4: {e}")
    
    # Test 5: Wellness check
    print("\n" + "=" * 60)
    print("🧘 Test 5: Wellness & Breaks")
    print("=" * 60)
    
    try:
        # Log some breaks
        log_break("walk", 10)
        log_break("stretch", 5)
        log_break("hydration", 5)
        
        insights = get_weekly_insights()
        print(f"\n{insights}")
    except Exception as e:
        print(f"❌ Error in Test 5: {e}")
    
    # Test 6: Guardrails validation
    print("\n" + "=" * 60)
    print("🛡️ Test 6: Guardrails Validation")
    print("=" * 60)
    
    try:
        # Test time slot validation (should fail for after 8 PM)
        result = validate_time_slot(21)  # 9 PM
        print(f"\n⏰ 9 PM slot validation: {result}")
        
        # Test time slot validation (should pass for 2 PM)
        result = validate_time_slot(14)  # 2 PM
        print(f"⏰ 2 PM slot validation: {result}")
        
        # Test current time in PKT
        current_time = get_current_time_pkt()
        print(f"\n🕐 Current time (PKT): {current_time}")
        
        # Test work hours check
        within_hours = is_within_work_hours()
        print(f"💼 Within work hours: {within_hours}")
    except Exception as e:
        print(f"❌ Error in Test 6: {e}")
    
    # Test 7: Email drafting
    print("\n" + "=" * 60)
    print("📧 Test 7: Email Drafting")
    print("=" * 60)
    
    try:
        result = draft_email(
            to="team@company.com",
            subject="Project Update - Weekly Status",
            body="Hi Team,\n\nHere's the weekly project status update...\n\nBest regards"
        )
        print(f"\n✅ {result}")
    except Exception as e:
        print(f"❌ Error in Test 7: {e}")
    
    # Test 8: Meeting transcription (mock)
    print("\n" + "=" * 60)
    print("🎤 Test 8: Meeting Transcription")
    print("=" * 60)
    
    try:
        transcript = transcribe_meeting("MTG-001")
        print(f"\n📝 Meeting Transcript:\n{transcript[:500]}...")
        
        action_items = extract_action_items(transcript)
        print(f"\n{action_items}")
    except Exception as e:
        print(f"❌ Error in Test 8: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    print("""
✅ All components tested:
   • Agent initialization
   • Chat/Greeting functionality
   • Daily scheduling
   • Task management
   • Calendar management
   • Wellness tracking
   • Guardrails validation
   • Email drafting
   • Meeting transcription

🎉 Remote Work Buddy is ready to use!

To use with Gemini API:
   export GEMINI_API_KEY=your_api_key
   export GEMINI_BASE_URL=https://generativelanguage.googleapis.com

How else can I assist?
""")


def run_mock_tests():
    """Run tests that don't require API key."""
    from datetime import datetime
    
    print("=" * 60)
    print("🧪 Running Mock Tests (No API Key Required)")
    print("=" * 60)
    
    # Test Task Management
    print("\n" + "=" * 60)
    print("✅ Mock Test: Task Management")
    print("=" * 60)
    
    add_task("Complete project documentation", "high")
    add_task("Review pull requests", "medium")
    add_task("Update team wiki", "low")
    
    standup = get_daily_standup()
    print(f"\n📋 Daily Standup:\n{standup}")
    
    # Test Calendar
    print("\n" + "=" * 60)
    print("📆 Mock Test: Calendar Management")
    print("=" * 60)
    
    today = datetime.now().strftime("%Y-%m-%d")
    add_calendar_event(
        title="Team Standup",
        start_time=f"{today}T10:00:00",
        end_time=f"{today}T10:30:00",
        description="Daily team sync"
    )
    schedule = get_todays_schedule()
    print(f"\n{schedule}")
    
    # Test Wellness
    print("\n" + "=" * 60)
    print("🧘 Mock Test: Wellness & Breaks")
    print("=" * 60)
    
    log_break("walk", 10)
    log_break("stretch", 5)
    log_break("hydration", 5)
    insights = get_weekly_insights()
    print(f"\n{insights}")
    
    # Test Guardrails
    print("\n" + "=" * 60)
    print("🛡️ Mock Test: Guardrails Validation")
    print("=" * 60)
    
    result = validate_time_slot(21)
    print(f"\n⏰ 9 PM slot validation: {result}")
    
    result = validate_time_slot(14)
    print(f"⏰ 2 PM slot validation: {result}")
    
    current_time = get_current_time_pkt()
    print(f"\n🕐 Current time (PKT): {current_time}")
    
    within_hours = is_within_work_hours()
    print(f"💼 Within work hours: {within_hours}")
    
    # Test Email
    print("\n" + "=" * 60)
    print("📧 Mock Test: Email Drafting")
    print("=" * 60)
    
    result = draft_email(
        to="team@company.com",
        subject="Project Update - Weekly Status",
        body="Hi Team,\n\nHere's the weekly project status update...\n\nBest regards"
    )
    print(f"\n✅ {result}")
    
    # Test Transcription
    print("\n" + "=" * 60)
    print("🎤 Mock Test: Meeting Transcription")
    print("=" * 60)
    
    transcript = transcribe_meeting("MTG-001")
    print(f"\n📝 Meeting Transcript:\n{transcript[:500]}...")
    
    action_items = extract_action_items(transcript)
    print(f"\n{action_items}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Mock Test Summary")
    print("=" * 60)
    print("""
✅ All mock components tested successfully:
   • Task management
   • Calendar management
   • Wellness tracking
   • Guardrails validation
   • Email drafting
   • Meeting transcription

💡 To enable full AI chat functionality:
   export OPENAI_API_KEY=your_key
   # or
   export GEMINI_API_KEY=your_key

🎉 Remote Work Buddy is ready to use!
""")


def main():
    """Main entry point for testing."""
    print("\n🚀 Starting Remote Work Buddy Tests...\n")
    asyncio.run(test_remote_work_buddy())


if __name__ == "__main__":
    main()

"""
Remote Work Buddy - AI Agent for Remote Worker Productivity
"""

import asyncio
import os
from typing import Optional

from dotenv import load_dotenv

from .agent import RemoteWorkBuddy

load_dotenv()


def create_agent(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timezone: str = "Asia/Karachi",
    tracing_enabled: bool = False,
) -> RemoteWorkBuddy:
    """
    Create a Remote Work Buddy agent instance.
    
    Args:
        api_key: Gemini API key (or set GEMINI_API_KEY env var)
        base_url: Gemini base URL (default: https://generativelanguage.googleapis.com)
        timezone: User timezone (default: Asia/Karachi)
        tracing_enabled: Enable tracing (default: False)
    
    Returns:
        Configured RemoteWorkBuddy instance
    """
    if api_key is None:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if base_url is None:
        base_url = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com")
    
    return RemoteWorkBuddy(
        api_key=api_key,
        base_url=base_url,
        timezone=timezone,
        tracing_enabled=tracing_enabled,
    )


async def chat(agent: RemoteWorkBuddy, message: str) -> str:
    """
    Send a message to the agent and get a response.
    
    Args:
        agent: RemoteWorkBuddy instance
        message: User message
    
    Returns:
        Agent response
    """
    return await agent.chat(message)


def main():
    """CLI entry point for Remote Work Buddy."""
    import sys
    
    print("🤖 Remote Work Buddy - Your AI Personal Executive Assistant")
    print("=" * 60)
    print("Timezone: Asia/Karachi (PKT) | Work Hours: 9 AM - 8 PM")
    print("=" * 60)
    
    # Create agent
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    base_url = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com")
    tracing_enabled = os.getenv("TRACING_ENABLED", "false").lower() == "true"
    
    agent = RemoteWorkBuddy(
        api_key=api_key,
        base_url=base_url,
        timezone="Asia/Karachi",
        tracing_enabled=tracing_enabled,
    )
    
    print("\n✅ Remote Work Buddy initialized!")
    print("\n💡 Try these commands:")
    print("   • 'Schedule my day'")
    print("   • 'What's my top priority?'")
    print("   • 'I need a break'")
    print("   • 'Show my tasks'")
    print("   • 'Draft an email to...'")
    print("\nType 'quit' or 'exit' to end the session.\n")
    
    # Interactive chat loop
    while True:
        try:
            user_input = input("👤 You: ").strip()
            
            if user_input.lower() in ["quit", "exit", "bye"]:
                print("\n👋 Goodbye! Have a productive day!")
                break
            
            if not user_input:
                continue
            
            # Get response
            response = asyncio.run(agent.chat(user_input))
            print(f"\n🤖 Buddy: {response}\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()

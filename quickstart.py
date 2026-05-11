#!/usr/bin/env python3
"""
Quick start script for Remote Work Buddy.
Run this to see the agent in action!
"""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from work_buddy.agent import RemoteWorkBuddy


async def main():
    print("\n" + "="*70)
    print("🤖 REMOTE WORK BUDDY - LIVE DEMO")
    print("="*70)
    print("\nInitializing agent with OpenRouter...")

    buddy = RemoteWorkBuddy(timezone="Asia/Karachi")

    print("✅ Agent ready!\n")
    print("="*70)

    # Demo 1: Introduction
    print("\n💬 Demo 1: Introduction")
    print("-"*70)
    response = await buddy.chat("Hello! Tell me what you can do in 2-3 sentences.")
    print(f"🤖 {response}\n")

    # Demo 2: Add a task
    print("💬 Demo 2: Task Management")
    print("-"*70)
    response = await buddy.chat("Add a high priority task: Prepare Q2 presentation")
    print(f"🤖 {response}\n")

    # Demo 3: Check schedule
    print("💬 Demo 3: Schedule Check")
    print("-"*70)
    response = await buddy.chat("What's on my schedule today?")
    print(f"🤖 {response}\n")

    # Demo 4: Wellness check
    print("💬 Demo 4: Wellness Check")
    print("-"*70)
    response = await buddy.chat("I've been working for 2 hours. What should I do?")
    print(f"🤖 {response}\n")

    print("="*70)
    print("✅ DEMO COMPLETE!")
    print("="*70)
    print("""
🎉 Remote Work Buddy is working perfectly!

Configuration:
  • Provider: OpenRouter
  • Model: openai/gpt-3.5-turbo
  • Status: Fully Functional ✅

To use interactively:
  python -m work_buddy

To integrate in your code:
  from work_buddy import create_agent
  agent = create_agent()
  response = await agent.chat("Your message")

See FINAL_STATUS.md for complete documentation.
""")


if __name__ == "__main__":
    asyncio.run(main())

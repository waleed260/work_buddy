## 2026-05-13 - Visual Hierarchy in Text-based Interfaces
**Learning:** In a CLI or chat-based interface, users can easily lose track of importance when tasks are presented in a simple list. Using consistent color-coded emojis (🔴, 🟡, 🔵) combined with priority sorting creates an immediate visual hierarchy that helps users focus on what matters most first.
**Action:** Always implement sorting by priority for task/event lists and use consistent visual markers for severity or priority levels across all status-related outputs.

## 2026-05-17 - Scanning Efficiency and Encouraging Empty States
**Learning:** Placing priority markers immediately after status indicators (e.g., `🔄 🔴 Title`) creates a tighter anchor for the eye, allowing users to scan both completion status and urgency in a single vertical pass. Additionally, using "delightful" empty states (emojis like 🥳, 🚀) transforms a "no results" dead-end into a moment of positive reinforcement, which is critical for a productivity-focused agent.
**Action:** When displaying lists, group meta-information (status, priority) before the content. Always enhance empty states with encouraging language and positive emojis.

## 2026-05-30 - Scannability Patterns for Multi-property List Items
**Learning:** For text-heavy CLI outputs representing objects with multiple properties (like email or Slack drafts), scannability is significantly enhanced by using bold Markdown for labels (e.g., **To:**, **Subject:**) to act as visual anchors. For multi-line text fields, indenting subsequent lines by two spaces preserves the vertical flow of the list markers, preventing the eye from getting "lost" in wrapped text.
**Action:** Apply bold label formatting and multi-line indentation (via `.replace('\n', '\n  ')`) to all complex list-based outputs. Ensure empty states remain delightful with consistent emoji use (✨, 🌊).

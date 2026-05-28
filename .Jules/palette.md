## 2026-05-13 - Visual Hierarchy in Text-based Interfaces
**Learning:** In a CLI or chat-based interface, users can easily lose track of importance when tasks are presented in a simple list. Using consistent color-coded emojis (🔴, 🟡, 🔵) combined with priority sorting creates an immediate visual hierarchy that helps users focus on what matters most first.
**Action:** Always implement sorting by priority for task/event lists and use consistent visual markers for severity or priority levels across all status-related outputs.

## 2026-05-17 - Scanning Efficiency and Encouraging Empty States
**Learning:** Placing priority markers immediately after status indicators (e.g., `🔄 🔴 Title`) creates a tighter anchor for the eye, allowing users to scan both completion status and urgency in a single vertical pass. Additionally, using "delightful" empty states (emojis like 🥳, 🚀) transforms a "no results" dead-end into a moment of positive reinforcement, which is critical for a productivity-focused agent.
**Action:** When displaying lists, group meta-information (status, priority) before the content. Always enhance empty states with encouraging language and positive emojis.

## 2026-05-29 - Polish in CLI-based "Interfaces"
**Learning:** For AI agents, the text-based CLI output is the primary interface. Using bold Markdown for labels, providing content previews (e.g., email body snippets), and correctly indenting multi-line messages significantly reduces cognitive load and makes the tool output feel "designed" rather than just a raw data dump.
**Action:** Apply consistent formatting (bold labels, bullet points, indentation) to all tool outputs that return lists or objects. Always provide a snippet of content if the full content might be too long.

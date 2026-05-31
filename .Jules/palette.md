## 2026-05-13 - Visual Hierarchy in Text-based Interfaces
**Learning:** In a CLI or chat-based interface, users can easily lose track of importance when tasks are presented in a simple list. Using consistent color-coded emojis (🔴, 🟡, 🔵) combined with priority sorting creates an immediate visual hierarchy that helps users focus on what matters most first.
**Action:** Always implement sorting by priority for task/event lists and use consistent visual markers for severity or priority levels across all status-related outputs.

## 2026-05-17 - Scanning Efficiency and Encouraging Empty States
**Learning:** Placing priority markers immediately after status indicators (e.g., `🔄 🔴 Title`) creates a tighter anchor for the eye, allowing users to scan both completion status and urgency in a single vertical pass. Additionally, using "delightful" empty states (emojis like 🥳, 🚀) transforms a "no results" dead-end into a moment of positive reinforcement, which is critical for a productivity-focused agent.
**Action:** When displaying lists, group meta-information (status, priority) before the content. Always enhance empty states with encouraging language and positive emojis.

## 2026-05-31 - Labeling and Multi-line Alignment in CLI
**Learning:** In text-based interfaces, bolding property labels (e.g., **To:**, **Subject:**) creates visual anchors that significantly speed up scanning. Furthermore, when dealing with multi-line content in bulleted lists, indenting subsequent lines by two spaces to align them with the vertical start of the text block (ignoring the bullet) prevents visual clutter and maintains structural integrity.
**Action:** Use bold Markdown for property labels in CLI outputs. Ensure multi-line strings in lists are indented by two spaces using `.replace('\n', '\n  ')` to create a consistent vertical alignment for content blocks.

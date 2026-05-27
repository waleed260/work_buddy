## 2026-05-13 - Visual Hierarchy in Text-based Interfaces
**Learning:** In a CLI or chat-based interface, users can easily lose track of importance when tasks are presented in a simple list. Using consistent color-coded emojis (🔴, 🟡, 🔵) combined with priority sorting creates an immediate visual hierarchy that helps users focus on what matters most first.
**Action:** Always implement sorting by priority for task/event lists and use consistent visual markers for severity or priority levels across all status-related outputs.

## 2026-05-17 - Scanning Efficiency and Encouraging Empty States
**Learning:** Placing priority markers immediately after status indicators (e.g., `🔄 🔴 Title`) creates a tighter anchor for the eye, allowing users to scan both completion status and urgency in a single vertical pass. Additionally, using "delightful" empty states (emojis like 🥳, 🚀) transforms a "no results" dead-end into a moment of positive reinforcement, which is critical for a productivity-focused agent.
**Action:** When displaying lists, group meta-information (status, priority) before the content. Always enhance empty states with encouraging language and positive emojis.

## 2026-05-28 - Information Density and Scanning in Text Lists
**Learning:** For list items with multiple properties (like Email or Slack drafts), using bold Markdown for labels (e.g., **To:**, **Channel:**) creates clear visual anchors that significantly improve scanning speed. Furthermore, when content can be multi-line (like Slack messages), indenting subsequent lines by two spaces preserves the list's vertical rhythm and prevents the eye from losing the "bullet point" context.
**Action:** Always use bold labels for property-value pairs in text lists. Implement `message.replace('\n', '\n  ')` for multi-line content in list views to maintain clear visual hierarchy.

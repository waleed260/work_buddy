## 2026-05-13 - Visual Hierarchy in Text-based Interfaces
**Learning:** In a CLI or chat-based interface, users can easily lose track of importance when tasks are presented in a simple list. Using consistent color-coded emojis (🔴, 🟡, 🔵) combined with priority sorting creates an immediate visual hierarchy that helps users focus on what matters most first.
**Action:** Always implement sorting by priority for task/event lists and use consistent visual markers for severity or priority levels across all status-related outputs.

## 2026-05-17 - Scanning Efficiency and Encouraging Empty States
**Learning:** Placing priority markers immediately after status indicators (e.g., `🔄 🔴 Title`) creates a tighter anchor for the eye, allowing users to scan both completion status and urgency in a single vertical pass. Additionally, using "delightful" empty states (emojis like 🥳, 🚀) transforms a "no results" dead-end into a moment of positive reinforcement, which is critical for a productivity-focused agent.
**Action:** When displaying lists, group meta-information (status, priority) before the content. Always enhance empty states with encouraging language and positive emojis.

## 2026-06-06 - Scannability of Multi-line List Items in CLI
**Learning:** When presenting list items with multiple properties (like Email or Slack drafts) in a CLI, using bold labels (e.g., **To:**) provides clear visual anchors. Furthermore, indenting subsequent lines of an item to align with the first line's text (ignoring the bullet) prevents the text from "leaking" back to the left margin, which maintains the vertical rhythm and makes it much easier to distinguish between different list items.
**Action:** Use bold Markdown for property labels and ensure multi-line content is indented by exactly two spaces to match the primary bullet's text alignment.

## 2026-07-28 - Information Parity Across CLI Task Views
**Learning:** In productivity interfaces, users expect consistency when transitioning tasks between different views (such as general task lists and daily standup summaries). Omitting priority levels or due dates in any view creates informational disparity, leading to friction and loss of context. Standardizing layout representations across all list-based outputs ensures seamless tracking and high scannability.
**Action:** Standardize task-line rendering logic in a single helper to enforce consistent presentation of status, priority, and due dates across all application modules.

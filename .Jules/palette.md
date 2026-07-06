## 2026-05-13 - Visual Hierarchy in Text-based Interfaces
**Learning:** In a CLI or chat-based interface, users can easily lose track of importance when tasks are presented in a simple list. Using consistent color-coded emojis (🔴, 🟡, 🔵) combined with priority sorting creates an immediate visual hierarchy that helps users focus on what matters most first.
**Action:** Always implement sorting by priority for task/event lists and use consistent visual markers for severity or priority levels across all status-related outputs.

## 2026-05-17 - Scanning Efficiency and Encouraging Empty States
**Learning:** Placing priority markers immediately after status indicators (e.g., `🔄 🔴 Title`) creates a tighter anchor for the eye, allowing users to scan both completion status and urgency in a single vertical pass. Additionally, using "delightful" empty states (emojis like 🥳, 🚀) transforms a "no results" dead-end into a moment of positive reinforcement, which is critical for a productivity-focused agent.
**Action:** When displaying lists, group meta-information (status, priority) before the content. Always enhance empty states with encouraging language and positive emojis.

## 2026-06-06 - Scannability of Multi-line List Items in CLI
**Learning:** When presenting list items with multiple properties (like Email or Slack drafts) in a CLI, using bold labels (e.g., **To:**) provides clear visual anchors. Furthermore, indenting subsequent lines of an item to align with the first line's text (ignoring the bullet) prevents the text from "leaking" back to the left margin, which maintains the vertical rhythm and makes it much easier to distinguish between different list items.
**Action:** Use bold Markdown for property labels and ensure multi-line content is indented by exactly two spaces to match the primary bullet's text alignment.

## 2026-07-07 - Informational Parity in State Transitions
**Learning:** Maintaining metadata (like priority emojis and due dates) when an item transitions from 'In Progress' to 'Completed' provides closure and preserves context regarding the original task's importance. Users find it more satisfying and informative to see exactly what high-priority items they've cleared in their daily summaries.
**Action:** Centralize formatting logic for items that appear in multiple states or views into a single helper function. This ensures identical visual presentation and informational parity across the entire interface.

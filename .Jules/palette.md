## 2026-05-13 - Visual Hierarchy in Text-based Interfaces
**Learning:** In a CLI or chat-based interface, users can easily lose track of importance when tasks are presented in a simple list. Using consistent color-coded emojis (🔴, 🟡, 🔵) combined with priority sorting creates an immediate visual hierarchy that helps users focus on what matters most first.
**Action:** Always implement sorting by priority for task/event lists and use consistent visual markers for severity or priority levels across all status-related outputs.

## 2026-05-17 - Scanning Efficiency and Encouraging Empty States
**Learning:** Placing priority markers immediately after status indicators (e.g., `🔄 🔴 Title`) creates a tighter anchor for the eye, allowing users to scan both completion status and urgency in a single vertical pass. Additionally, using "delightful" empty states (emojis like 🥳, 🚀) transforms a "no results" dead-end into a moment of positive reinforcement, which is critical for a productivity-focused agent.
**Action:** When displaying lists, group meta-information (status, priority) before the content. Always enhance empty states with encouraging language and positive emojis.

## 2026-06-06 - Scannability of Multi-line List Items in CLI
**Learning:** When presenting list items with multiple properties (like Email or Slack drafts) in a CLI, using bold labels (e.g., **To:**) provides clear visual anchors. Furthermore, indenting subsequent lines of an item to align with the first line's text (ignoring the bullet) prevents the text from "leaking" back to the left margin, which maintains the vertical rhythm and makes it much easier to distinguish between different list items.
**Action:** Use bold Markdown for property labels and ensure multi-line content is indented by exactly two spaces to match the primary bullet's text alignment.

## 2026-06-15 - Unifying Task Formatting and Tone Consistency
**Learning:** In text-based terminal utilities, small variations in message construction (such as varying checkmark placement, missing details in completed lists, or mismatched conversational tones) undermine the user's confidence in the system's precision. Centralizing output generation to a single formatting utility ensures visual symmetry across all workspace components (like general tasks and the daily standup), while adhering to a balanced reinforcement loop pattern creates a satisfying, cohesive user journey.
**Action:** Centralize text presentation and formatting structures to formatting helper functions, and enforce identical visual properties across distinct lists with informational parity. Ensure all tool confirmation messages append an appropriate reinforcement phrase.

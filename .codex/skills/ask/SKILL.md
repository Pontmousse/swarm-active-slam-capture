---
name: ask
description: Answer questions without modifying the codebase. Use when the user wants analysis, explanations, or guidance only, allowing read-only inspection commands for context but no edits, installs, service starts, formatting, or other state-changing actions.
---

# Answer Only

## Rules

- Answer the user’s question without editing, creating, deleting, formatting, or rewriting files.
- Do not run commands that modify the repository, install packages, start services, or change environment state.
- Use read-only commands only when needed for context, such as `rg`, `Get-Content`, `git status`, `git diff`, `git show`, and directory listings.
- Prefer answering from existing context before running commands.
- If the user asks for a code change, provide guidance or a patch suggestion in the response instead of applying it.
- Keep answers concise and focused.
- State whether the answer is based on inspection or inference when that distinction matters.
# Available Tools

This document describes the tools available to nanobot.

## File Operations

### read_file
Read the contents of a file.
- Parameters: `path` (str)
- Returns: file contents as string

### write_file
Write content to a file (creates parent directories if needed).
- Parameters: `path` (str), `content` (str)
- Returns: confirmation string

### edit_file
Edit a file by replacing specific text.
- Parameters: `path` (str), `old_text` (str), `new_text` (str)
- Returns: confirmation string

### list_dir
List contents of a directory.
- Parameters: `path` (str)
- Returns: directory listing as string

## Shell Execution

### exec
Execute a shell command and return output.
- Parameters: `command` (str), `working_dir` (str, optional)
- Returns: command output as string

Safety Notes:
- Commands have a configurable timeout (default 60s)
- Dangerous commands are blocked (rm -rf, format, dd, shutdown, etc.)
- Output is truncated at 10,000 characters
- Optional `restrictToWorkspace` config to limit paths

## Web Access

### web_fetch
Fetch and extract main content from a URL.
- Parameters: `url` (str), `extractMode` (str, default "markdown"), `maxChars` (int, default 50000)
- Returns: extracted page content as string

Notes:
- Content is extracted using readability
- Supports markdown or plain text extraction
- Output is truncated at 50,000 characters by default

## Communication

### message
Send a message to the user (used internally).
- Parameters: `content` (str), `channel` (str, optional), `chat_id` (str, optional)
- Returns: confirmation string

## Background Tasks

### spawn
Spawn a subagent to handle a task in the background.
- Parameters: `task` (str), `label` (str, optional)
- Returns: task ID string

Use for complex or time-consuming tasks that can run independently. The subagent will complete the task and report back when done.

## Scheduled Reminders (Cron)

Use the `exec` tool to manage scheduled reminders with `nanobot cron` commands.

### Set a recurring reminder
- Every day at 9am: `nanobot cron add --name "morning" --message "Good morning!" --cron "0 9 * * *"`
- Every 2 hours: `nanobot cron add --name "water" --message "Drink water!" --every 7200`

### Set a one-time reminder
- At a specific time: `nanobot cron add --name "meeting" --message "Meeting starts now!" --at "2025-01-31T15:00:00"`

### Manage reminders
- List all jobs: `nanobot cron list`
- Remove a job: `nanobot cron remove <job_id>`

## Heartbeat Task Management

The `HEARTBEAT.md` file in the workspace is checked every 30 minutes.
Use file operations (edit_file, write_file) to manage periodic tasks in that file.

---

## Adding Custom Tools

To add custom tools:
1. Create a class that extends `Tool` in `nanobot/agent/tools/`
2. Implement `name`, `description`, `parameters`, and `execute`
3. Register it in `AgentLoop._register_default_tools()`

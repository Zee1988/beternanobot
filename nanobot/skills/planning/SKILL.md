---
name: planning
description: Manus-style file-based planning for complex tasks. Creates task_plan.md, findings.md, and progress.md as persistent working memory. Use for multi-step tasks, research, or anything requiring >5 tool calls.
always: true
---

# Planning with Files

Use persistent markdown files as working memory on disk for complex tasks.

```
Context Window = RAM (volatile, limited)
Filesystem = Disk (persistent, unlimited)

→ Anything important gets written to disk.
```

## Directory Structure

Planning files are stored in `/tmp/nanobot/plans/` to keep workspace clean:

```
/tmp/nanobot/plans/
├── project-name/
│   ├── task_plan.md
│   ├── findings.md
│   └── progress.md
└── another-project/
    └── ...
```

This keeps your workspace directory clean from planning artifacts.

## Quick Start

Before ANY complex task:

1. **Create `task_plan.md`** — phases, progress checkboxes, decisions, errors
2. **Create `findings.md`** — requirements, research, technical decisions
3. **Create `progress.md`** — session log, test results, error log
4. **Re-read plan before decisions** — refreshes goals in attention window
5. **Update after each phase** — mark complete, log errors

Use `{baseDir}/templates/` as starting points for each file.

Or initialize all three at once:

```bash
bash {baseDir}/scripts/init-session.sh [project-name]
```

This creates files under `_plans/[project-name]/` automatically.

## Cleanup

After task completion, clean up planning files:

```bash
bash {baseDir}/scripts/cleanup-session.sh [project-name]
```

This removes the `_plans/[project-name]/` directory and all planning files.

## File Purposes

| File | Purpose | When to Update |
|------|---------|----------------|
| `task_plan.md` | Phases, progress, decisions | After each phase |
| `findings.md` | Research, discoveries | After ANY discovery |
| `progress.md` | Session log, test results | Throughout session |

## Critical Rules

### 1. Create Plan First
Never start a complex task without `task_plan.md`. Non-negotiable.

### 2. The 2-Action Rule
After every 2 search/browse/view operations, IMMEDIATELY save key findings to files.
This prevents information from being lost when context fills up.

### 3. Read Before Decide
Before major decisions, re-read the plan file. This keeps goals in your attention window.

### 4. Update After Act
After completing any phase:
- Mark phase status: `in_progress` → `complete`
- Log any errors encountered
- Note files created/modified

### 5. Log ALL Errors
Every error goes in the plan file. This builds knowledge and prevents repetition.

```markdown
## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| FileNotFoundError | 1 | Created default config |
| API timeout | 2 | Added retry logic |
```

### 6. Never Repeat Failures
Track what you tried. Mutate the approach on each retry.

## 3-Strike Error Protocol

```
ATTEMPT 1: Diagnose & Fix → Read error, identify root cause, apply targeted fix
ATTEMPT 2: Alternative Approach → Different method, different tool or library
ATTEMPT 3: Broader Rethink → Question assumptions, search for solutions
AFTER 3 FAILURES: Escalate to user with what you tried and the specific error
```

## Read vs Write Decision Matrix

| Situation | Action | Reason |
|-----------|--------|--------|
| Just wrote a file | DON'T read | Content still in context |
| Viewed image/PDF | Write findings NOW | Multimodal data lost on reset |
| Browser returned data | Write to file | Screenshots don't persist |
| Starting new phase | Read plan/findings | Re-orient if context stale |
| Error occurred | Read relevant file | Need current state to fix |
| Resuming after gap | Read all planning files | Recover full state |

## The 5-Question Reboot Test

If you can answer these, your context management is solid:

| Question | Answer Source |
|----------|---------------|
| Where am I? | Current phase in task_plan.md |
| Where am I going? | Remaining phases |
| What's the goal? | Goal statement in plan |
| What have I learned? | findings.md |
| What have I done? | progress.md |

## When to Use This Pattern

**Use for:**
- Multi-step tasks (3+ phases)
- Research tasks
- Building/creating projects
- Tasks requiring >5 tool calls
- Anything requiring organization

**Skip for:**
- Simple questions
- Single-file edits
- Quick lookups

## Anti-Patterns

| Don't | Do Instead |
|-------|------------|
| State goals once and forget | Re-read plan before decisions |
| Hide errors and retry silently | Log errors to plan file |
| Stuff everything in context | Store large content in files |
| Start executing immediately | Create plan file FIRST |
| Repeat failed actions | Track attempts, mutate approach |

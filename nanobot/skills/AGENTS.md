<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-11 | Updated: 2026-02-11 -->

# skills

## Purpose

Skill definitions for extending agent capabilities. Each skill is a directory containing a `SKILL.md` file with instructions, triggers, and optionally scripts. Skills are loaded by the agent's context builder.

## Key Files

| File | Description |
|------|-------------|
| `README.md` | Documentation on creating skills |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `cron/` | Scheduled task management skill |
| `github/` | GitHub integration skill |
| `skill-creator/` | Meta-skill for creating new skills |
| `summarize/` | Text summarization skill |
| `tmux/` | Terminal multiplexer skill |
| `weather/` | Weather information skill |

## For AI Agents

### Working In This Directory

- Each skill is a directory with a `SKILL.md` file
- Skills are loaded into agent context automatically
- Skills can include helper scripts in subdirectories
- Keep skills focused and well-documented

### Skill Structure

```
myskill/
├── SKILL.md      # Required: skill definition
└── scripts/      # Optional: helper scripts
    └── helper.sh
```

### SKILL.md Format

```markdown
# Skill Name

## Description
What this skill does.

## Triggers
When to activate this skill.

## Instructions
How to use the skill.
```

### Testing Requirements

- Test skill loading and parsing
- Test trigger detection
- Test script execution if applicable

### Common Patterns

- Markdown format for easy editing
- Clear trigger keywords
- Step-by-step instructions
- Example usage included

## Dependencies

### Internal

- Loaded by `nanobot.agent.skills`
- Used by `nanobot.agent.context`

### External

- None (pure markdown files)

<!-- MANUAL: -->

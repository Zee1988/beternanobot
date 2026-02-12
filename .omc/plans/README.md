# Subagent V2 Implementation Plans

## Files

- **subagent-v2-final.md** - Consolidated implementation plan (1,754 lines)
  - Complete source code for all files (new and modified)
  - All test code
  - Verification commands
  - Backward compatibility notes

## Quick Reference

### New Files (2)
1. `nanobot/agent/subagent_types.py` - Status enum, Entry, ResultQueue
2. `nanobot/agent/subagent_registry.py` - Bounded registry with async persistence

### Modified Files (5)
1. `nanobot/config/schema.py` - Add SubagentConfig
2. `nanobot/agent/subagent.py` - Complete rewrite with all fixes
3. `nanobot/agent/tools/spawn.py` - Remove mutable state
4. `nanobot/agent/loop.py` - Inject origin params, manage sweeper
5. `nanobot/cli/commands.py` - Pass subagent_config

### Test Files (4)
1. `tests/test_subagent_types.py`
2. `tests/test_subagent_registry.py`
3. `tests/test_subagent_improved.py`
4. `tests/test_sweeper.py`

## Problems Solved

- **C1-C3 (CRITICAL)**: Async persistence, single timeout source, bounded registry
- **H1-H4 (HIGH)**: Crash recovery, message loss prevention, concurrency safety, validation
- **M2-M5 (MEDIUM)**: Async sweeper, simplified nesting, enum status

## Implementation

See `subagent-v2-final.md` for complete implementation details.

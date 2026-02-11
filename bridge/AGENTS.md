<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-11 | Updated: 2026-02-11 -->

# bridge

## Purpose

A TypeScript/Node.js bridge that enables WhatsApp integration for nanobot. Since WhatsApp's web protocol requires JavaScript, this bridge runs as a separate WebSocket server that the Python nanobot connects to for sending and receiving WhatsApp messages.

## Key Files

| File | Description |
|------|-------------|
| `package.json` | Node.js dependencies and scripts |
| `tsconfig.json` | TypeScript compiler configuration |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `src/` | TypeScript source files (see `src/AGENTS.md`) |

## For AI Agents

### Working In This Directory

- This is a TypeScript project, not Python
- Install with `npm install` or `yarn`
- Build with TypeScript compiler
- Runs as a WebSocket server on port 3001 by default

### Testing Requirements

- No dedicated test suite currently
- Test manually by connecting the Python WhatsApp channel

### Common Patterns

- WebSocket server for bidirectional communication
- Event-based messaging between Node.js and Python
- whatsapp-web.js library for WhatsApp protocol

## Dependencies

### Internal

- Used by `nanobot/channels/whatsapp.py` for WhatsApp integration

### External

| Package | Purpose |
|---------|---------|
| `ws` | WebSocket server implementation |
| `whatsapp-web.js` | WhatsApp Web protocol implementation |

<!-- MANUAL: -->

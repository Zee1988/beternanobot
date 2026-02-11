<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-11 | Updated: 2026-02-11 -->

# src

## Purpose

TypeScript source files for the WhatsApp bridge. Contains the WebSocket server that connects Python nanobot to WhatsApp Web.

## Key Files

| File | Description |
|------|-------------|
| `index.ts` | Entry point, starts the bridge server |
| `server.ts` | `BridgeServer` WebSocket server implementation |
| `whatsapp.ts` | `WhatsAppClient` wrapper around whatsapp-web.js |
| `types.d.ts` | TypeScript type definitions |

## For AI Agents

### Working In This Directory

- This is TypeScript code compiled to JavaScript
- `BridgeServer` handles Python ↔ Node.js communication
- `WhatsAppClient` handles Node.js ↔ WhatsApp communication
- Messages flow: Python → WebSocket → Node.js → WhatsApp

### Message Protocol

```typescript
// Inbound from WhatsApp to Python
{ type: 'message', from: '...', text: '...', ... }
{ type: 'qr', qr: '...' }  // QR code for auth
{ type: 'status', status: '...' }

// Outbound from Python to WhatsApp
{ type: 'send', to: '...', text: '...' }
```

### Testing Requirements

- Test WebSocket message handling
- Mock whatsapp-web.js for unit tests
- Test reconnection logic

### Common Patterns

- Event-driven architecture
- Broadcast to all connected clients
- Error handling with JSON error responses
- Session persistence in auth directory

## Dependencies

### Internal

- Used by `nanobot/channels/whatsapp.py`

### External

| Package | Purpose |
|---------|---------|
| `ws` | WebSocket server |
| `whatsapp-web.js` | WhatsApp protocol |

<!-- MANUAL: -->

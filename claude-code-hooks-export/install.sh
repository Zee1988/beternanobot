#!/bin/bash
# Claude Code Hooks Installer
# Run this on a new machine to install the hooks

set -euo pipefail

CLAUDE_DIR="$HOME/.claude"

echo "Installing Claude Code Hooks..."

# Create directories
mkdir -p "$CLAUDE_DIR/scripts"
mkdir -p "$CLAUDE_DIR/hooks"
mkdir -p "$CLAUDE_DIR/data/claude-code-results"

# Copy scripts
echo "Copying scripts..."
cp scripts/* "$CLAUDE_DIR/scripts/"
chmod +x "$CLAUDE_DIR/scripts/*"

# Copy hook
echo "Copying hook..."
cp hooks/* "$CLAUDE_DIR/hooks/"
chmod +x "$CLAUDE_DIR/hooks/*"

# Update settings.json
echo "Updating settings.json..."
SETTINGS_FILE="$CLAUDE_DIR/settings.json"

if [ -f "$SETTINGS_FILE" ]; then
    # Backup
    cp "$SETTINGS_FILE" "${SETTINGS_FILE}.bak"

    # Check if jq is available
    if command -v jq &>/dev/null; then
        # Merge hooks config
        jq -s '.[0] * .[1]' "$SETTINGS_FILE" settings-hooks.json > "${SETTINGS_FILE}.tmp"
        mv "${SETTINGS_FILE}.tmp" "$SETTINGS_FILE"
        echo "Hooks added to settings.json"
    else
        echo "jq not found. Please manually add hooks from settings-hooks.json"
    fi
else
    cp settings-hooks.json "$SETTINGS_FILE"
    echo "Created settings.json with hooks"
fi

# Install nanobot skill (optional)
if [ -d "$HOME/.nanobot/workspace/skills" ]; then
    echo "Installing nanobot skill..."
    mkdir -p "$HOME/.nanobot/workspace/skills/claude-code"
    cp -r nanobot-skill/* "$HOME/.nanobot/workspace/skills/claude-code/"
    chmod +x "$HOME/.nanobot/workspace/skills/claude-code/scripts/"*
fi

echo ""
echo "Installation complete!"
echo "Please restart Claude Code to apply changes."
echo ""
echo "Usage:"
echo "  $CLAUDE_DIR/scripts/dispatch-claude-code.sh -p 'Your prompt' -n task-name"
echo ""
echo "Results:"
echo "  $CLAUDE_DIR/data/claude-code-results/latest.json"

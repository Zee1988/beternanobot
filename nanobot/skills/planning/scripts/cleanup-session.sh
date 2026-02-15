#!/bin/bash
# Clean up planning files for a project
# Usage: ./cleanup-session.sh [project-name]

PROJECT_NAME="${1:-project}"

# Use /tmp for planning files
PLANS_DIR="/tmp/nanobot/plans/$PROJECT_NAME"

if [ -d "$PLANS_DIR" ]; then
    echo "Cleaning up planning files for: $PROJECT_NAME"
    rm -rf "$PLANS_DIR"
    echo "Removed: $PLANS_DIR"

    # Remove parent directory if empty
    if [ -d "_plans" ] && [ -z "$(ls -A _plans)" ]; then
        rmdir "_plans"
        echo "Removed empty _plans directory"
    fi
else
    echo "No planning files found for: $PROJECT_NAME"
fi

echo ""
echo "Cleanup complete!"

#!/bin/bash

# Package Claude sessions from the current repo into a tar.gz archive

set -e

REPO_ROOT="$(git rev-parse --show-toplevel)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${1:-$REPO_ROOT/dist}"
OUTPUT_FILE="$OUTPUT_DIR/claude-sessions-$TIMESTAMP.tar.gz"

# Sanitize repo path: replace / and _ with -
SANITIZED=$(echo "$REPO_ROOT" | sed 's|/|-|g; s|_|-|g')
CLAUDE_DIR="$HOME/.claude/projects/$SANITIZED"

if [ ! -d "$CLAUDE_DIR" ]; then
    echo "No Claude sessions found at $CLAUDE_DIR"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Packaging Claude sessions from $CLAUDE_DIR..."
tar -czvf "$OUTPUT_FILE" -C "$HOME/.claude/projects" "./$SANITIZED"

echo "Created: $OUTPUT_FILE"

#!/bin/bash

# Package Cursor session databases for the current repo into gzipped copies.
#
# Cursor stores conversation history in SQLite databases (state.vscdb):
#   - Per-workspace: <CURSOR_BASE>/workspaceStorage/<hash>/state.vscdb
#     (workspace.json in each dir maps to the project folder)
#   - Global (composer/agent mode): <CURSOR_BASE>/globalStorage/state.vscdb
#
# IMPORTANT: We extract only conversation data (composerData + bubbleId rows)
# and exclude large blob entries (agentKv:blob:*, checkpointId:*, etc.) which
# can be >5GB. This reduces the packaged size from ~1GB+ to ~10-50MB.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${1:-$REPO_ROOT/dist}"

# Detect platform
case "$(uname -s)" in
    Darwin)
        CURSOR_BASE="${CURSOR_BASE:-$HOME/Library/Application Support/Cursor/User}"
        ;;
    Linux)
        CURSOR_BASE="${CURSOR_BASE:-$HOME/.config/Cursor/User}"
        ;;
    *)
        echo "Unsupported platform: $(uname -s)"
        exit 1
        ;;
esac

if [ ! -d "$CURSOR_BASE" ]; then
    echo "Cursor data directory not found at $CURSOR_BASE"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Extract only conversation rows for sessions that reference a specific repo path.
# Finds composerIds whose bubbles mention the repo path, then extracts only those
# sessions' composerData + bubbleId rows. This avoids packaging data from unrelated projects.
#
# Uses Python (sqlite3 module) since it's more reliably available than sqlite3 CLI.
extract_conversations() {
    local src_db="$1"
    local dest_db="$2"
    local repo_path="$3"

    if command -v python3 &>/dev/null; then
        python3 - "$src_db" "$dest_db" "$repo_path" <<'PYEOF'
import sqlite3, sys, json, re

src, dest, repo_path = sys.argv[1], sys.argv[2], sys.argv[3]
src_conn = sqlite3.connect(src)
dest_conn = sqlite3.connect(dest)
dest_conn.execute("CREATE TABLE IF NOT EXISTS cursorDiskKV(key TEXT, value BLOB)")

# Step 1: Find composer IDs whose bubbles reference the repo path.
# We search bubble values for the repo path string — this catches tool call args,
# file mentions, workspace URIs, etc.
print(f"Scanning for sessions referencing: {repo_path}")
cursor = src_conn.execute(
    "SELECT DISTINCT substr(key, 10, 36) FROM cursorDiskKV "
    "WHERE key LIKE 'bubbleId:%' AND cast(value as text) LIKE ?",
    (f"%{repo_path}%",),
)
composer_ids = set(row[0] for row in cursor)

# Also check composerData values directly (some sessions reference repo in context)
cursor = src_conn.execute(
    "SELECT DISTINCT substr(key, 15) FROM cursorDiskKV "
    "WHERE key LIKE 'composerData:%' AND cast(value as text) LIKE ?",
    (f"%{repo_path}%",),
)
composer_ids.update(row[0] for row in cursor)

if not composer_ids:
    print("No sessions found referencing this repo")
    dest_conn.close()
    src_conn.close()
    sys.exit(0)

print(f"Found {len(composer_ids)} sessions referencing this repo")

# Step 2: Extract composerData and bubbleId rows for matching sessions only
count = 0
for cid in composer_ids:
    # composerData
    row = src_conn.execute(
        "SELECT key, value FROM cursorDiskKV WHERE key = ?",
        (f"composerData:{cid}",),
    ).fetchone()
    if row:
        dest_conn.execute("INSERT INTO cursorDiskKV VALUES (?, ?)", row)
        count += 1

    # All bubbles for this session
    cursor = src_conn.execute(
        "SELECT key, value FROM cursorDiskKV WHERE key LIKE ?",
        (f"bubbleId:{cid}:%",),
    )
    batch = cursor.fetchall()
    if batch:
        dest_conn.executemany("INSERT INTO cursorDiskKV VALUES (?, ?)", batch)
        count += len(batch)

dest_conn.commit()
dest_conn.close()
src_conn.close()
print(f"Extracted {count} rows for {len(composer_ids)} sessions")
PYEOF
    elif command -v sqlite3 &>/dev/null; then
        # sqlite3 CLI fallback — less precise filtering (copies all conversations)
        echo "Warning: python3 not found, falling back to sqlite3 CLI (may include unrelated sessions)"
        sqlite3 "$src_db" <<SQL
ATTACH '$dest_db' AS out;
CREATE TABLE out.cursorDiskKV(key TEXT, value BLOB);
INSERT INTO out.cursorDiskKV
  SELECT key, value FROM cursorDiskKV
  WHERE key LIKE 'composerData:%'
     OR key LIKE 'bubbleId:%';
SQL
    else
        echo "Error: neither python3 nor sqlite3 found, cannot package Cursor sessions"
        return 1
    fi
}

# Check if a vscdb contains rows referencing the repo path.
# Returns 0 (true) if matches found, 1 (false) otherwise.
db_has_repo_refs() {
    local db_path="$1"
    local repo_path="$2"

    if command -v python3 &>/dev/null; then
        python3 - "$db_path" "$repo_path" <<'PYEOF'
import sqlite3, sys
db, repo = sys.argv[1], sys.argv[2]
conn = sqlite3.connect(db)
row = conn.execute(
    "SELECT 1 FROM cursorDiskKV WHERE cast(value as text) LIKE ? LIMIT 1",
    (f"%{repo}%",),
).fetchone()
conn.close()
sys.exit(0 if row else 1)
PYEOF
    elif command -v sqlite3 &>/dev/null; then
        local count
        count="$(sqlite3 "$db_path" "SELECT COUNT(*) FROM cursorDiskKV WHERE cast(value as text) LIKE '%${repo_path}%' LIMIT 1;" 2>/dev/null || echo 0)"
        [ "$count" -gt 0 ]
    else
        # Can't check — assume it has relevant data
        return 0
    fi
}

packaged=0

# Find workspace state.vscdb matching this repo
WORKSPACE_DIR="$CURSOR_BASE/workspaceStorage"
if [ -d "$WORKSPACE_DIR" ]; then
    for ws in "$WORKSPACE_DIR"/*/workspace.json; do
        [ -f "$ws" ] || continue
        ws_dir="$(dirname "$ws")"
        # workspace.json contains {"folder": "file:///path/to/project"}
        # Extract the folder URI and strip the file:// prefix
        folder="$(grep -o '"folder":"[^"]*"' "$ws" 2>/dev/null | head -1 | sed 's/"folder":"//; s/"$//; s|^file://||')" || true
        if [ "$folder" = "$REPO_ROOT" ] && [ -f "$ws_dir/state.vscdb" ]; then
            echo "Found matching workspace: $ws_dir"
            DEST_DB="$OUTPUT_DIR/cursor-workspace-sessions-$TIMESTAMP.vscdb"
            extract_conversations "$ws_dir/state.vscdb" "$DEST_DB" "$REPO_ROOT"
            gzip -f "$DEST_DB"
            echo "Created: ${DEST_DB}.gz"
            packaged=1
            break
        fi
    done
fi

# Also check globalStorage state.vscdb (composer/agent data in newer Cursor).
# Only package if it contains sessions referencing this repo.
GLOBAL_DB="$CURSOR_BASE/globalStorage/state.vscdb"
if [ -f "$GLOBAL_DB" ]; then
    if db_has_repo_refs "$GLOBAL_DB" "$REPO_ROOT"; then
        echo "Packaging global Cursor state (filtering conversation data only)..."
        DEST_DB="$OUTPUT_DIR/cursor-global-sessions-$TIMESTAMP.vscdb"
        extract_conversations "$GLOBAL_DB" "$DEST_DB" "$REPO_ROOT"
        gzip -f "$DEST_DB"
        echo "Created: ${DEST_DB}.gz"
        packaged=1
    else
        echo "Skipping global Cursor state (no sessions found for this repo)"
    fi
fi

if [ "$packaged" -eq 0 ]; then
    echo "No Cursor AI sessions found for repo $REPO_ROOT (skipping)"
    exit 0
fi

echo "Done packaging Cursor sessions."

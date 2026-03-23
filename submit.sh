#!/usr/bin/env bash
set -euo pipefail

# submit.sh - Submit your take-home solution
# This script was generated specifically for your assignment.

SERVICE_URL="https://take-home-service.lumalabs-ext.workers.dev"
TOKEN="e617afe8-dbdb-4e8f-9dd1-117a4fe6595b"
REVIEW_USER="luma-take-home-review-bot"

echo "============================================"
echo "  Submitting Take-Home Solution"
echo "============================================"
echo ""

# 1. Ensure we're in a git repo
if ! git rev-parse --is-inside-work-tree &>/dev/null; then
    echo "Error: Not inside a git repository."
    echo "Please run this from your solution's repo root."
    exit 1
fi

# 2. Package AI session history (non-fatal)
echo "--- Packaging AI session history ---"
if [[ -x "./dist/package-all-sessions.sh" ]]; then
    ./dist/package-all-sessions.sh || echo "Warning: session packaging had errors (continuing anyway)"
    # Stage any packaged sessions
    git add session-packages/ 2>/dev/null || true
fi
echo ""

# 3. Commit and push any remaining changes
echo "--- Pushing latest changes ---"
git add -A
git diff --cached --quiet || git commit -m "Final submission"
git push origin HEAD
echo ""

# 4. Get repo URL
REPO_URL=$(git remote get-url origin | sed 's/\.git$//' | sed 's|^git@github.com:|https://github.com/|')
BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "Repo: $REPO_URL"
echo "Branch: $BRANCH"
echo ""

# 5. Add review bot as collaborator
echo "--- Adding review bot collaborator ---"
REPO_PATH=$(echo "$REPO_URL" | sed 's|https://github.com/||')
if command -v gh &>/dev/null; then
    gh api "repos/${REPO_PATH}/collaborators/${REVIEW_USER}" -X PUT -f permission=read 2>/dev/null \
        && echo "Added ${REVIEW_USER} as collaborator" \
        || echo "Warning: Could not add collaborator (you may need to add manually)"
else
    echo "Warning: gh CLI not found. Please manually add '${REVIEW_USER}' as a read-only collaborator."
fi
echo ""

# 6. Notify the service
echo "--- Registering submission ---"
if curl -sf -X POST "${SERVICE_URL}/submit/${TOKEN}" \
    -H "Content-Type: application/json" \
    -d "{\"repo_url\": \"${REPO_URL}\", \"branch\": \"${BRANCH}\"}"; then
    echo ""
    echo "============================================"
    echo "  Submission registered successfully!"
    echo "============================================"
    echo ""
    echo "Your solution has been submitted for review."
    echo "You'll hear from us soon. Good luck!"
else
    echo ""
    echo "Error: Failed to register submission."
    echo "Please email your repo URL to your contact at Luma."
    exit 1
fi

#!/usr/bin/env bash
set -e

if [ $# -eq 0 ]; then
  echo "Usage: ./push.sh \"commit message\""
  exit 1
fi

COMMIT_MSG="$*"

echo "=== Staging all changes ==="
git add .
echo ""

if git diff --cached --quiet; then
  echo "No staged changes to commit."
  exit 0
fi

echo "=== Committing with message: $COMMIT_MSG ==="
git commit -m "$COMMIT_MSG"
echo ""

echo "=== Pushing to remote ==="
git push
echo ""

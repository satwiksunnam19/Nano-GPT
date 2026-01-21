#!/bin/bash
# Full push script: commits changes, removes >100 MB files from history,
# keeps backup branch, re-adds remote, force pushes clean repo.
# Usage: ./git_push_clean.sh /path/to/project "Commit message" <remote_url>

PROJECT_PATH="$1"
COMMIT_MESSAGE="$2"
REMOTE_URL="$3"   # full HTTPS URL with token or SSH URL
BRANCH_NAME="${4:-main}"

if [ -z "$PROJECT_PATH" ] || [ -z "$COMMIT_MESSAGE" ] || [ -z "$REMOTE_URL" ]; then
  echo "❌ Usage: $0 /path/to/project \"Commit message\" <remote_url> [branch]"
  echo "Example: $0 ~/myrepo \"My commit\" https://user:TOKEN@github.com/user/repo.git main"
  exit 1
fi

cd "$PROJECT_PATH" || { echo "❌ Directory not found: $PROJECT_PATH"; exit 1; }

# --- Basic Git Setup ---
if [ ! -d ".git" ]; then
  echo "🧰 Initializing Git repository..."
  git init
fi

if ! git remote | grep origin >/dev/null; then
  echo "🔗 Adding remote origin..."
  git remote add origin "$REMOTE_URL"
else
  git remote set-url origin "$REMOTE_URL"
fi

# Always sync all changes (adds, modifies, deletes)
git add -A

if git diff --cached --quiet && git diff --quiet; then
  echo "ℹ️ No new changes to commit."
else
  git commit -m "$COMMIT_MESSAGE"
fi

git branch -M "$BRANCH_NAME" 2>/dev/null

# --- Ensure git-filter-repo ---
if ! command -v git-filter-repo >/dev/null; then
  echo "Installing git-filter-repo (requires brew)…"
  brew install git-filter-repo || { echo "❌ git-filter-repo install failed"; exit 1; }
fi

echo "🔍 Scanning entire Git history for blobs >100MB…"
git rev-list --objects --all | \
  git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | \
  awk '$3 > 100000000 {print $4}' > /tmp/large_files.txt

if [ -s /tmp/large_files.txt ]; then
  echo "⚠️  Large files found and will be removed from all commits:"
  cat /tmp/large_files.txt

  # Create backup branch before rewriting
  BACKUP_BRANCH="backup-before-cleanup-$(date +%Y%m%d%H%M%S)"
  echo "🔒 Creating backup branch $BACKUP_BRANCH"
  git branch "$BACKUP_BRANCH"

  while read -r filepath; do
    echo "🗑️  Removing $filepath from all commits…"
    git filter-repo --path "$filepath" --invert-paths --force
  done < /tmp/large_files.txt

  # Re-add origin automatically after filter-repo
  if ! git remote | grep origin >/dev/null; then
    echo "🔗 Re-adding origin remote…"
    git remote add origin "$REMOTE_URL"
  fi

  git checkout "$BRANCH_NAME" || echo "⚠️ Could not checkout $BRANCH_NAME, staying on current branch"
fi

# --- Push Clean Repo ---
echo "📤 Force pushing cleaned $BRANCH_NAME to remote…"
git push origin "$BRANCH_NAME" --force

echo "✅ Push complete. Clean history on GitHub."
echo "ℹ️  Original history preserved locally in branch: $BACKUP_BRANCH (if created)."

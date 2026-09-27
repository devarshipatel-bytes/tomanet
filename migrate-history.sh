#!/usr/bin/env bash
# Rewrites all commit authors/committers to a new identity, then pushes the
# full history (all branches + tags) to a new, empty remote repo.
#
# Runs against a MIRROR CLONE in a temp dir — your current working directory
# (including its untracked files) is never touched.
set -euo pipefail

# === FILL THESE IN ===
NEW_NAME="Your Name"
NEW_EMAIL="you@example.com"
NEW_REPO_URL="https://github.com/your-account/new-repo.git"   # or SSH: git@github.com:account/repo.git
GITHUB_TOKEN=""   # only needed for HTTPS push auth; leave blank for SSH or if you use a credential helper
# ======================

if ! command -v git-filter-repo >/dev/null 2>&1; then
  echo "git-filter-repo is not installed. Install it with one of:"
  echo "  pip install git-filter-repo"
  echo "  sudo apt install git-filter-repo"
  exit 1
fi

if [[ "$NEW_NAME" == "Your Name" || "$NEW_EMAIL" == "you@example.com" || "$NEW_REPO_URL" == *"your-account/new-repo"* ]]; then
  echo "Edit the placeholders at the top of this script first (NEW_NAME, NEW_EMAIL, NEW_REPO_URL)."
  exit 1
fi

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(mktemp -d)"
MIRROR_DIR="$WORK_DIR/mirror.git"

echo "Mirroring $SRC_DIR into $MIRROR_DIR ..."
git clone --mirror "$SRC_DIR" "$MIRROR_DIR"
cd "$MIRROR_DIR"

echo "Rewriting every commit's author/committer to: $NEW_NAME <$NEW_EMAIL> ..."
export NEW_NAME NEW_EMAIL
git filter-repo --force --commit-callback '
import os
name = os.environ["NEW_NAME"].encode()
email = os.environ["NEW_EMAIL"].encode()
commit.author_name = name
commit.author_email = email
commit.committer_name = name
commit.committer_email = email
'

echo "Pushing rewritten history to $NEW_REPO_URL ..."
PUSH_URL="$NEW_REPO_URL"
if [[ -n "$GITHUB_TOKEN" && "$NEW_REPO_URL" == https://* ]]; then
  PUSH_URL="$(echo "$NEW_REPO_URL" | sed -E "s#https://#https://${GITHUB_TOKEN}@#")"
fi
git push --mirror "$PUSH_URL"

echo
echo "Done. All branches/tags pushed to the new repo with rewritten authorship."
echo "Temp mirror left at: $MIRROR_DIR (delete manually once you've verified the new repo)."
echo
echo "Your current working directory ($SRC_DIR) was NOT modified — its commit"
echo "hashes are now different from what's on the new remote. To keep working"
echo "from here, the simplest path is to re-clone the new repo fresh:"
echo "  git clone $NEW_REPO_URL <new-folder>"
echo "and copy your untracked files (figures/, *.tex, *.pdf, *.pptx, setup-git-creds.sh) over."

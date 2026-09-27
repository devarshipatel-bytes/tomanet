#!/usr/bin/env bash
# Repo-local git identity + credentials. Run once: ./setup-git-creds.sh
set -euo pipefail
cd "$(dirname "$0")"

read -rp "Name: " name
read -rp "Email: " email
read -rp "GitHub username: " ghuser
read -rsp "GitHub token (PAT): " token; echo

git config --local user.name "$name"
git config --local user.email "$email"
git config --local credential.helper "store --file=$(pwd)/.git/credentials"

printf 'https://%s:%s@github.com\n' "$ghuser" "$token" > .git/credentials
chmod 600 .git/credentials

echo "Done. Identity + creds are local to this repo only (.git/credentials, not committed)."

#!/usr/bin/env bash
# Fails if git history contains secrets, personal audio/enrollments, or large binaries.
set -euo pipefail
cd "$(dirname "$0")/.."
fail=0
if command -v gitleaks >/dev/null; then gitleaks detect --no-banner --redact -v --log-opts="--all" || fail=1; else echo "gitleaks not installed (CI runs it)"; fi
# personal data / credentials ever committed?
if git log --all --name-only --pretty=format: | grep -E '^(data/personal/|\.env$|.*\.env$|study/data/)' | grep -v README | sort -u | grep .; then echo "personal/credential path in history"; fail=1; fi
# any audio outside allow-listed synthetic sample dirs?
if git log --all --name-only --pretty=format: | grep -Ei '\.(wav|flac|mp3|m4a|ogg)$' | grep -vE '^results/.*/audio/synthetic_' | sort -u | grep .; then echo "non-synthetic audio in history"; fail=1; fi
# blobs > 5 MB
big=$(git rev-list --objects --all | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | awk '$1=="blob" && $3>5000000 {print $4" "$3}')
if [ -n "$big" ]; then echo "large blobs:"; echo "$big"; fail=1; fi
[ $fail -eq 0 ] && echo "scan clean" || { echo "scan FAILED"; exit 1; }

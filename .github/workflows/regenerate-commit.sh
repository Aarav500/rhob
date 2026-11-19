#!/usr/bin/env bash
# Commit regenerated artifacts to $BRANCH and push, for .github/workflows/regenerate.yml.
#
# Env: BRANCH, RUN_URL, FILES (space-separated paths), WHAT (for the message), GITHUB_SHA.
#
# The artifacts were produced from $GITHUB_SHA. Before every push, after every rebase,
# the sources that produce them must still be what they were there; otherwise the commit
# would put the artifacts next to code that does not reproduce them. The push is never
# forced, so a push that lands between the check and the push is rejected and the next
# attempt pulls and checks again.
set -euo pipefail

git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"

# shellcheck disable=SC2086  # FILES is a list of paths without spaces
git add $FILES
if git diff --cached --quiet; then
  echo "nothing changed in: $FILES"
  exit 0
fi
git commit -q -F - <<EOF
Regenerate $WHAT on the CI lock

Produced by $RUN_URL at $GITHUB_SHA. Only the artifacts: the docs and
tests that quote them are updated in a reviewed follow-up.
EOF

for attempt in 1 2 3; do
  if git pull -q --rebase origin "$BRANCH"; then
    if ! git diff --exit-code --stat "$GITHUB_SHA" HEAD -- src scripts requirements pyproject.toml; then
      echo "sources on $BRANCH changed since $GITHUB_SHA; not committing these artifacts"
      exit 1
    fi
    if git push -q origin "HEAD:$BRANCH"; then
      exit 0
    fi
  fi
  sleep $((attempt * 15))
done
exit 1

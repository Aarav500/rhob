#!/usr/bin/env bash
# Commit regenerated artifacts to $BRANCH and push, for .github/workflows/regenerate.yml.
#
# Env: BRANCH, RUN_URL, FILES (space-separated paths), WHAT (for the message), GITHUB_SHA.
# Optional: SOURCE_RUN_URL and SOURCE_SHA, when the artifacts were drawn by an earlier run
# (`from_run`) and only merged by this one; the plan job has checked that the sources that
# drew them are this run's.
#
# The artifacts were produced from $GITHUB_SHA. Before every push, after every rebase,
# the sources that produce them must still be what they were there, or the commit would
# put the artifacts next to code that does not reproduce them; and the artifact files
# must still be exactly what this run wrote. The push is never
# forced, so a push that lands between the check and the push is rejected and the next
# attempt pulls and checks again.
set -euo pipefail

SOURCE_RUN_URL="${SOURCE_RUN_URL:-}"
SOURCE_SHA="${SOURCE_SHA:-}"

git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"

# shellcheck disable=SC2086  # FILES is a list of paths without spaces
git add $FILES
if git diff --cached --quiet; then
  echo "nothing changed in: $FILES"
  exit 0
fi
if [ -n "$SOURCE_RUN_URL" ] && [ "$SOURCE_RUN_URL" != "$RUN_URL" ]; then
  produced="Drawn by $SOURCE_RUN_URL at ${SOURCE_SHA:-$GITHUB_SHA}, merged by
$RUN_URL at $GITHUB_SHA (same src/, requirements/, pyproject.toml
and drawing scripts)."
else
  produced="Produced by $RUN_URL at $GITHUB_SHA."
fi
pins=""
for v in OPENBLAS_CORETYPE OPENBLAS_NUM_THREADS MKL_CBWR MKL_NUM_THREADS OMP_NUM_THREADS \
  ATEN_CPU_CAPABILITY; do
  if [ -n "${!v-}" ]; then pins="$pins $v=${!v}"; fi
done
git commit -q -F - <<EOF
Regenerate $WHAT on the CI lock

$produced${pins:+
Numerics pins:$pins.}
Only the artifacts: the docs and tests that quote them are updated in a
reviewed follow-up.
EOF
ours=$(git rev-parse HEAD)

for attempt in 1 2 3; do
  if git pull -q --rebase origin "$BRANCH"; then
    if ! git diff --exit-code --stat "$GITHUB_SHA" HEAD -- src scripts requirements pyproject.toml; then
      echo "sources on $BRANCH changed since $GITHUB_SHA; not committing these artifacts"
      exit 1
    fi
    # A rebase can fold someone else's edit to the same artifact into ours without a
    # conflict; what is pushed must be exactly what this run wrote.
    # shellcheck disable=SC2086
    if ! git diff --quiet "$ours" HEAD -- $FILES; then
      echo "the artifacts were edited on $BRANCH during the run; not committing"
      exit 1
    fi
    if git push -q origin "HEAD:$BRANCH"; then
      exit 0
    fi
  else
    # A conflicted rebase stays in progress and would make every later attempt fail for
    # that reason instead of the real one. Back out to our commit and try again; a
    # conflict that persists means a newer commit on the branch wrote the same files.
    git rebase --abort 2>/dev/null || true
    git reset -q --hard "$ours"
    echo "could not rebase onto origin/$BRANCH (attempt $attempt): another commit there touches $FILES?"
  fi
  sleep $((attempt * 15))
done
exit 1

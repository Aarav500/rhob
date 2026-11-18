#!/usr/bin/env bash
# Regenerate requirements/ci-admission-linux-py312.txt, the hashed lock that the nightly
# `admission-slow` job in .github/workflows/tests.yml installs. requirements/README.md says
# when to run this and what to check before committing the result.
#
#   scripts/update_ci_lock.sh            resolve against PyPI as it stands now
#   scripts/update_ci_lock.sh CUTOFF     resolve against PyPI as it stood at CUTOFF, an
#                                        RFC 3339 UTC time such as 2026-11-16T07:00:00Z
#   scripts/update_ci_lock.sh --check    re-resolve at the cutoff the committed lock records,
#                                        and exit 1 if any pin or hash would change
#
# Deterministic by construction: the resolution goes through `uv pip compile
# --exclude-newer CUTOFF`, which ignores every file uploaded after CUTOFF, and the cutoff is
# written into the header. Rerunning with the recorded cutoff reproduces the file, unless a
# pinned release has since been yanked or deleted from PyPI, or a different uv resolves
# differently (the header records which uv wrote it). User and project uv configuration is
# ignored, and the index, resolution strategy and platform are passed explicitly.
#
# Needs uv on PATH and network access to pypi.org. Writes nothing but the lock.

set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
lock="$repo/requirements/ci-admission-linux-py312.txt"

# The extras the admission-slow lock is built for. Keep in step with scripts/check_ci_lock.py
# and drift.yml's install line (tests/test_ci_lock.py checks both).
extras=(dev continuous mujoco pettingzoo)
python_version="3.12"
# The lock is resolved for glibc 2.28, so it also installs on older Linux, and then
# resolved again for glibc 2.40, the newest tag uv can target; ubuntu-latest is newer
# still. If the two disagree, some newest release ships only wheels the 2.28 target cannot
# use, so the lock would not be what pip resolves on the runner, and the script stops.
platform="x86_64-manylinux_2_28"
runner_platform="x86_64-manylinux_2_40"
index="https://pypi.org/simple"

die() { echo "update_ci_lock.sh: $*" >&2; exit 1; }

command -v uv >/dev/null || die "uv is not on PATH (https://docs.astral.sh/uv/)"

check=0
cutoff=""
case "${1:-}" in
  --check)
    check=1
    [ -f "$lock" ] || die "no lock at $lock to check"
    cutoff="$(sed -n 's/^# exclude-newer: *//p' "$lock" | head -n 1)"
    [ -n "$cutoff" ] || die "$lock records no '# exclude-newer:' line"
    ;;
  -h|--help)
    sed -n '2,19p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 0
    ;;
  "")
    cutoff="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    ;;
  *)
    cutoff="$1"
    ;;
esac
[[ "$cutoff" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$ ]] \
  || die "cutoff must look like 2026-11-16T07:00:00Z, got '$cutoff'"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

extra_args=()
for e in "${extras[@]}"; do extra_args+=(--extra "$e"); done

compile() {  # compile PLATFORM OUTFILE
  # env -u: an index, constraint or torch-backend setting in the caller's environment
  # would otherwise change the resolution without appearing in the command line.
  # --refresh: a locally cached index page older than the cutoff can lack files uploaded
  # before it, and would then write a lock that a later --check contradicts.
  (cd "$repo" && env -u UV_INDEX -u UV_INDEX_URL -u UV_DEFAULT_INDEX -u UV_EXTRA_INDEX_URL \
      -u UV_FIND_LINKS -u UV_CONSTRAINT -u UV_OVERRIDE -u UV_TORCH_BACKEND \
      -u UV_PRERELEASE -u UV_RESOLUTION -u UV_EXCLUDE_NEWER \
    uv pip compile pyproject.toml "${extra_args[@]}" \
      --python-version "$python_version" \
      --python-platform "$1" \
      --generate-hashes \
      --exclude-newer "$cutoff" \
      --default-index "$index" \
      --resolution highest \
      --no-config \
      --refresh \
      --no-header \
      --quiet \
      --output-file "$2")
}

pins() { grep -E '^[A-Za-z0-9]' "$1" | sed 's/ *\\$//'; }

compile "$platform" "$work/body.txt"
compile "$runner_platform" "$work/runner.txt"
if ! diff <(pins "$work/body.txt") <(pins "$work/runner.txt") > "$work/platform.diff"; then
  cat "$work/platform.diff" >&2
  die "the $platform and $runner_platform resolutions differ (< and > above). A newest \
release has no wheel for glibc 2.28, so a $platform lock would not be what CI's pip \
installs. Decide which to keep, and say why in requirements/README.md."
fi

if [ "$check" = 1 ]; then
  if diff -u <(grep -v '^#' "$lock") <(grep -v '^#' "$work/body.txt"); then
    echo "lock reproduces at its recorded cutoff $cutoff: $(pins "$lock" | wc -l) pins"
    exit 0
  fi
  die "the committed lock differs from a fresh resolution at its own cutoff $cutoff (diff \
above). pyproject.toml changed without a regeneration, the lock was edited by hand, or \
PyPI or uv changed."
fi

if git -C "$repo" diff --quiet HEAD -- pyproject.toml 2>/dev/null; then
  source_line="pyproject.toml as committed at $(git -C "$repo" rev-parse --short HEAD)"
else
  source_line="pyproject.toml with uncommitted changes on top of $(git -C "$repo" rev-parse --short HEAD 2>/dev/null || echo 'an unknown commit')"
fi
uv_version="$(uv --version 2>/dev/null | awk '{print $2}')"
extras_csv="$(IFS=,; echo "${extras[*]}")"

{
  cat <<EOF
# requirements/ci-admission-linux-py312.txt -- generated, do not edit by hand.
#
# WHAT THIS IS
# The fixed, hashed stack the nightly \`admission-slow\` job in .github/workflows/tests.yml
# installs, so that a new release reaches the physics screen only when this file is
# deliberately regenerated. (The CPython 3.12 patch release, pip and the runner image still
# move.) It is uv's resolution of
#     pip install -e ".[$extras_csv]"
# with every distribution that install pulls in, transitively, at one exact version, and
# the sha256 of every file PyPI holds for that version. torch is PyPI's Linux build, with
# the CUDA wheels it depends on. rhob itself is not listed: CI installs it after this file
# with \`pip install --no-deps -e .\`.
#
# FOR: CPython $python_version on Linux x86_64 only. Resolved for $platform, and
# identically for $runner_platform. Not for macOS, Windows, ARM or another Python.
# NOT: the environment of any committed artifact. Each artifact records its own versions in
# its provenance block, and requirements-lock.txt is a different, older file.
#
# HOW IT WAS MADE
# scripts/update_ci_lock.sh $cutoff, which ran
#     uv pip compile pyproject.toml $(printf -- '--extra %s ' "${extras[@]}")\\
#         --python-version $python_version --python-platform $platform \\
#         --generate-hashes --exclude-newer $cutoff \\
#         --default-index $index --resolution highest --no-config --refresh
# exclude-newer: $cutoff
# uv: $uv_version
# from: $source_line
#
# Rerun \`scripts/update_ci_lock.sh $cutoff\` to reproduce it, \`--check\` to
# compare, and see requirements/README.md before regenerating.
#
EOF
  cat "$work/body.txt"
} > "$work/lock.txt"

mkdir -p "$(dirname "$lock")"
mv "$work/lock.txt" "$lock"
echo "wrote $lock: $(pins "$lock" | wc -l) pins, resolved as of $cutoff with uv $uv_version"

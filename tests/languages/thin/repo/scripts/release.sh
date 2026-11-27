#!/usr/bin/env bash
# Tags a release and uploads the archive.
set -euo pipefail

ARCHIVE_DIR="dist-archives"

check_clean() {
  if [ -n "$(git status --porcelain)" ]; then
    echo "working tree is not clean" >&2
    exit 1
  fi
}

function make_archive {
  local version="$1"
  mkdir -p "$ARCHIVE_DIR"
  git archive --format=tar.gz -o "$ARCHIVE_DIR/app-$version.tar.gz" "v$version"
}

check_clean
make_archive "$1"

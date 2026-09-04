#!/usr/bin/env bash
# Build and optionally deploy the unified GitHub Pages site to gh-pages branch only.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BUILD_ONLY=false
SKIP_BUILD=false
REMOTE="${REMOTE:-origin}"
BRANCH="${BRANCH:-gh-pages}"
GIT_AUTHOR_NAME="${GIT_AUTHOR_NAME:-pedro-loures}"
GIT_AUTHOR_EMAIL="${GIT_AUTHOR_EMAIL:-pedro-loures@users.noreply.github.com}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-only) BUILD_ONLY=true; shift ;;
    --skip-build) SKIP_BUILD=true; shift ;;
    --remote) REMOTE="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

if [[ "$SKIP_BUILD" != true ]]; then
  echo "Building Pages site..."
  tree-testing/.venv/bin/python tree-testing/scripts/build_pages_site.py \
    --output-dir "$ROOT/docs" || \
  python3 tree-testing/scripts/build_pages_site.py --output-dir "$ROOT/docs"
fi

if [[ ! -d "$ROOT/docs" ]]; then
  echo "Build failed: docs/ not found" >&2
  exit 1
fi

if [[ ! -d "$ROOT/docs/data" ]]; then
  echo "Warning: docs/data/ is empty — site shell only" >&2
fi

if [[ "$BUILD_ONLY" == true ]]; then
  echo "Build complete: $ROOT/docs"
  echo "Preview: cd docs && python3 -m http.server 8080"
  exit 0
fi

if ! git rev-parse --git-dir >/dev/null 2>&1; then
  echo "Not a git repository. Run with --build-only to preview locally." >&2
  exit 1
fi

REMOTE_URL="$(git remote get-url "$REMOTE")"
STAGING="$ROOT/.pages-deploy"
rm -rf "$STAGING"
mkdir -p "$STAGING"
rsync -a --delete "$ROOT/docs/" "$STAGING/"
touch "$STAGING/.nojekyll"

echo "Preparing $BRANCH deployment..."
cd "$STAGING"
git init -q
git checkout -q -b "$BRANCH"
git add -A
git -c user.name="$GIT_AUTHOR_NAME" -c user.email="$GIT_AUTHOR_EMAIL" \
  commit -m "chore: deploy GitHub Pages site"
git branch -M "$BRANCH"
git remote add origin "$REMOTE_URL"
git push -f origin "$BRANCH"
cd "$ROOT"
rm -rf "$STAGING"
git worktree prune 2>/dev/null || true
git branch -D gh-pages 2>/dev/null || true

echo "Deployed to $REMOTE/$BRANCH"
echo "Done. Enable Pages: Settings → Pages → branch $BRANCH / root"

#!/usr/bin/env bash
# Build and optionally deploy the unified GitHub Pages site to gh-pages branch only.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BUILD_ONLY=false
SKIP_BUILD=false
REMOTE="${REMOTE:-origin}"
BRANCH="${BRANCH:-gh-pages}"

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

# Safety: refuse to deploy if docs/data is missing entirely
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

WORKTREE="$ROOT/.pages-deploy"
rm -rf "$WORKTREE"
mkdir -p "$WORKTREE"

echo "Preparing orphan $BRANCH branch..."
git fetch "$REMOTE" "$BRANCH" 2>/dev/null || true

if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
  git worktree add --force "$WORKTREE" "$BRANCH"
else
  git worktree add --detach "$WORKTREE"
  cd "$WORKTREE"
  git checkout --orphan "$BRANCH"
  git rm -rf . 2>/dev/null || true
  cd "$ROOT"
fi

# Copy built site into worktree
rsync -a --delete "$ROOT/docs/" "$WORKTREE/"
touch "$WORKTREE/.nojekyll"

cd "$WORKTREE"
git add -A
if git diff --cached --quiet; then
  echo "No changes to deploy."
else
  git commit -m "chore: deploy GitHub Pages site"
  git push "$REMOTE" "$BRANCH" --force
  echo "Deployed to $REMOTE/$BRANCH"
fi

cd "$ROOT"
git worktree remove "$WORKTREE" --force 2>/dev/null || rm -rf "$WORKTREE"

echo "Done. Enable Pages: Settings → Pages → branch $BRANCH / root"

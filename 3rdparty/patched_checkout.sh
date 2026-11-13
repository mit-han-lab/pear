# shellcheck shell=bash
# Sourced, not run: a third-party project's checkout as its upstream commit with the project's patches applied, one
# commit each, and the patches back from the checkout's commits. The caller sets PROJECT (its directory here),
# UPSTREAM_URL, UPSTREAM_COMMIT, PATCHES_DIR, CHECKOUT_DIR and CHANGE_IN_PROGRESS.

# the patches' author and date: commits made in a checkout and exported carry no machine's identity and no time of
# their own; the date is the one of the mbox line git starts every patch with
PATCH_AUTHOR_NAME=pear
PATCH_AUTHOR_EMAIL=pear@localhost
PATCH_DATE="Mon, 17 Sep 2001 00:00:00 +0000"
# a patch file's name holds its whole subject
PATCH_NAME_MAX_LENGTH=140
# the upstream commit as the messages show it
SHORT_COMMIT=${UPSTREAM_COMMIT:0:8}

# Builds the checkout when it is missing, unless the argument is --check, then says in one line what the checkout is.
# Status 1 unless it is the upstream commit with the patches; with CHANGE_IN_PROGRESS=1 a checkout that differs passes.
checkout() {
  local mode=$1 difference patches
  if [ -n "$mode" ] && [ "$mode" != --check ]; then
    echo "unknown argument $mode (--check)"
    return 1
  fi

  if [ ! -e "$CHECKOUT_DIR" ]; then
    if [ "$mode" = --check ]; then
      echo "$PROJECT: $CHECKOUT_DIR is missing: bash $0 builds it"
      return 1
    fi
    build_checkout
  fi

  difference=$(checkout_difference)
  if [ -z "$difference" ]; then
    patches=("$PATCHES_DIR"/*.patch)
    echo "$PROJECT: $CHECKOUT_DIR = $SHORT_COMMIT + ${#patches[@]} patches (3rdparty/$PROJECT/patches)"
  elif [ "${CHANGE_IN_PROGRESS:-}" = 1 ]; then
    echo "$PROJECT: $CHECKOUT_DIR $difference: accepted as a change in progress"
  else
    echo "$PROJECT: $CHECKOUT_DIR $difference: it is not $SHORT_COMMIT + 3rdparty/$PROJECT/patches"
    return 1
  fi
}

# Built in a directory of its own and renamed into place: builders that start together each build one, the first
# rename wins and the others drop theirs. A build that fails leaves nothing behind.
build_checkout() {
  mkdir -p "$(dirname "$CHECKOUT_DIR")"
  PARTIAL_DIR=$(mktemp -d "$CHECKOUT_DIR.partial.XXXXXX")
  trap 'rm -rf "$PARTIAL_DIR"' EXIT

  git init -q "$PARTIAL_DIR"
  git -C "$PARTIAL_DIR" config user.name "$PATCH_AUTHOR_NAME"
  git -C "$PARTIAL_DIR" config user.email "$PATCH_AUTHOR_EMAIL"
  git -C "$PARTIAL_DIR" fetch -q --depth 1 "$UPSTREAM_URL" "$UPSTREAM_COMMIT"
  git -C "$PARTIAL_DIR" checkout -q -b pear "$UPSTREAM_COMMIT"
  # --keep-non-patch: the subjects keep their [component] tag
  git -C "$PARTIAL_DIR" am -q --keep-non-patch "$PATCHES_DIR"/*.patch

  # rename(2) fails on the checkout a faster builder put in place, and the exit removes this one (perl's rename: the
  # mv of macOS has no -T)
  perl -e 'rename($ARGV[0], $ARGV[1]) or exit 1' "$PARTIAL_DIR" "$CHECKOUT_DIR" || true
}

# What the checkout differs in from the upstream commit with the patches, in a few words; nothing when it is that.
checkout_difference() {
  # no checkout at all, or one of another history
  if ! git -C "$CHECKOUT_DIR" merge-base --is-ancestor "$UPSTREAM_COMMIT" HEAD 2> /dev/null; then
    echo "is not based on $SHORT_COMMIT"
  elif [ "$(patch_ids_of_patches)" != "$(patch_ids_of_checkout)" ]; then
    echo "has other commits than the patches"
  elif [ -n "$(git -C "$CHECKOUT_DIR" status --porcelain)" ]; then
    echo "has uncommitted or untracked files"
  fi
}

# one id per patch file, of its diff alone: messages and dates do not count
patch_ids_of_patches() {
  local patch
  for patch in "$PATCHES_DIR"/*.patch; do
    git patch-id --stable < "$patch" | cut -d ' ' -f 1
  done
}

patch_ids_of_checkout() {
  local commit
  for commit in $(git -C "$CHECKOUT_DIR" rev-list --reverse "$UPSTREAM_COMMIT..HEAD"); do
    git -C "$CHECKOUT_DIR" show "$commit" | git patch-id --stable | cut -d ' ' -f 1
  done
}

# Rewrites the patches from the checkout's commits since the upstream commit (git format-patch, one file per commit),
# once the new files are known to rebuild the checkout.
export_patches() {
  EXPORT_DIR=$(mktemp -d)
  CHECK_DIR=$CHECKOUT_DIR.check
  trap remove_export_directories EXIT

  require_exportable_checkout
  # full blob ids: the same files from any clone; no "i/N" in the subjects: a new patch leaves the other files alone
  git -C "$CHECKOUT_DIR" format-patch -q -N --no-signature --zero-commit --full-index \
    --filename-max-length="$PATCH_NAME_MAX_LENGTH" -o "$EXPORT_DIR" "$UPSTREAM_COMMIT..HEAD"
  # a file's first Date line is its header's
  PATCH_DATE=$PATCH_DATE perl -pi -e \
    'if (/^Date: / && !$dated{$ARGV}++) { $_ = "Date: $ENV{PATCH_DATE}\n" }' "$EXPORT_DIR"/*.patch
  require_rebuilt_checkout
  replace_patches
  ls "$PATCHES_DIR"
}

remove_export_directories() {
  rm -rf "$EXPORT_DIR"
  if [ -d "$CHECK_DIR" ]; then
    git -C "$CHECKOUT_DIR" worktree remove --force "$CHECK_DIR"
  fi
}

require_exportable_checkout() {
  local patch_author="$PATCH_AUTHOR_NAME <$PATCH_AUTHOR_EMAIL>" commit_authors commit author
  if ! git -C "$CHECKOUT_DIR" merge-base --is-ancestor "$UPSTREAM_COMMIT" HEAD; then
    echo "$CHECKOUT_DIR is not based on $UPSTREAM_COMMIT (UPSTREAM)"
    exit 1
  fi
  if [ -n "$(git -C "$CHECKOUT_DIR" status --porcelain)" ]; then
    echo "$CHECKOUT_DIR has uncommitted or untracked files: commit them, or remove them"
    exit 1
  fi

  # a patch carries its commit's author
  commit_authors=$(git -C "$CHECKOUT_DIR" log --format='%h %an <%ae>' "$UPSTREAM_COMMIT..HEAD")
  while read -r commit author; do
    if [ "$author" != "$patch_author" ]; then
      echo "commit $commit is by $author, not by $patch_author: git commit --amend --reset-author in the checkout"
      exit 1
    fi
  done <<< "$commit_authors"
}

# the exported files, applied to the upstream commit in a work tree of their own, give the checkout's tree
require_rebuilt_checkout() {
  git -C "$CHECKOUT_DIR" worktree add -q --detach "$CHECK_DIR" "$UPSTREAM_COMMIT"
  git -C "$CHECK_DIR" am -q --keep-non-patch "$EXPORT_DIR"/*.patch
  if [ "$(git -C "$CHECK_DIR" rev-parse 'HEAD^{tree}')" != "$(git -C "$CHECKOUT_DIR" rev-parse 'HEAD^{tree}')" ]; then
    echo "the exported patches do not rebuild $CHECKOUT_DIR"
    exit 1
  fi
}

# only the files that differ are written, and the ones without a commit removed
replace_patches() {
  local patch name
  for patch in "$PATCHES_DIR"/*.patch; do
    name=${patch##*/}
    if [ ! -e "$EXPORT_DIR/$name" ]; then
      rm -f "$patch"
    fi
  done
  for patch in "$EXPORT_DIR"/*.patch; do
    name=${patch##*/}
    if ! cmp -s "$patch" "$PATCHES_DIR/$name"; then
      cp "$patch" "$PATCHES_DIR/"
    fi
  done
}

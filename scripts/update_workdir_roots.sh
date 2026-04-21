#!/usr/bin/env bash
# Copyright (c) 2026 ETH Zurich
# Authors: see CONTRIBUTORS.md
# Licensed under the MIT License. See the LICENSE file in the repository root.
set -euo pipefail

# Safe bulk updater for default workdir in SLURM shell scripts.
# Targets only scripts/training and scripts/inference.
#
# Usage:
#   scripts/update_workdir_roots.sh --dry-run \
#     --old-workdir '/users/$USER/projects/ESFM' \
#     --new-workdir '/users/$USER/projects/ESFM_rebase'
#
#   scripts/update_workdir_roots.sh --apply ...

MODE=""
OLD_WORKDIR=''
NEW_WORKDIR=''
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

count_line_equals() {
  local needle="$1"
  shift
  local total=0
  local c
  for scope in "$@"; do
    c=$(grep -R -n -F -x -- "$needle" "$scope" 2>/dev/null | wc -l | tr -d ' ')
    total=$((total + c))
  done
  echo "$total"
}

count_line_starts_with() {
  local needle="$1"
  shift
  local total=0
  local c
  for scope in "$@"; do
    c=$(grep -R -n -F -- "$needle" "$scope" 2>/dev/null | wc -l | tr -d ' ')
    total=$((total + c))
  done
  echo "$total"
}

print_line_equals() {
  local needle="$1"
  shift
  for scope in "$@"; do
    grep -R -n -F -x -- "$needle" "$scope" 2>/dev/null || true
  done
}

usage() {
  cat <<'EOF'
Safe bulk workdir updater

Required:
  --dry-run | --apply
  --old-workdir PATH
  --new-workdir PATH

Examples:
  scripts/update_workdir_roots.sh --dry-run \
    --old-workdir '/users/$USER/projects/ESFM' \
    --new-workdir '/users/$USER/projects/ESFM_rebase'

  scripts/update_workdir_roots.sh --apply ...
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      MODE="dry-run"
      shift
      ;;
    --apply)
      MODE="apply"
      shift
      ;;
    --old-workdir)
      OLD_WORKDIR="$2"
      shift 2
      ;;
    --new-workdir)
      NEW_WORKDIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$MODE" || -z "$OLD_WORKDIR" || -z "$NEW_WORKDIR" ]]; then
  echo "Missing required arguments." >&2
  usage
  exit 1
fi

cd "$ROOT_DIR"

mapfile -t sh_files < <(
  find scripts/training scripts/inference -type f -name '*.sh' 2>/dev/null | sort
)

if [[ ${#sh_files[@]} -eq 0 ]]; then
  echo "No shell files found under scripts/training or scripts/inference."
  exit 1
fi

echo "Repo root: $ROOT_DIR"
echo "Mode: $MODE"
echo "Target shell files: ${#sh_files[@]}"

workdir_count_before=$(count_line_starts_with 'workdir="' scripts/training scripts/inference)
workdir_target_count_before=$(count_line_equals "workdir=\"$OLD_WORKDIR\"" scripts/training scripts/inference)

echo "Current workdir lines: $workdir_count_before"
echo "Current workdir lines matching old-workdir: $workdir_target_count_before"

if [[ "$MODE" == "dry-run" ]]; then
  echo
  echo "Planned replacement in scripts/training + scripts/inference:"
  echo "  workdir=\"$OLD_WORKDIR\"  ->  workdir=\"$NEW_WORKDIR\""
  echo
  echo "Matching lines:"
  print_line_equals "workdir=\"$OLD_WORKDIR\"" scripts/training scripts/inference
  exit 0
fi

backup_dir=".path_update_backups/workdir-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$backup_dir"

echo "Creating backups in $backup_dir"
for f in "${sh_files[@]}"; do
  mkdir -p "$backup_dir/$(dirname "$f")"
  cp -a "$f" "$backup_dir/$f"
done

echo "Applying replacements..."
for f in "${sh_files[@]}"; do
  tmp_file="$(mktemp)"
  awk -v old="workdir=\"$OLD_WORKDIR\"" -v new="workdir=\"$NEW_WORKDIR\"" '{
    if ($0 == old) {
      print new
    } else {
      print
    }
  }' "$f" > "$tmp_file"
  mv "$tmp_file" "$f"
done

workdir_target_count_after=$(count_line_equals "workdir=\"$NEW_WORKDIR\"" scripts/training scripts/inference)

echo "Updated workdir lines matching new-workdir: $workdir_target_count_after"
echo
echo "Done. Review changes with:"
echo "  git diff -- scripts/training scripts/inference"
echo "If needed, restore from backups in: $backup_dir"

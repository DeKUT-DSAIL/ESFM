#!/usr/bin/env bash
# Copyright (c) 2026 ETH Zurich
# Authors: see CONTRIBUTORS.md
# Licensed under the MIT License. See the LICENSE file in the repository root.
set -euo pipefail

# Safe bulk updater for log_dir base path in YAML config files.
# Targets files under configs/.
#
# Usage:
#   scripts/update_logdir_roots.sh --dry-run \
#     --old-log-base '/iopsstor/scratch/cscs/fozdemir/ESFM_outputs' \
#     --new-log-base '/new/path/to/ckpts/ESFM_outputs'
#
#   scripts/update_logdir_roots.sh --apply ...

MODE=""
OLD_LOG_BASE=''
NEW_LOG_BASE=''
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

count_line_starts_with() {
  local needle="$1"
  local scope="$2"
  grep -R -n -F -- "$needle" "$scope" 2>/dev/null | wc -l | tr -d ' '
}

print_line_starts_with() {
  local needle="$1"
  local scope="$2"
  grep -R -n -F -- "$needle" "$scope" 2>/dev/null || true
}

usage() {
  cat <<'EOF'
Safe bulk log_dir updater

Required:
  --dry-run | --apply
  --old-log-base PATH
  --new-log-base PATH

Examples:
  scripts/update_logdir_roots.sh --dry-run \
    --old-log-base '/iopsstor/scratch/cscs/fozdemir/ESFM_outputs' \
    --new-log-base '/new/path/to/ckpts/ESFM_outputs'

  scripts/update_logdir_roots.sh --apply ...
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
    --old-log-base)
      OLD_LOG_BASE="$2"
      shift 2
      ;;
    --new-log-base)
      NEW_LOG_BASE="$2"
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

if [[ -z "$MODE" || -z "$OLD_LOG_BASE" || -z "$NEW_LOG_BASE" ]]; then
  echo "Missing required arguments." >&2
  usage
  exit 1
fi

cd "$ROOT_DIR"

mapfile -t yaml_files < <(find configs -type f -name '*.yaml' | sort)

if [[ ${#yaml_files[@]} -eq 0 ]]; then
  echo "No YAML files found under configs/."
  exit 1
fi

echo "Repo root: $ROOT_DIR"
echo "Mode: $MODE"
echo "Target YAML files: ${#yaml_files[@]}"

logdir_target_count_before=$(count_line_starts_with "log_dir: $OLD_LOG_BASE/" configs)
echo "Current log_dir lines matching old-log-base: $logdir_target_count_before"

if [[ "$MODE" == "dry-run" ]]; then
  echo
  echo "Planned replacement in configs/*.yaml:"
  echo "  log_dir: $OLD_LOG_BASE/...  ->  log_dir: $NEW_LOG_BASE/..."
  echo
  echo "Matching lines:"
  print_line_starts_with "log_dir: $OLD_LOG_BASE/" configs
  exit 0
fi

backup_dir=".path_update_backups/logdir-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$backup_dir"

echo "Creating backups in $backup_dir"
for f in "${yaml_files[@]}"; do
  mkdir -p "$backup_dir/$(dirname "$f")"
  cp -a "$f" "$backup_dir/$f"
done

echo "Applying replacements..."
for f in "${yaml_files[@]}"; do
  tmp_file="$(mktemp)"
  awk -v old="$OLD_LOG_BASE" -v new="$NEW_LOG_BASE" '{
    prefix = "log_dir: " old "/"
    if (index($0, prefix) == 1) {
      suffix = substr($0, length(prefix) + 1)
      print "log_dir: " new "/" suffix
    } else {
      print
    }
  }' "$f" > "$tmp_file"
  mv "$tmp_file" "$f"
done

logdir_target_count_after=$(count_line_starts_with "log_dir: $NEW_LOG_BASE/" configs)
echo "Updated log_dir lines matching new-log-base: $logdir_target_count_after"
echo
echo "Done. Review changes with:"
echo "  git diff -- configs"
echo "If needed, restore from backups in: $backup_dir"

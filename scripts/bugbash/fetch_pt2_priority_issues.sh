#!/bin/bash
# Fetches high priority, triage review, and UBN issues for a given PyTorch
# module label and writes them to a CSV file.
#
# Usage:
#   ./scripts/fetch_priority_issues.sh "module: dynamo"
#   ./scripts/fetch_priority_issues.sh "module: inductor" --output my_report.csv
#   ./scripts/fetch_priority_issues.sh --all
#
# Requirements: gh (GitHub CLI, authenticated), python3

set -euo pipefail

REPO="pytorch/pytorch"
ONCALL_LABEL="oncall: pt2"
PRIORITY_LABELS=("high priority" "triage review" "pt2: ubn")

# Modules that only fetch priority-labeled issues
MODULE_LABELS=(
    "module: dynamo"
    "module: inductor"
    "module: dispatch"
    "module: pt2-dispatcher"
    "module: aotdispatch"
    "module: ddp"
    "module: fsdp"
    "module: dtensor"
)

# Smaller modules: fetch ALL open issues (not just priority-labeled)
MODULE_LABELS_ALL_ISSUES=(
    "oncall: export"
    "module: aotinductor"
    "module: dynamic shapes"
    "module: shape checking"
    "module: cuda graphs"
)
OUTPUT_FILE=""
FETCH_ALL=false
FETCH_UNCATEGORIZED=false
KEYWORD_SEARCHES=()

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] [MODULE_LABEL]

Fetch high priority / UBN / triage review issues for a PyTorch module and write to CSV.

Arguments:
  MODULE_LABEL    GitHub label, e.g. "module: dynamo"

Options:
  --all           Fetch issues for all known module labels (includes uncategorized)
  --uncategorized Also fetch priority issues not matching any known module label
  --search TERM   Also fetch oncall: pt2 issues matching keyword (repeatable)
  --output FILE   Output CSV file (default: priority_issues_<date>.csv)
  --help          Show this help

Known module labels (priority issues only):
$(printf '  %s\n' "${MODULE_LABELS[@]}")

Known module labels (all open issues):
$(printf '  %s\n' "${MODULE_LABELS_ALL_ISSUES[@]}")
EOF
    exit 0
}

# ── Parse arguments ──────────────────────────────────────────────────────────
MODULES_TO_FETCH=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --all)       FETCH_ALL=true; FETCH_UNCATEGORIZED=true; shift ;;
        --uncategorized) FETCH_UNCATEGORIZED=true; shift ;;
        --search)    KEYWORD_SEARCHES+=("$2"); shift 2 ;;
        --output)    OUTPUT_FILE="$2"; shift 2 ;;
        --help|-h)   usage ;;
        *)           MODULES_TO_FETCH+=("$1"); shift ;;
    esac
done

MODULES_ALL_ISSUES_TO_FETCH=()
if $FETCH_ALL; then
    MODULES_TO_FETCH=("${MODULE_LABELS[@]}")
    MODULES_ALL_ISSUES_TO_FETCH=("${MODULE_LABELS_ALL_ISSUES[@]}")
fi

if [[ ${#MODULES_TO_FETCH[@]} -eq 0 && ${#MODULES_ALL_ISSUES_TO_FETCH[@]} -eq 0 ]]; then
    echo "Error: specify a module label or use --all"
    echo "Run with --help for usage."
    exit 1
fi

if [[ -z "$OUTPUT_FILE" ]]; then
    OUTPUT_FILE="priority_issues_$(date +%Y%m%d).csv"
fi

# ── Preflight checks ────────────────────────────────────────────────────────
if ! command -v gh &>/dev/null; then
    echo "Error: gh (GitHub CLI) is required. Install: https://cli.github.com/"
    exit 1
fi

if ! gh auth status &>/dev/null; then
    echo "Error: gh is not authenticated. Run: gh auth login"
    exit 1
fi

# ── Fetch issues ─────────────────────────────────────────────────────────────
TMPDIR_WORK=$(mktemp -d)
trap 'rm -rf "$TMPDIR_WORK"' EXIT

JSON_FILE="$TMPDIR_WORK/issues.json"
echo "[]" > "$JSON_FILE"

# Helper: run gh and write result to batch.json, falling back to [] on failure
fetch_issues() {
    if gh issue list "$@" > "$TMPDIR_WORK/batch_raw.json" 2>/dev/null \
       && python3 -c "import json; json.load(open('$TMPDIR_WORK/batch_raw.json'))" 2>/dev/null; then
        mv "$TMPDIR_WORK/batch_raw.json" "$TMPDIR_WORK/batch.json"
    else
        echo "[]" > "$TMPDIR_WORK/batch.json"
    fi
}

for module in "${MODULES_TO_FETCH[@]}"; do
    for priority in "${PRIORITY_LABELS[@]}"; do
        echo "Fetching: ${module} + ${priority} + ${ONCALL_LABEL} ..."
        fetch_issues \
            --repo "$REPO" \
            --label "$ONCALL_LABEL" \
            --label "$module" \
            --label "$priority" \
            --state open \
            --limit 500 \
            --json number,title,labels,author,createdAt,updatedAt,url,assignees,comments

        # Merge into main JSON, tagging each issue with the module and priority
        python3 -c "
import json, sys
with open('$JSON_FILE') as f:
    existing = json.load(f)
with open('$TMPDIR_WORK/batch.json') as f:
    batch = json.load(f)
seen = {e['number'] for e in existing}
for issue in batch:
    if issue['number'] not in seen:
        issue['_module'] = '$module'
        issue['_priority'] = '$priority'
        existing.append(issue)
        seen.add(issue['number'])
with open('$JSON_FILE', 'w') as f:
    json.dump(existing, f)
"
    done
done

# ── Fetch all open issues for smaller modules ────────────────────────────────
for module in "${MODULES_ALL_ISSUES_TO_FETCH[@]}"; do
    echo "Fetching: ${module} + ${ONCALL_LABEL} (all open issues) ..."
    fetch_issues \
        --repo "$REPO" \
        --label "$ONCALL_LABEL" \
        --label "$module" \
        --state open \
        --limit 500 \
        --json number,title,labels,author,createdAt,updatedAt,url,assignees,comments

    python3 -c "
import json
with open('$JSON_FILE') as f:
    existing = json.load(f)
with open('$TMPDIR_WORK/batch.json') as f:
    batch = json.load(f)
seen = {e['number'] for e in existing}
for issue in batch:
    if issue['number'] not in seen:
        issue['_module'] = '$module'
        label_names = {l['name'] for l in issue.get('labels', [])}
        if 'pt2: ubn' in label_names:
            issue['_priority'] = 'pt2: ubn'
        elif 'high priority' in label_names:
            issue['_priority'] = 'high priority'
        elif 'triage review' in label_names:
            issue['_priority'] = 'triage review'
        else:
            issue['_priority'] = ''
        existing.append(issue)
        seen.add(issue['number'])
with open('$JSON_FILE', 'w') as f:
    json.dump(existing, f)
"
done

# ── Fetch uncategorized issues ────────────────────────────────────────────────
if $FETCH_UNCATEGORIZED; then
    # Build a JSON array of all known module labels for filtering
    ALL_KNOWN_LABELS=("${MODULE_LABELS[@]}" "${MODULE_LABELS_ALL_ISSUES[@]}")
    MODULE_LABELS_JSON=$(printf '%s\n' "${ALL_KNOWN_LABELS[@]}" | python3 -c "import sys,json; print(json.dumps([l.strip() for l in sys.stdin]))")

    for priority in "${PRIORITY_LABELS[@]}"; do
        echo "Fetching: uncategorized + ${priority} + ${ONCALL_LABEL} ..."
        fetch_issues \
            --repo "$REPO" \
            --label "$ONCALL_LABEL" \
            --label "$priority" \
            --state open \
            --limit 500 \
            --json number,title,labels,author,createdAt,updatedAt,url,assignees,comments

        # Add issues that don't have any of our known module labels
        python3 -c "
import json
with open('$JSON_FILE') as f:
    existing = json.load(f)
with open('$TMPDIR_WORK/batch.json') as f:
    batch = json.load(f)
known_modules = set($MODULE_LABELS_JSON)
seen = {e['number'] for e in existing}
for issue in batch:
    if issue['number'] in seen:
        continue
    issue_labels = {l['name'] for l in issue.get('labels', [])}
    if not issue_labels & known_modules:
        issue['_module'] = '(uncategorized)'
        issue['_priority'] = '$priority'
        existing.append(issue)
        seen.add(issue['number'])
with open('$JSON_FILE', 'w') as f:
    json.dump(existing, f)
"
    done
fi

# ── Fetch keyword search issues ──────────────────────────────────────────────
for keyword in "${KEYWORD_SEARCHES[@]+"${KEYWORD_SEARCHES[@]}"}"; do
    echo "Fetching: keyword '${keyword}' + ${ONCALL_LABEL} ..."
    fetch_issues \
        --repo "$REPO" \
        --label "$ONCALL_LABEL" \
        --search "$keyword" \
        --state open \
        --limit 500 \
        --json number,title,labels,author,createdAt,updatedAt,url,assignees,comments

    python3 -c "
import json
with open('$JSON_FILE') as f:
    existing = json.load(f)
with open('$TMPDIR_WORK/batch.json') as f:
    batch = json.load(f)
seen = {e['number'] for e in existing}
for issue in batch:
    if issue['number'] not in seen:
        issue['_module'] = '(search: $keyword)'
        label_names = {l['name'] for l in issue.get('labels', [])}
        if 'pt2: ubn' in label_names:
            issue['_priority'] = 'pt2: ubn'
        elif 'high priority' in label_names:
            issue['_priority'] = 'high priority'
        elif 'triage review' in label_names:
            issue['_priority'] = 'triage review'
        else:
            issue['_priority'] = ''
        existing.append(issue)
        seen.add(issue['number'])
with open('$JSON_FILE', 'w') as f:
    json.dump(existing, f)
"
done

COUNT=$(python3 -c "import json; print(len(json.load(open('$JSON_FILE'))))")
echo "Found $COUNT issues total."

if [[ "$COUNT" -eq 0 ]]; then
    echo "No issues found. Exiting."
    exit 0
fi

# ── Write CSV ────────────────────────────────────────────────────────────────
python3 - "$JSON_FILE" "$OUTPUT_FILE" <<'PYEOF'
import csv
import json
import sys
from datetime import datetime

json_path = sys.argv[1]
output_path = sys.argv[2]

with open(json_path) as f:
    issues = json.load(f)

# Sort: by module, then priority (UBN > high > triage), then newest-updated first
priority_order = {"pt2: ubn": 0, "high priority": 1, "triage review": 2, "": 3, "(none)": 3}
issues.sort(key=lambda i: (
    i.get("_module", ""),
    priority_order.get(i.get("_priority", ""), 99),
    -(datetime.fromisoformat(i["updatedAt"].replace("Z", "+00:00")).timestamp()
      if i.get("updatedAt") else 0),
))

headers = [
    "Issue #",
    "Title",
    "Module",
    "Priority",
    "Author",
    "Assignees",
    "Created",
    "Updated",
    "Comments",
    "All Labels",
    "URL",
]

num_cols = len(headers)
with open(output_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(headers)

    prev_module = None
    for issue in issues:
        cur_module = issue.get("_module", "")
        if prev_module is not None and cur_module != prev_module:
            writer.writerow([""] * num_cols)
        prev_module = cur_module
        label_names = [l["name"] for l in issue.get("labels", [])]
        assignee_names = ", ".join(
            a.get("login", "") for a in issue.get("assignees", [])
        ) or "(none)"
        author = issue.get("author", {}).get("login", "")

        created = issue.get("createdAt", "")[:10]
        updated = issue.get("updatedAt", "")[:10]

        if "pt2: ubn" in label_names:
            display_priority = "pt2: ubn"
        elif "high priority" in label_names:
            display_priority = "high priority"
        elif "triage review" in label_names:
            display_priority = "triage review"
        else:
            display_priority = issue.get("_priority", "") or "(none)"

        writer.writerow([
            issue["number"],
            issue["title"],
            issue.get("_module", ""),
            display_priority,
            author,
            assignee_names,
            created,
            updated,
            len(issue.get("comments", [])),
            ", ".join(label_names),
            issue["url"],
        ])

print(f"Wrote {len(issues)} issues to {output_path}")
PYEOF

echo "Done: $OUTPUT_FILE"

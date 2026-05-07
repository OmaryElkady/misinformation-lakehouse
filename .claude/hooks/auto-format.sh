#!/bin/bash
# Runs ruff/black after every file Claude writes or edits.
# Silently succeeds if formatters aren't installed — never blocks Claude.

FILE="${CLAUDE_TOOL_OUTPUT_FILE:-$1}"
if [ -z "$FILE" ]; then exit 0; fi

EXT="${FILE##*.}"

case "$EXT" in
  py)
    if command -v ruff >/dev/null 2>&1; then
      ruff format "$FILE" 2>/dev/null
    fi
    if command -v black >/dev/null 2>&1; then
      black --quiet "$FILE" 2>/dev/null
    fi
    ;;
esac

exit 0

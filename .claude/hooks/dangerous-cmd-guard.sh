#!/bin/bash
# Blocks destructive shell commands before Claude executes them.
# Exits with code 2 to block + explain. Exits 0 to allow.

INPUT=$(cat)
CMD=$(echo "$INPUT" | grep -oP '"command"\s*:\s*"\K[^"]+' 2>/dev/null || echo "$INPUT")
BLOCKED=""

case "$CMD" in
  *'rm -rf /'*|*'rm -rf ~'*|*'rm -rf .'*|*'rm -rf *'*)
    BLOCKED="Recursive force-delete targeting root, home, or current directory";;

  *'DROP TABLE'*|*'drop table'*|*'DROP DATABASE'*|*'drop database'*|*'TRUNCATE TABLE'*|*'truncate table'*)
    BLOCKED="Destructive database operation";;

  *'--force'*push*|*'push --force'*|*'push -f'*)
    BLOCKED="Force push can overwrite remote history";;

  *'chmod 777'*|*'chmod -R 777'*)
    BLOCKED="Setting world-writable permissions is a security risk";;

  *'mkfs'*|*'dd if='*)
    BLOCKED="Disk formatting or raw disk write detected";;

  *':(){:|:&};:'*)
    BLOCKED="Fork bomb detected";;

  *'curl'*'| bash'*|*'curl'*'| sh'*|*'wget'*'| bash'*|*'wget'*'| sh'*)
    BLOCKED="Piping remote content directly to shell is dangerous";;

  *'shutdown'*|*'reboot'*|*'halt'*|*'poweroff'*)
    BLOCKED="System power management command";;

  *'kill -9 -1'*|*'killall'*)
    BLOCKED="Mass process kill command";;
esac

if [ -n "$BLOCKED" ]; then
  echo "BLOCKED: $BLOCKED. Run this manually if you're sure." >&2
  exit 2
fi

exit 0

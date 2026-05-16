#!/usr/bin/env bash
# Launch a dedicated Chrome for post_to_instagram.py CDP automation.
#
# Pattern adapted from apply_jobs/start_chrome.sh: persistent profile so
# the Instagram login carries across runs, idempotent so re-running this
# while Chrome is already up is a no-op.

PORT=9222
PROFILE="/tmp/chrome-ave-debug-profile"

if curl -s "http://localhost:${PORT}/json/version" > /dev/null 2>&1; then
    echo "Chrome already on port ${PORT}"
    exit 0
fi

echo "Launching dedicated Chrome on port ${PORT} (profile: ${PROFILE})..."
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
    --remote-debugging-port=${PORT} \
    --remote-allow-origins=* \
    --user-data-dir="${PROFILE}" \
    --no-first-run \
    --no-default-browser-check \
    "https://www.instagram.com/" &

for _ in $(seq 1 15); do
    if curl -s "http://localhost:${PORT}/json/version" > /dev/null 2>&1; then
        echo "Chrome ready on port ${PORT}"
        echo "Log into Instagram in the opened window if it isn't already, then run post_to_instagram.py"
        exit 0
    fi
    sleep 1
done

echo "ERROR: Chrome didn't start in time"
exit 1

#!/bin/bash
# Starts the Ventuno home server (camera stream + NPU hand tracking + home state) unless it is already listening.
# Run from cron (@reboot and every minute) so the board works without the Mac; the Mac UI agent calls it too.
PORT=8090
SRC=/home/arduino/handpose/src
LOG=/home/arduino/handpose/logs/camera_stream.log
VENV_PY=/home/arduino/handpose/.venv/bin/python

ss -ltn | grep -q ":$PORT " && exit 0
mkdir -p "$(dirname "$LOG")"
if [ -x "$VENV_PY" ]; then PY=$VENV_PY; HANDS=--hands; else PY=python3; HANDS=; fi
cd "$SRC" || exit 1
# Detach from the caller's stdio (adb shell / cron) so it doesn't wait on the server.
exec < /dev/null > "$LOG" 2>&1
setsid nohup "$PY" "$SRC/camera_stream.py" --port "$PORT" $HANDS &

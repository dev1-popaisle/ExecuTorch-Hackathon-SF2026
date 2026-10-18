#!/bin/bash
# ====================================================================================================
# Sign & Say - Smart Home Mobility Project
# Author: Umit D. Sami
# Open source project - ExecuTorch Hackathon 2026 (San Francisco) - 11.18.2026
#
# This is a smart home mobility project for physically challenged individuals. The project is an IoT
# system running on the Arduino VENTUNO Q without the use of any cloud resources (everything runs inside
# the device, including advanced AI computation tasks such as speech and vision recognition).
# The project allows a physically challenged person to set the temperature in the house, turn the lights
# on and off, and change TV channels in two ways:
#   1. VENTUNO Q Voice AI assistant module
#   2. VENTUNO Q Hand Recognition with ExecuTorch running in the device
# ====================================================================================================
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

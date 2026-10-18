#!/bin/bash
# Installs the Ventuno Home UI as a per-user launchd agent. It starts at login, waits for the
# VENTUNO Q on USB, then serves http://127.0.0.1:9999 and opens it in the browser.
#   ./install.sh            install / update
#   ./install.sh uninstall  remove
set -euo pipefail

LABEL="com.ventuno.home-ui"
APP_DIR="$HOME/Library/Application Support/VentunoHome"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
DOMAIN="gui/$(id -u)"

if [[ "${1:-}" == "uninstall" ]]; then
    launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
    rm -f "$PLIST"
    echo "Uninstalled $LABEL (files left in $APP_DIR)"
    exit 0
fi

ADB="$(command -v adb || true)"
[[ -z "$ADB" && -x /opt/homebrew/bin/adb ]] && ADB=/opt/homebrew/bin/adb
[[ -z "$ADB" ]] && { echo "adb not found: brew install android-platform-tools"; exit 1; }

# Copy out of ~/Desktop: launchd agents can't read TCC-protected folders without extra permissions.
mkdir -p "$APP_DIR/logs" "$HOME/Library/LaunchAgents"
cp "$SRC_DIR/ventuno_ui.py" "$SRC_DIR/index.html" "$SRC_DIR/hls.min.js" "$APP_DIR/"
cp "$SRC_DIR/../device/handpose/camera_stream.py" "$SRC_DIR/../device/handpose/hand_tracker.py" \
   "$SRC_DIR/../device/handpose/gestures.py" "$SRC_DIR/../device/handpose/run_home_server.sh" "$APP_DIR/"

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>$LABEL</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/python3</string>
        <string>$APP_DIR/ventuno_ui.py</string>
    </array>
    <key>WorkingDirectory</key><string>$APP_DIR</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key><string>$(dirname "$ADB"):/usr/bin:/bin:/usr/sbin:/sbin</string>
        <key>PYTHONUNBUFFERED</key><string>1</string>
    </dict>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <key>ProcessType</key><string>Interactive</string>
    <key>StandardOutPath</key><string>$APP_DIR/logs/ui.log</string>
    <key>StandardErrorPath</key><string>$APP_DIR/logs/ui.log</string>
</dict>
</plist>
EOF

launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
launchctl bootstrap "$DOMAIN" "$PLIST"
launchctl kickstart -k "$DOMAIN/$LABEL"
echo "Installed $LABEL -> $APP_DIR (log: $APP_DIR/logs/ui.log)"

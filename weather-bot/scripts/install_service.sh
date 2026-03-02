#!/usr/bin/env bash
# install_service.sh — deploy weather-bot as an always-on systemd service
#
# Run this ON THE VM after pulling the latest code:
#   gcloud compute ssh tombaxter@YOUR_VM_NAME
#   cd ~/repo && git pull origin main
#   bash weather-bot/scripts/install_service.sh

set -euo pipefail

SERVICE_NAME="weather-bot"
SERVICE_FILE="$(dirname "${BASH_SOURCE[0]}")/${SERVICE_NAME}.service"
INSTALL_PATH="/etc/systemd/system/${SERVICE_NAME}.service"

echo "=== Weather Bot Service Installer ==="
echo ""

# ── 1. Verify the .env file exists ──────────────────────────────────────────
ENV_FILE="$(dirname "${BASH_SOURCE[0]}")/../.env"
if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: .env file not found at $ENV_FILE"
    echo "Create it from .env.example and fill in your API keys before continuing."
    exit 1
fi
echo "✓ .env file found"

# ── 2. Verify venv exists ────────────────────────────────────────────────────
VENV_PYTHON="$(dirname "${BASH_SOURCE[0]}")/../venv/bin/python3"
if [ ! -f "$VENV_PYTHON" ]; then
    echo "ERROR: venv not found at $VENV_PYTHON"
    echo "Create it with: cd weather-bot && python3 -m venv venv && venv/bin/pip install -r requirements.txt"
    exit 1
fi
echo "✓ venv found at $VENV_PYTHON"

# ── 3. Kill any existing cron job for the bot ────────────────────────────────
echo ""
echo "--- Checking for existing cron jobs ---"
# Show current crontab so you can see what's there
crontab -l 2>/dev/null || echo "(no crontab)"

# Remove lines containing main.py or weather-bot from crontab
if crontab -l 2>/dev/null | grep -qE "main\.py|weather.bot"; then
    echo "Removing bot cron entries..."
    crontab -l 2>/dev/null | grep -vE "main\.py|weather.bot" | crontab -
    echo "✓ Bot cron entries removed"
else
    echo "✓ No bot cron entries found"
fi

# Show remaining crontab (data sync crons should still be there)
echo "Remaining crontab:"
crontab -l 2>/dev/null || echo "(empty)"

# ── 4. Stop any existing bot process running in screen/tmux/nohup ────────────
echo ""
echo "--- Stopping any existing bot processes ---"
pkill -f "main.py" 2>/dev/null && echo "✓ Killed existing main.py process" || echo "✓ No existing main.py process"

# ── 5. Install the systemd service file ─────────────────────────────────────
echo ""
echo "--- Installing systemd service ---"
sudo cp "$SERVICE_FILE" "$INSTALL_PATH"
sudo chmod 644 "$INSTALL_PATH"
echo "✓ Service file installed at $INSTALL_PATH"

# Reload systemd to pick up the new file
sudo systemctl daemon-reload
echo "✓ systemd daemon reloaded"

# Enable: auto-start on boot
sudo systemctl enable "$SERVICE_NAME"
echo "✓ Service enabled (will auto-start on boot)"

# ── 6. Start the service ─────────────────────────────────────────────────────
echo ""
echo "--- Starting service ---"
sudo systemctl start "$SERVICE_NAME"
echo "✓ Service started"

# ── 7. Verify it's running ───────────────────────────────────────────────────
echo ""
echo "--- Status ---"
sleep 3   # give it a moment to initialise
sudo systemctl status "$SERVICE_NAME" --no-pager -l

echo ""
echo "=== Done ==="
echo ""
echo "Useful commands:"
echo "  Watch logs live:       journalctl -u weather-bot -f"
echo "  Last 100 log lines:    journalctl -u weather-bot -n 100 --no-pager"
echo "  Stop the bot:          sudo systemctl stop weather-bot"
echo "  Restart the bot:       sudo systemctl restart weather-bot"
echo "  Disable auto-start:    sudo systemctl disable weather-bot"
echo "  Check if running:      sudo systemctl is-active weather-bot"

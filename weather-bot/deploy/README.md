# VM Deployment (GCP)

This deploy bundle is designed to be safe and reproducible:
- No API secrets are stored in git.
- `systemd` service user/workdir are templated at install time.
- Healthcheck restarts the service if recent bot activity goes stale in `journald` or `bot.log`.
- Log rotation prevents unbounded log growth.
- Realtime settlement runs as a separate read-only sidecar service, never inside the trading engine.
- Setup/redeploy preserve VM runtime state instead of deleting the remote workdir.

## Files
- `setup_gcloud_vm.sh`: one-shot VM bootstrap + service install, preserving existing VM runtime data if rerun.
- `redeploy.sh`: sync code + reinstall deps + restart service without overwriting runtime data.
- `safe_remote_sync.py`: shared helper that updates managed code while preserving runtime-managed files.
- `weather-bot.service.template`: systemd unit template.
- `weather-settlement-watcher.service.template`: realtime settlement sidecar unit.
- `weather-dashboard.service.template`: authoritative VM-local Streamlit dashboard.
- `weather-dashboard-api.service.template`: read-only VM API for public Streamlit Cloud dashboards.
- `weather-bot-logrotate`: logrotate template.
- `healthcheck.sh`: stale heartbeat detector and restarter.
- `install_cron_jobs.sh`: installs healthcheck + daily calibration cron.
- `ops.sh`: daily operations shortcuts (status/heartbeat/trades/restart).

## Quick Start
1. From local `weather-bot/`:
   - `chmod +x deploy/*.sh`
   - `./deploy/setup_gcloud_vm.sh`
2. Set secrets on VM:
   - `gcloud compute ssh weather-bot --zone us-east1-b --command 'sudo nano /etc/weather-bot.env'`
3. Restart service:
   - `gcloud compute ssh weather-bot --zone us-east1-b --command 'sudo systemctl restart weather-bot'`
4. Install cron jobs on VM:
   - `gcloud compute ssh weather-bot --zone us-east1-b --command 'cd ~/weather-bot && chmod +x deploy/*.sh && ./deploy/install_cron_jobs.sh'`

## Runtime Data Preservation

Rerunning `setup_gcloud_vm.sh` or `redeploy.sh` will preserve VM-managed runtime state in place:

- `logs/`
- `venv/` and `.venv/`
- top-level `.env`
- all non-Python files under `data/` such as `positions*.json`, watcher snapshots, caches, and local SQLite/DB files

Only managed code is refreshed. The deploy scripts no longer `rm -rf` the entire remote `weather-bot` directory.

If you intentionally want a clean reset of runtime state, back it up first and delete those files explicitly on the VM. Do not use the normal deploy scripts for destructive resets.

## Observability
- Service logs: `sudo journalctl -u weather-bot -f`
- Settlement watcher logs: `sudo journalctl -u weather-settlement-watcher -f`
- Dashboard API logs: `sudo journalctl -u weather-dashboard-api -f`
- File logs: `tail -f ~/weather-bot/logs/bot.log` if your installed service writes to files
- Settlement watcher file logs: `tail -f ~/weather-bot/logs/settlement_watcher.log`
- Dashboard API file logs: `tail -f ~/weather-bot/logs/dashboard_api.log`
- Healthcheck log: `tail -f ~/weather-bot/logs/healthcheck.log`
- Last heartbeat: `sudo journalctl -u weather-bot --no-pager -n 300 | grep HEARTBEAT | tail -1`
- Fallback heartbeat: `rg HEARTBEAT ~/weather-bot/logs/bot.log | tail -1`
- Settlement snapshot: `jq '.' ~/weather-bot/data/settlement_status.json`
- Settlement overlay rows: `jq '.rows | length' ~/weather-bot/data/settlement_snapshot.json`
- Trade audit query: `python3 scripts/query_trade_audit.py --date 2026-03-23 --strategy PRIME_ALPHA --limit 20`
- Replay deterministic PRIME_ALPHA fixture: `python3 scripts/replay_prime_alpha.py --fixture data/prime_alpha_scenarios/atlanta_2026-03-23.json`

## Public Streamlit Cloud via VM API

To keep the public `streamlit.app` link while reading live VM data:

1. Install or restart the API service on the VM:
   - `bash deploy/install_dashboard_api_service.sh`
2. Expose `127.0.0.1:8510` through your preferred reverse proxy or tunnel.
3. Set the public Streamlit app secrets:
   - `DASHBOARD_DATA_SOURCE=api`
   - `DASHBOARD_API_BASE_URL=https://<public-api-base-url>`
   - `DASHBOARD_API_TOKEN=<value from /etc/weather-bot.env>`

The API is read-only and the dashboard remains read-only in `api` mode.

## Daily Operations
- `./deploy/ops.sh status`
- `./deploy/ops.sh heartbeat`
- `./deploy/ops.sh trades`
- `./deploy/ops.sh logs`
- `./deploy/ops.sh restart`

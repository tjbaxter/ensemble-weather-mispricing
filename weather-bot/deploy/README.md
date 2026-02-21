# VM Deployment (GCP)

This deploy bundle is designed to be safe and reproducible:
- No API secrets are stored in git.
- `systemd` service user/workdir are templated at install time.
- Healthcheck restarts the service if `HEARTBEAT` goes stale.
- Log rotation prevents unbounded log growth.

## Files
- `setup_gcloud_vm.sh`: one-shot VM bootstrap + service install.
- `redeploy.sh`: sync code + reinstall deps + restart service.
- `weather-bot.service.template`: systemd unit template.
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

## Observability
- Service logs: `sudo journalctl -u weather-bot -f`
- File logs: `tail -f ~/weather-bot/logs/bot.log`
- Healthcheck log: `tail -f ~/weather-bot/logs/healthcheck.log`
- Last heartbeat: `rg HEARTBEAT ~/weather-bot/logs/bot.log | tail -1`

## Daily Operations
- `./deploy/ops.sh status`
- `./deploy/ops.sh heartbeat`
- `./deploy/ops.sh trades`
- `./deploy/ops.sh logs`
- `./deploy/ops.sh restart`

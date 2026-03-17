#!/usr/bin/env bash
set -euo pipefail
python3 - <<'PY'
from pathlib import Path
import shutil
src_root = Path('/home/tombaxter/weather-bot')
dst_root = Path('/home/tombaxter/repo/weather-bot')
files_data = [
    'positions.json','positions_live.json','positions_shadow_2a.json','positions_shadow_2b.json','positions_shadow_2c.json',
    'positions_shadow_purdey.json','positions_shadow_cavendish.json','positions_shadow_purdey2.json',
    'positions_shadow_cavendish3.json','positions_shadow_true_alpha.json','positions_shadow_props_kelly.json',
    'polymarket_cache.json','commercial_forecast_log.json','model_snapshot_log.json','model_accuracy_log.json',
    'accuracy_rows_cache.json','morning_obs_cache.json'
]
for name in files_data:
    s = src_root / 'data' / name
    d = dst_root / 'data' / name
    if s.exists():
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, d)

for name in ['resolved.csv','signals.csv','trades.csv']:
    s = src_root / 'logs' / name
    d = dst_root / 'logs' / name
    if s.exists():
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, d)

for slug in ['shadow_2a','shadow_2b','shadow_2c','shadow_purdey','shadow_cavendish','shadow_purdey2','shadow_cavendish3','shadow_true_alpha','shadow_props_kelly']:
    s = src_root / 'logs' / slug / 'resolved.csv'
    d = dst_root / 'logs' / slug / 'resolved.csv'
    if s.exists():
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, d)
PY

cd /home/tombaxter/repo
bash /home/tombaxter/repo/weather-bot/scripts/git_push_data.sh >> /tmp/git_sync.log 2>&1

"""Resubmit campaign rows that have no DONE, giving up after 3 attempts."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
MANIFEST = _REPO / 'slurm' / 'campaign.tsv'
LOG = _REPO / 'CAMPAIGN_LOG.md'
RESULTS = _REPO / 'results'
LOGS = _REPO / 'logs'
MAX_ATTEMPTS = 3
MAX_CONCURRENT = 24  # matches campaign.sbatch; see its comment on the core cap


def load_rows() -> list[tuple[int, str]]:
    rows = []
    index = 0
    for line in MANIFEST.read_text().splitlines():
        if not line.strip() or line.startswith('run_id'):
            continue
        run_id = line.split('\t', 1)[0]
        rows.append((index, run_id))
        index += 1
    return rows


def attempts(run_id: str) -> list[Path]:
    root = RESULTS / run_id
    if not root.exists():
        return []
    return sorted(
        path for path in root.glob('attempt_*')
        if path.name.split('_', 1)[-1].isdigit()
    )


def has_done(run_id: str) -> bool:
    return any((path / 'DONE').is_file() for path in attempts(run_id))


def log(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    with LOG.open('a') as handle:
        handle.write(f'- {stamp}: {message}\n')


def tail(path: Path, lines: int = 40) -> str:
    if not path.exists():
        return f'(no log at {path})'
    text = path.read_text(errors='replace').splitlines()
    return '\n'.join(text[-lines:])


def main() -> int:
    missing = []
    for index, run_id in load_rows():
        if has_done(run_id):
            continue
        n_attempts = len(attempts(run_id))
        if n_attempts >= MAX_ATTEMPTS:
            log_tail = tail(LOGS / f'latest_{index}.out')
            # Prefer the newest slurm log matching this array id if present.
            array_logs = sorted(LOGS.glob(f'*_{index}.out'))
            if array_logs:
                log_tail = tail(array_logs[-1])
            log(
                f'PERMANENT FAIL {run_id} after {n_attempts} attempts.\n'
                f'  log tail:\n{log_tail}')
            print(f'give up: {run_id}')
            continue
        missing.append(str(index))
    if not missing:
        print('nothing to resubmit')
        return 0
    array = ','.join(missing)
    cmd = ['sbatch', f'--array={array}%{MAX_CONCURRENT}', str(_REPO / 'slurm' / 'campaign.sbatch')]
    print(' '.join(cmd))
    subprocess.check_call(cmd, cwd=_REPO)
    log(f'resubmitted array indices {array}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

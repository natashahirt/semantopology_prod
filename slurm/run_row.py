#!/usr/bin/env python3
"""Run one campaign.tsv row. Called by campaign.sbatch.

Set ``CAMPAIGN_MANIFEST`` (a path relative to the repo, or absolute) to run a
row from a gate manifest instead, e.g. ``slurm/projection_gate.tsv``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
MANIFEST = _REPO / os.environ.get('CAMPAIGN_MANIFEST', 'slurm/campaign.tsv')


def load_row(index: int) -> tuple[str, list[str]]:
    rows = []
    for line in MANIFEST.read_text().splitlines():
        if not line.strip() or line.startswith('run_id'):
            continue
        run_id, payload = line.split('\t', 1)
        rows.append((run_id, json.loads(payload)))
    if index < 0 or index >= len(rows):
        raise SystemExit(f'array index {index} out of range 0..{len(rows)-1}')
    return rows[index]


def already_done(run_id: str) -> bool:
    root = _REPO / 'results' / run_id
    return any(path.is_file() for path in root.glob('attempt_*/DONE'))


def main() -> int:
    index = int(os.environ.get('SLURM_ARRAY_TASK_ID', sys.argv[1] if len(sys.argv) > 1 else '0'))
    run_id, argv = load_row(index)
    if already_done(run_id):
        print(f'{run_id}: DONE exists, skipping')
        return 0
    cmd = [sys.executable, str(_REPO / 'run.py'), *argv]
    print(' '.join(cmd), flush=True)
    return subprocess.call(cmd, cwd=_REPO)


if __name__ == '__main__':
    raise SystemExit(main())

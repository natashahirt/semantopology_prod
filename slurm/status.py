"""Print campaign status: done / running / failed / pending per experiment."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
MANIFEST = _REPO / 'slurm' / 'campaign.tsv'
RESULTS = _REPO / 'results'


def load_ids() -> list[str]:
    ids = []
    for line in MANIFEST.read_text().splitlines():
        if not line.strip() or line.startswith('run_id'):
            continue
        ids.append(line.split('\t', 1)[0])
    return ids


def attempt_state(run_id: str) -> str:
    root = RESULTS / run_id
    if not root.exists():
        return 'pending'
    dones = list(root.glob('attempt_*/DONE'))
    if dones:
        return 'done'
    attempts = sorted(root.glob('attempt_*'))
    if not attempts:
        return 'pending'
    latest = attempts[-1]
    run_json = latest / 'run.json'
    if run_json.exists():
        try:
            record = json.loads(run_json.read_text())
        except json.JSONDecodeError:
            return 'failed'
        if record.get('exit_status') == 'ok':
            return 'done'
        return 'failed'
    return 'running'


def main() -> int:
    counts = defaultdict(lambda: defaultdict(int))
    totals = defaultdict(int)
    for run_id in load_ids():
        experiment = run_id.split('/', 1)[0]
        state = attempt_state(run_id)
        counts[experiment][state] += 1
        totals[state] += 1
    print(f"{'exp':<6} {'done':>6} {'run':>6} {'fail':>6} {'pend':>6}")
    for experiment in sorted(counts):
        row = counts[experiment]
        print(
            f"{experiment:<6} {row['done']:6d} {row['running']:6d} "
            f"{row['failed']:6d} {row['pending']:6d}")
    print(
        f"{'ALL':<6} {totals['done']:6d} {totals['running']:6d} "
        f"{totals['failed']:6d} {totals['pending']:6d}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

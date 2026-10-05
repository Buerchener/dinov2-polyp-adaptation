"""CPU-only utilities shared by the controller and foreground runtime."""
import contextlib
import hashlib
import json
import os
import time
import uuid
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.pending-' + uuid.uuid4().hex)
    try:
        with tmp.open('w') as f:
            f.write(json.dumps(value, indent=2) + '\n')
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


@contextlib.contextmanager
def local_lock(path):
    """OS lock survives stale files, but never a terminated owner process."""
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise RuntimeError('Another controller owns this experiment') from e
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def remaining_budget(budget, attempts, now=None):
    """Count all attempts, including prep/failure and unresolved allocations."""
    now = time.time() if now is None else now
    seconds = float(budget.get('prior_seconds', 0))
    cu = float(budget.get('prior_cu_projection', 0))
    for a in attempts:
        end = a.get('release_observed_at', now)
        duration = max(0, end - a['allocation_requested_at'])
        seconds += duration
        cu += duration * a['own_rate_upper_cu_hour'] / 3600
    return dict(used_seconds=seconds, used_cu_projection=cu,
                remaining_seconds=budget['max_seconds'] - seconds,
                remaining_cu_projection=budget['max_cu_projection'] - cu)


def deadlines(budget, attempts, allocated_at):
    left = remaining_budget(budget, attempts, allocated_at)
    allowance = min(left['remaining_seconds'],
                    left['remaining_cu_projection'] * 3600 / budget['own_rate_upper_cu_hour'])
    reserve = budget['save_release_reserve_seconds']
    if allowance <= reserve + budget['preparation_seconds']:
        raise RuntimeError('Remaining cumulative budget cannot cover preparation and save/release reserve')
    return dict(release_by=allocated_at + allowance,
                train_by=allocated_at + allowance - reserve,
                prep_by=allocated_at + budget['preparation_seconds'])


def safe_extract(archive, destination):
    import stat
    import zipfile
    destination = Path(destination)
    with zipfile.ZipFile(archive) as z:
        for entry in z.infolist():
            p = Path(entry.filename)
            if p.is_absolute() or '..' in p.parts or '\\' in entry.filename:
                raise ValueError('Unsafe archive path')
            if stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError('Archive symlink is not supported')
        z.extractall(destination)

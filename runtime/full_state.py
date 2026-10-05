"""Full-state generations; checkpoint loads are limited to own verified files."""
import json
import os
import random
import shutil
import tempfile
import time
from pathlib import Path

from core import atomic_json, digest


REQUIRED = {'format', 'experiment_lock', 'model', 'optimizer', 'scheduler',
            'scaler', 'modes', 'rng', 'progress', 'loader', 'evaluations', 'artifact_sha256'}


def capture(model, optimizer, *, progress, loader, evaluations,
            scheduler=None, scaler=None):
    import numpy as np
    import torch
    return dict(format='colab-full-state-v1', model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=None if scheduler is None else scheduler.state_dict(),
                scaler=None if scaler is None else scaler.state_dict(),
                modes=dict(scheduler=None if scheduler is None else type(scheduler).__qualname__,
                           scaler=None if scaler is None else type(scaler).__qualname__),
                rng=dict(python=random.getstate(), numpy=np.random.get_state(),
                         torch=torch.get_rng_state(),
                         cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None),
                progress=progress, loader=loader, evaluations=evaluations)


def audit(state, experiment_lock):
    if not REQUIRED.issubset(state) or state['format'] != 'colab-full-state-v1':
        raise ValueError('Inference weights are not a full checkpoint')
    if state['experiment_lock'] != experiment_lock:
        raise ValueError('Code/data/protocol lock mismatch')
    p = state['progress']
    if p.get('boundary') != 'full_update' or min(p['epoch'], p['step'], p['optimizer_updates']) < 0:
        raise ValueError('Unsafe checkpoint progress')
    if not {'state', 'param_groups'}.issubset(state['optimizer']):
        raise ValueError('Optimizer state missing')
    if not {'python', 'numpy', 'torch', 'cuda'}.issubset(state['rng']):
        raise ValueError('RNG state missing')
    if not state['loader'].get('resume_strategy') or 'next_batch' not in state['loader']:
        raise ValueError('Exact next-batch strategy/state must be explicit')
    for key in ('scheduler', 'scaler'):
        if (state['modes'][key] is None) != (state[key] is None):
            raise ValueError('Optional state/mode mismatch')


def restore(state, model, optimizer, scheduler=None, scaler=None):
    import numpy as np
    import torch
    expected = dict(scheduler=None if scheduler is None else type(scheduler).__qualname__,
                    scaler=None if scaler is None else type(scaler).__qualname__)
    if state['modes'] != expected:
        raise ValueError('Scheduler/AMP mode changed')
    cuda = state['rng']['cuda']
    if cuda is not None and (not torch.cuda.is_available() or len(cuda) != torch.cuda.device_count()):
        raise ValueError('CUDA RNG topology changed')
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(state['scheduler'])
    if scaler is not None:
        scaler.load_state_dict(state['scaler'])
    random.setstate(state['rng']['python'])
    np.random.set_state(state['rng']['numpy'])
    torch.set_rng_state(state['rng']['torch'])
    if cuda is not None:
        torch.cuda.set_rng_state_all(cuda)


class Store:
    def __init__(self, root, experiment_lock):
        self.root, self.lock = Path(root), experiment_lock
        self.root.mkdir(parents=True, exist_ok=True)

    def _read_generation(self, folder, expected_manifest_sha):
        import torch
        if digest(folder / 'manifest.json') != expected_manifest_sha:
            raise ValueError('Manifest hash mismatch')
        manifest = json.loads((folder / 'manifest.json').read_text())
        for name, record in manifest['files'].items():
            if Path(name).name != name:
                raise ValueError('Unsafe artifact name')
            p = folder / name
            if p.stat().st_size != record['bytes'] or digest(p) != record['sha256']:
                raise ValueError('Checkpoint/artifact hash mismatch: ' + name)
        # Own state only: pickle is never loaded before the lock and hashes pass.
        if manifest['experiment_lock'] != self.lock:
            raise ValueError('Experiment changed')
        state = torch.load(folder / 'resume.pt', map_location='cpu', weights_only=False)
        audit(state, self.lock)
        if state['progress'] != manifest['progress']:
            raise ValueError('Progress differs from manifest')
        if state['artifact_sha256'] != {name: item['sha256'] for name, item in manifest['files'].items()
                                       if name != 'resume.pt'}:
            raise ValueError('Artifact hashes are not bound to full state')
        return state, manifest

    def load(self):
        latest = json.loads((self.root / 'latest.json').read_text())
        name = latest['generation']
        if Path(name).name != name:
            raise ValueError('Unsafe generation name')
        folder = self.root / 'generations' / name
        state, manifest = self._read_generation(folder, latest['manifest_sha256'])
        return state, manifest, latest

    def commit(self, state, artifacts):
        import torch
        state = dict(state, experiment_lock=self.lock,
                     artifact_sha256={name: digest(source) for name, source in artifacts.items()})
        audit(state, self.lock)
        previous = self.load()[2] if (self.root / 'latest.json').exists() else None
        if previous:
            old = self.load()[0]['progress']
            if state['progress']['optimizer_updates'] < old['optimizer_updates']:
                raise ValueError('Refuse checkpoint rollback')
        generations = self.root / 'generations'
        generations.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.pending-', dir=generations) as td:
            pending = Path(td) / 'generation'
            pending.mkdir()
            with (pending / 'resume.pt').open('wb') as f:
                torch.save(state, f)
                f.flush()
                os.fsync(f.fileno())
            for name, source in artifacts.items():
                if Path(name).name != name or name in ('resume.pt', 'manifest.json'):
                    raise ValueError('Artifact must have a safe unique basename')
                with Path(source).open('rb') as inp, (pending / name).open('wb') as out:
                    shutil.copyfileobj(inp, out)
                    out.flush()
                    os.fsync(out.fileno())
                if digest(source) != digest(pending / name):
                    raise ValueError('Artifact destination readback mismatch')
            files = {p.name: dict(sha256=digest(p), bytes=p.stat().st_size)
                     for p in pending.iterdir()}
            manifest = dict(experiment_lock=self.lock, progress=state['progress'],
                            evaluations=state['evaluations'], files=files,
                            storage_readback_verified=True, cloud_durability_verified=False,
                            committed_at=time.time())
            atomic_json(pending / 'manifest.json', manifest)
            name = files['resume.pt']['sha256']
            destination = generations / name
            if previous and previous['generation'] == name:
                self.load()
                return previous
            if destination.exists():
                # Torch can serialize identical state bytes again; keep original manifest.
                if digest(destination / 'resume.pt') != name:
                    raise ValueError('Existing generation is corrupt')
            else:
                pending.replace(destination)
            # Full destination byte/schema readback precedes latest-pointer commit.
            manifest_sha = digest(destination / 'manifest.json')
            self._read_generation(destination, manifest_sha)
            latest = dict(generation=name, previous=previous['generation'] if previous else None,
                          manifest_sha256=manifest_sha)
            atomic_json(self.root / 'latest.json', latest)
            try:
                self.load()
            except Exception:
                if previous:
                    atomic_json(self.root / 'latest.json', previous)
                else:
                    (self.root / 'latest.json').unlink(missing_ok=True)
                raise
            # DriveFS readback is not proof of cloud durability. Retain every
            # generation during this bounded recovery; never prune from this path.
            return latest


    def accept_cloud_receipt(self, receipt):
        """Receipt supplied only after independent cloud downloads/hash validation.

        A mounted-file read or this method itself does not constitute verification.
        This method authenticates identity/content binding, not the caller.
        """
        if receipt.get('source') != 'independent_drive_download':
            raise ValueError('Independent cloud readback required')
        if receipt.get('experiment_lock') != self.lock:
            raise ValueError('Wrong experiment')
        generation = receipt['generation']
        if Path(generation).name != generation or len(generation) != 64:
            raise ValueError('Invalid generation')
        state, manifest = self._read_generation(
            self.root / 'generations' / generation, receipt['manifest_sha256'])
        if receipt.get('files') != manifest['files']:
            raise ValueError('Incomplete or mismatched cloud verification')
        record = dict(receipt, progress=state['progress'])
        prior_path = self.root / 'cloud-confirmed.json'
        if prior_path.exists():
            prior = json.loads(prior_path.read_text())
            if prior['progress']['optimizer_updates'] > record['progress']['optimizer_updates']:
                raise ValueError('Cloud confirmation rollback')
        atomic_json(prior_path, record)
        return record

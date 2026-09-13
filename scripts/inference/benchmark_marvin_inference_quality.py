#!/usr/bin/env python3
"""Generate graded warehouse tasks and run paired inference-quality ablations."""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from scripts.inference.benchmark_marvin_collision_optimizations import (
    DEFAULT_CONFIG, ROOT, execute_case,
)


def immutable_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Existing experiment differs; use a new output directory: {path}")
    else:
        path.write_text(json.dumps(value, indent=2, allow_nan=False))


def a5_defaults(base):
    config = deepcopy(base)
    config['n_trajectory_samples'] = 32
    config['collision_optimization'] = {
        'pair_streaming': {'enabled': True, 'pair_chunk_size': 1024},
        'reduced_guide_geometry': {'enabled': True, 'profile': 'foam_pika_100'},
    }
    config.setdefault('dense_validation', {})['chunking'] = {
        'enabled': True, 'candidate_chunk_size': 8,
        'time_chunk_size': 32, 'self_pair_chunk_size': 1024,
    }
    config.setdefault('gradient_pruning', {}).setdefault('spatial', {}).setdefault(
        'link_broad_phase', {}).update(
            enabled=True, full_scan=True, scan_geometry='parent_bounds',
            environment_margin=0.2, self_margin=0.1)
    return config


def build_cases(base):
    base = a5_defaults(base)
    cases = [('baseline', deepcopy(base), 1), ('candidates-4x32', deepcopy(base), 4)]
    for key, values in (
        ('ddim_sampling_timesteps', [10, 20, 30, 50]),
        ('t_start_guide_steps_fraction', [0.2, 0.5, 0.75, 1.0]),
        ('n_guide_steps', [2, 4, 8, 12]),
    ):
        for value in values:
            if base['ddim'][key] == value:
                continue
            config = deepcopy(base)
            config['ddim'][key] = value
            cases.append((f'{key}-{value}', config, 1))
    for end in [2.0, 4.0]:
        config = deepcopy(base)
        for component in ['position', 'orientation']:
            name = 'CostTaskSpaceEEGoal' + component.title()
            start = float(config['costs'][name]['weight'])
            config['ddim'][f'ee_goal_{component}_weight_start'] = start
            config['ddim'][f'ee_goal_{component}_weight_end'] = start * end
        cases.append((f'ee-late-{end:g}x', config, 1))
    config = deepcopy(base)
    config['compute_costs_with_xrecon'] = not bool(base['compute_costs_with_xrecon'])
    cases.append(('xrecon-toggle', config, 1))
    return cases


def select_checkpoints(run_dir, top_k):
    """Rank saved full checkpoints by matching-step validation total loss.

    Local trainer-owned NPY files contain dictionaries (trusted pickle input).
    Validation is a training-loss proxy, not a planning-success ranking.
    """
    train = yaml.safe_load((run_dir / 'args.yaml').read_text())
    prefix = 'ema_' if train.get('use_ema', False) else ''
    losses = np.load(run_dir / 'checkpoints/val_losses.npy', allow_pickle=True)
    ranked = {}
    for step, metrics in losses:
        loss = float(metrics['VALIDATION total_loss'])
        filename = f'{prefix}model__iter_{int(step):06d}.pth'
        if np.isfinite(loss) and (run_dir / 'checkpoints' / filename).is_file():
            ranked[filename] = {'checkpoint': filename, 'step': int(step), 'val_loss': loss}
    result = sorted(ranked.values(), key=lambda item: (item['val_loss'], item['step']))[:top_k]
    if not result:
        raise ValueError(f'No saved full checkpoints matching validation steps: {run_dir}')
    return train, result


def checkpoint_cases(base, runs, top_k):
    cases, audit = [], []
    for run in runs:
        run = run.resolve()
        train, selected = select_checkpoints(run, top_k)
        variant = train['bimanual_network_variant']
        if train['dataset_subdir'] != base['dataset_subdir']:
            raise ValueError(f'Dataset differs for {run}; compare datasets in a separate experiment')
        for item in selected:
            config = a5_defaults(base)
            config['runtime']['network_variant'] = variant
            config['model_dir_ddpm_bspline'] = str(run)
            config['checkpoint'] = item['checkpoint']
            name = f'checkpoint-{variant}-{run.parent.name}-{item["step"]}'
            cases.append((name, config, 1))
            audit.append(dict(item, variant=variant, run=str(run)))
    return cases, audit


def model_fingerprints(cases):
    result = {}
    for _, config, _ in cases:
        run = Path(os.path.expandvars(config['model_dir_ddpm_bspline'])).expanduser().resolve()
        if not (run / 'args.yaml').exists():
            runs = list(run.glob('*/args.yaml'))
            if len(runs) != 1:
                raise ValueError(f'Configure an unambiguous model run: {run}')
            run = runs[0].parent
        train = yaml.safe_load((run / 'args.yaml').read_text())
        checkpoint = config.get('checkpoint') or (
            ('ema_' if train.get('use_ema') else '') + 'model_current.pth')
        path = run / 'checkpoints' / checkpoint
        stat = path.stat()
        result[str(path)] = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                            'training_args': train}
    return result


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['case']].append(row)
    summary = {}
    for case, items in groups.items():
        tasks = defaultdict(list)
        for item in items:
            tasks[(item['request_index'], item['seed'])].append(item)
        complete = [group for group in tasks.values() if len(group) == group[0]['batches']]
        summary[case] = {
            'complete_request_seed_groups': len(complete),
            'successful_groups': sum(any(x['status'] == 'success' for x in group) for group in complete),
            'success_rate': (sum(any(x['status'] == 'success' for x in group) for group in complete)
                             / len(complete)) if complete else None,
            'mean_total_wall_s': float(np.mean([sum(x['wall_seconds'] for x in group)
                                              for group in complete])) if complete else None,
            'status_counts': dict((status, sum(x['status'] == status for x in items))
                                  for status in sorted({x['status'] for x in items})),
            'candidate_counts_reported': {
                key: sum((x.get('candidates') or {}).get(key, 0) for x in items)
                for key in ['generated', 'dense_checked', 'valid']
            },
        }
    return summary


def summarize_by_tier(rows):
    return {tier: summarize([row for row in rows if row.get('tier', 'external') == tier])
            for tier in sorted({row.get('tier', 'external') for row in rows})}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'scripts/inference/logs/marvin-quality-graded')
    parser.add_argument('--requests', nargs='+', type=Path,
                        help='Optional external requests; default freshly sampled graded tasks')
    parser.add_argument('--regions', type=Path, default=ROOT / 'data_generation_cfgs/EnvWarehouse-RobotMarvinBimanual-independent.yaml')
    parser.add_argument('--tiers', nargs='+', choices=['easy', 'medium', 'hard', 'extreme'],
                        default=['easy', 'medium', 'hard', 'extreme'])
    parser.add_argument('--tasks-per-scenario', type=int, default=3)
    parser.add_argument('--max-sampling-attempts', type=int, default=30)
    parser.add_argument('--sampling-timeout-s', type=float, default=15)
    parser.add_argument('--seeds', nargs='+', type=int, default=[12345, 23456, 34567])
    parser.add_argument('--checkpoint-runs', nargs='*', type=Path,
                        help='Explicit run directories containing args.yaml; default original A/B/C/D runs')
    parser.add_argument('--top-k', type=int, choices=[1, 2, 3], default=3)
    parser.add_argument('--cases', nargs='+', help='Optional exact case names; see materialized manifest')
    parser.add_argument('--skip-checkpoints', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--prepare-only', action='store_true', help='Generate tasks and manifest without inference')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--timeout-s', type=float, default=600)
    args = parser.parse_args(argv)
    if min(args.tasks_per_scenario, args.max_sampling_attempts, args.sampling_timeout_s, args.timeout_s) <= 0:
        parser.error('Task counts, attempt limits and timeouts must be positive')
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s + 3 >= 2**32 for s in args.seeds):
        parser.error('seeds must be unique and within [0, 2**32-4]')
    if not args.prepare_only and args.device.startswith('cuda'):
        import torch
        if not torch.cuda.is_available():
            parser.error('CUDA unavailable in this interpreter; activate mpd-splines-public and check GPU access, '
                         'or use --prepare-only to generate tasks without inference')
    base = yaml.safe_load(args.config.read_text())
    # Resolve relative source paths before materializing YAML elsewhere.
    for key in ['start_goal_states_path', 'start_goal_regions_path']:
        if base.get(key):
            path = Path(os.path.expandvars(base[key])).expanduser()
            base[key] = str((args.config.resolve().parent / path).resolve())
    cases = build_cases(base)
    audit = []
    if not args.skip_checkpoints:
        runs = args.checkpoint_runs
        if runs is None:
            runs = [ROOT / f'logs/marvin_bimanual/warehouse_independent_v3_dual_ee_variant_{v}/1726484688'
                    for v in 'ABCD']
        extra, audit = checkpoint_cases(base, runs, args.top_k)
        cases.extend(extra)
    if args.cases:
        unknown = set(args.cases) - {name for name, _, _ in cases}
        if unknown:
            parser.error(f'Unknown cases: {sorted(unknown)}')
        cases = [case for case in cases if case[0] in args.cases]
    task_metadata = []
    if args.requests:
        requests = [json.loads(path.read_text()) for path in args.requests]
        task_metadata = [dict(tier='external', scenario='external') for _ in requests]
    else:
        task_dir = args.output_dir.resolve() / 'tasks'
        subprocess.run([sys.executable, '-m', 'scripts.inference.generate_marvin_quality_tasks',
                        '--output-dir', str(task_dir), '--regions', str(args.regions.resolve()),
                        '--tiers', *args.tiers, '--count', str(args.tasks_per_scenario),
                        '--attempts', str(args.max_sampling_attempts), '--timeout', str(args.sampling_timeout_s)],
                       cwd=ROOT, check=True)
        generated = json.loads((task_dir / 'tasks.json').read_text())
        requests = [task['request'] for task in generated['tasks']]
        task_metadata = [{k: v for k, v in task.items() if k != 'request'} for task in generated['tasks']]
        print('Sampling counts/shortfalls: ' + json.dumps(generated['scenarios']), flush=True)
        if not requests:
            parser.error('No collision-valid task generated; see tasks/tasks.json for shortfalls')
    # A checkpoint hash belongs to one model; silently clearing it would weaken
    # the user's request contract. Require model-neutral benchmark inputs.
    if any(req.get('checkpoint_hash') or req.get('deadline_unix_ns') for req in requests):
        parser.error('Benchmark requests must have no checkpoint_hash or absolute deadline')
    manifest = {'cases': [{'name': n, 'config': c, 'batches': b} for n, c, b in cases],
                'requests': requests, 'seeds': args.seeds, 'checkpoint_selection': audit,
                'model_fingerprints': model_fingerprints(cases),
                'task_metadata': task_metadata,
                'device': args.device, 'timeout_s': args.timeout_s}
    immutable_json(args.output_dir / 'manifest.json', manifest)
    print(json.dumps({'cases': len(cases), 'requests': len(requests),
                      'subprocess_runs': sum(b for _, _, b in cases) * len(requests) * len(args.seeds),
                      'checkpoint_selection': audit}, indent=2))
    if args.prepare_only:
        return 0
    rows = []
    for name, config, batches in cases:
        config_path = args.output_dir.resolve() / 'configs' / f'{name}.yaml'
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        for index, original in enumerate(requests):
            for seed in args.seeds:
                for batch in range(batches):
                    artifact = args.output_dir.resolve() / 'runs' / name / f'r{index}-s{seed}-b{batch}'
                    artifact.mkdir(parents=True, exist_ok=True)
                    request = deepcopy(original)
                    request['seed'] = seed + batch
                    request_path = artifact / 'input-request.json'
                    immutable_json(request_path, request)
                    report = artifact / 'benchmark-result.json'
                    if report.exists():
                        row = json.loads(report.read_text())
                    else:
                        run_args = SimpleNamespace(request=request_path, start_goal_source='dataset',
                            sample_index=0, seed=seed + batch, device=args.device,
                            timeout_s=args.timeout_s, gpu_poll_interval=0.2)
                        row = execute_case(config_path, artifact, run_args)
                        row.update(case=name, request_index=index, seed=seed, batch=batch, batches=batches)
                        row.update(tier=task_metadata[index]['tier'], scenario=task_metadata[index]['scenario'])
                        immutable_json(report, row)
                    rows.append(row)
                    (args.output_dir / 'summary.json').write_text(json.dumps(summarize(rows), indent=2))
                    (args.output_dir / 'summary-by-tier.json').write_text(json.dumps(summarize_by_tier(rows), indent=2))
                    print(f'{name} request={index} seed={seed} batch={batch}: {row["status"]}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""Resumable, process-isolated IK/RRT sampling near observed workspace edges."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import csv
from contextlib import contextmanager
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import uuid

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import yaml

ROOT = Path(__file__).resolve().parents[2]


@contextmanager
def asset_initialization_lock(output):
    # Separate PID namespaces may expose the same PID to the legacy URDF
    # writer. Serialize construction/teardown of those shared temporary assets.
    import fcntl
    with (output / '.asset-init.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('w') as handle:
        handle.write(json.dumps(value, indent=2) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    directory = os.open(str(path.parent), os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


@contextmanager
def rrt_slot(output):
    """One RRT at a time, including separately invoked --cell workers."""
    import fcntl
    while True:
        for index in range(1):
            lock = (output / f'.rrt-slot-{index}.lock').open('a')
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close()
                continue
            try:
                yield index
            finally:
                lock.close()
            return
        time.sleep(.1)


def recover_corrupt_records(output):
    """Preserve broken files; never discard valid IK targets or pair manifests."""
    archive = output / 'recovery' / str(time.time_ns())
    moved = []
    def quarantine(path):
        if not path.exists():
            return
        relative = path.relative_to(output)
        target = archive / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        path.rename(target)
        moved.append(str(relative))
    for path in list(output.rglob('*.json')):
        if 'recovery' in path.relative_to(output).parts:
            continue
        try:
            read(path)
        except (ValueError, UnicodeError):
            quarantine(path)
    settings = read(output / 'settings.json')
    for cell in settings['regions']:
        dest = output / cell
        for path in (dest / 'ik').glob('*.json'):
            target = dest / 'targets' / path.name
            if not target.exists():
                save(target, read(path)['target'])
        pairs_path = dest / 'pairs.json'
        if pairs_path.exists():
            for pair in read(pairs_path):
                if not (dest / 'ik' / f"{pair['target_index']:04d}.json").exists():
                    raise RuntimeError(f'Pair references lost IK result: {pairs_path}; preserved for explicit recovery')
        for path in (dest / 'rrt').glob('*/edge-result.json'):
            r = read(path)
            if r['raw_success']:
                trajectory = path.parent / 'trajectory.npz'
                try:
                    with np.load(trajectory) as arrays:
                        assert arrays['raw_positions'].shape[1] == 14
                        if r['spline_success']:
                            assert arrays['spline_positions'].shape[1] == 14
                except (OSError, ValueError, KeyError, AssertionError, EOFError):
                    quarantine(trajectory)
                    quarantine(path)
        complete = dest / 'complete.json'
        if complete.exists():
            pairs = read(pairs_path) if pairs_path.exists() else []
            if len(list((dest/'ik').glob('*.json'))) != settings['targets_per_cell'] or any(
                not (dest/'rrt'/f'{i:04d}'/'edge-result.json').exists() for i in range(len(pairs))
            ):
                quarantine(complete)
    if moved:
        save(archive / 'receipt.json', dict(preserved_files=moved))
    return moved


def edge_regions():
    """Narrow boxes straddling observed boundaries, not inferred safe boxes."""
    regions = {}
    for arm in ('left', 'right'):
        sign = 1 if arm == 'left' else -1
        other = 'right' if arm == 'left' else 'left'
        specs = [
            ('shelf_lower_xedge', arm + '_cabinet', [.50, .62] if arm == 'left' else [.46, .58], [.64, .78], [.16, .30]),
            ('shelf_lower_yedge', arm + '_cabinet', [.30, .50], [.76, .90], [.18, .30]),
            ('shelf_upper_xedge', arm + '_cabinet', [.36, .46], [.62, .76], [.46, .56]),
            ('shelf_upper_yedge', arm + '_cabinet', [.26, .38], [.74, .86], [.46, .56]),
            ('cross_table_yedge', other + '_table', [.44, .80], [.14, .30], [-.02, .10]),
            ('cross_table_xedge', other + '_table', [.74, .84], [.00, .18], [-.02, .10]),
        ]
        for name, base, x, y, z in specs:
            direction = -sign if name.startswith('cross') else sign
            regions[f'{arm}_{name}'] = dict(arm=arm, base_region=base,
                translation=dict(x=[x], y=[sorted(v * direction for v in y)], z=[z]))
    return regions


def select_pairs(records, starts, maximum, starts_per_goal, rng):
    # One start for every valid distinct goal before reusing any goal.
    goals, seen = [], set()
    for r in records:
        if not r['collision_valid']:
            continue
        key = tuple(r['q_goal'])
        if key not in seen:
            seen.add(key)
            goals.append(r)
    rng.shuffle(goals)
    orders = [rng.permutation(len(starts)).tolist() for _ in goals]
    pairs = []
    for round_index in range(min(starts_per_goal, len(starts))):
        for record, order in zip(goals, orders):
            start_index = order[round_index]
            if np.linalg.norm(np.asarray(starts[start_index]) - record['q_goal']) < .08:
                continue
            pairs.append(dict(target_index=record['target_index'], start_index=start_index,
                q_start=starts[start_index], q_goal=record['q_goal']))
            if len(pairs) == maximum:
                return pairs
    return pairs


def prepare_saved_pairs(output):
    """Freeze pair queues after IK is complete, without starting any RRT."""
    import fcntl
    settings = read(output / 'settings.json')
    for cell in settings['regions']:
        dest = output / cell
        if (dest / 'pairs.json').exists():
            continue
        with (dest / '.worker.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if (dest / 'pairs.json').exists():
                continue
            records = [read(p) for p in sorted((dest/'ik').glob('*.json'))]
            if len(records) != settings['targets_per_cell']:
                raise RuntimeError(f'IK sampling incomplete for {cell}')
            save(dest/'pairs.json', select_pairs(records, read(dest/'starts.json'),
                settings['pairs_per_cell'], settings['starts_per_goal'], np.random.default_rng()))


def worker(args):
    import torch
    torch.set_num_threads(1)
    from scripts.generate_data.generate_marvin_warehouse_bimanual import MarvinWarehouseGenerator
    from scripts.inference.benchmark_marvin_workspace_generalization import run_rrt_once
    settings = read(args.output_dir / 'settings.json')
    spec = settings['regions'][args.cell]
    dest = args.output_dir / args.cell
    config = yaml.safe_load((args.output_dir / 'regions-snapshot.yaml').read_text())
    config['planner_allowed_time'] = settings['rrt_seconds']
    with asset_initialization_lock(args.output_dir):
        generator = MarvinWarehouseGenerator(config, None, progress_label=args.cell)
    region = deepcopy(config['placement_regions'][spec['base_region']])
    region['translation'] = spec['translation']
    generator.regions[args.cell] = region
    rng = generator.rng
    arm = spec['arm']
    sl = slice(0, 7) if arm == 'left' else slice(7, 14)
    lo, hi = generator.robot.joint_bounds_low_np[sl], generator.robot.joint_bounds_high_np[sl]
    try:
        starts_path = dest / 'starts.json'
        if not starts_path.exists():
            audit = read(args.output_dir / 'source-ik.json')['results'][f'{arm}:{arm}_table']
            starts = []
            for r in audit['records']:
                if r['collision_valid'] and r['q'] not in starts and generator.valid(r['q']):
                    starts.append(r['q'])
            if not starts:
                raise RuntimeError('No revalidated same-side table reference states')
            save(starts_path, starts)
        starts = read(starts_path)
        for index in range(settings['targets_per_cell']):
            result_path = dest / 'ik' / f'{index:04d}.json'
            if result_path.exists():
                continue
            target_path = dest / 'targets' / f'{index:04d}.json'
            if not target_path.exists():
                pos, rot = generator._sample_pose(args.cell)
                save(target_path, dict(position=pos.tolist(), rotation=rot.tolist(),
                    reference_index=int(rng.integers(len(starts)))))
            target = read(target_path)
            pos, rot = np.array(target['position']), np.array(target['rotation'])
            reference = np.array(starts[target['reference_index']])
            record = dict(target_index=index, target=target, ik_solved=False,
                collision_valid=False, restarts_used=0, pose_solutions=0,
                collision_rejected_solutions=0, best_position_error_m=None,
                best_orientation_error_rad=None)
            started = time.perf_counter()
            for restart in range(settings['ik_restarts']):
                q = reference.copy()
                seed = np.clip(reference[sl], lo, hi) if restart == 0 else rng.uniform(lo, hi)
                def residual(x):
                    q[sl] = x
                    pose = generator._pose(q, arm)
                    return np.r_[pose.translation - pos, Rotation.from_matrix(rot.T @ pose.rotation).as_rotvec()]
                fit = least_squares(residual, seed, bounds=(lo, hi), max_nfev=100,
                    ftol=1e-6, xtol=1e-6, gtol=1e-6)
                q[sl] = fit.x
                ep, er = float(np.linalg.norm(fit.fun[:3])), float(np.linalg.norm(fit.fun[3:]))
                record['restarts_used'] += 1
                if record['best_position_error_m'] is None or ep < record['best_position_error_m']:
                    record['best_position_error_m'], record['best_orientation_error_rad'] = ep, er
                if ep > .003 or er > np.deg2rad(2):
                    continue
                record['ik_solved'] = True
                record['pose_solutions'] += 1
                if generator.valid(q):
                    record.update(collision_valid=True, q_goal=q.tolist(),
                        achieved_position=generator._pose(q, arm).translation.tolist())
                    break
                record['collision_rejected_solutions'] += 1
            record['wall_seconds'] = time.perf_counter() - started
            save(result_path, record)
            if (index + 1) % 10 == 0:
                print(f'{args.cell} IK {index+1}/{settings["targets_per_cell"]}', flush=True)
        records = [read(p) for p in sorted((dest / 'ik').glob('*.json'))]
        if args.ik_only:
            return
        pairs_path = dest / 'pairs.json'
        if not pairs_path.exists():
            save(pairs_path, select_pairs(records, starts, settings['pairs_per_cell'], settings['starts_per_goal'], rng))
        pairs = read(pairs_path)
        for index, pair in enumerate(pairs):
            output = dest / 'rrt' / f'{index:04d}'
            if (output / 'edge-result.json').exists():
                continue
            request = dict(pair, request_id=f'{args.cell}-{index:04d}')
            save(output / 'request.json', request)
            # Recheck full 14D endpoints, including the other arm, before planning.
            if not generator.valid(pair['q_start']) or not generator.valid(pair['q_goal']):
                raise RuntimeError('Persisted pair failed endpoint revalidation')
            with rrt_slot(args.output_dir) as slot:
                started = time.perf_counter()
                started_ns = time.time_ns()
                result = run_rrt_once(generator, request, output)
                result.update(total_wall_seconds=time.perf_counter() - started,
                    ompl_exact=bool(result['stats'].get('rrt_exact', 0)),
                    target_index=pair['target_index'], start_index=pair['start_index'],
                    rrt_slot=slot, started_unix_ns=started_ns, finished_unix_ns=time.time_ns())
            if result['raw_success']:
                # Commit trajectory bytes before the durable completion record.
                with (output / 'trajectory.npz').open('rb') as trajectory:
                    os.fsync(trajectory.fileno())
            save(output / 'edge-result.json', result)
            print(f'{args.cell} RRT {index+1}/{len(pairs)} {result["status"]}', flush=True)
        save(dest / 'complete.json', dict(complete=True))
    finally:
        with asset_initialization_lock(args.output_dir):
            generator.close()


def summarize(output):
    settings = read(output / 'settings.json')
    rows = []
    for name, spec in settings['regions'].items():
        dest = output / name
        records = [read(p) for p in sorted((dest / 'ik').glob('*.json'))]
        pairs = read(dest / 'pairs.json') if (dest / 'pairs.json').exists() else []
        results = [read(p) for p in sorted((dest / 'rrt').glob('*/edge-result.json'))]
        by_index = {r['target_index']: r for r in records}
        def xyz_bounds(points):
            if not points:
                return None
            values = np.asarray(points)
            return {a: [float(values[:, i].min()), float(values[:, i].max())]
                    for i, a in enumerate('xyz')}
        rrt_goal_ids = {r['target_index'] for r in results if r['raw_success']}
        spline_goal_ids = {r['target_index'] for r in results if r['spline_success']}
        rows.append(dict(cell=name, arm=spec['arm'], bounds=spec['translation'],
            tcp_targets=len(records), ik_solved=sum(r['ik_solved'] for r in records),
            collision_valid=sum(r['collision_valid'] for r in records),
            ik_wall_seconds=sum(r['wall_seconds'] for r in records),
            pairs_generated=len(pairs), pairs_tested=len(results),
            pair_shortfall=settings['pairs_per_cell']-len(pairs),
            unique_starts=len({r['start_index'] for r in results}),
            unique_goals=len({r['target_index'] for r in results}),
            ompl_exact=sum(r['ompl_exact'] for r in results),
            rrt_raw_success=sum(r['raw_success'] for r in results),
            spline_success=sum(r['spline_success'] for r in results),
            rrt_wall_seconds=sum(r['total_wall_seconds'] for r in results),
            rrt_wall_median_seconds=statistics.median(r['total_wall_seconds'] for r in results) if results else None,
            rrt_wall_mean_seconds=statistics.mean(r['total_wall_seconds'] for r in results) if results else None,
            rrt_wall_p95_seconds=float(np.percentile([r['total_wall_seconds'] for r in results], 95)) if results else None,
            ik_rate=sum(r['ik_solved'] for r in records)/len(records) if records else None,
            collision_valid_rate=sum(r['collision_valid'] for r in records)/len(records) if records else None,
            rrt_raw_rate=sum(r['raw_success'] for r in results)/len(results) if results else None,
            spline_rate=sum(r['spline_success'] for r in results)/len(results) if results else None,
            unique_rrt_success_goals=len(rrt_goal_ids), unique_spline_success_goals=len(spline_goal_ids),
            ik_solved_target_xyz=xyz_bounds([r['target']['position'] for r in records if r['ik_solved']]),
            collision_valid_achieved_xyz=xyz_bounds([r['achieved_position'] for r in records if r['collision_valid']]),
            rrt_success_goal_xyz=xyz_bounds([by_index[i]['achieved_position'] for i in rrt_goal_ids]),
            spline_success_goal_xyz=xyz_bounds([by_index[i]['achieved_position'] for i in spline_goal_ids]),
            complete=(dest / 'complete.json').exists()))
    save(output / 'summary.json', dict(rows=rows, complete=all(r['complete'] for r in rows)))
    with (output / 'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# Marvin 边缘加密采样实测', '',
        'IK：每个独立 TCP 目标最多20次重启，3 mm / 2°容差，完整球与网格碰撞检查。',
        '另一臂在 IK 中固定于抽取的完整参考状态；RRT 为14D双臂规划，可协调绕障。',
        '参考起点来自上一轮同侧桌面有效状态，并在本轮重新检查。旋转范围沿用各目标区域。',
        '每个有效目标最多搭配5个不同起点，优先不同目标；统计条件于这些姿态和起点，不代表连续可达范围。',
        '10秒为 OMPL solve 预算，不包含原始路径复检、样条拟合和样条复检，实际墙钟可能超出。',
        '本报告没有动态限制验证或训练集写入。随机目标、参考状态及已生成pairs在恢复时保持不变。', '',
        '|区域|TCP|IK|无碰撞|RRT测试|OMPL exact|原始路径有效|样条有效|独立起点/目标|pair缺额|耗时中位秒|完成|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|']
    for r in rows:
        lines.append('|'+ '|'.join(str(v) for v in [r['cell'],r['tcp_targets'],r['ik_solved'],r['collision_valid'],r['pairs_tested'],r['ompl_exact'],r['rrt_raw_success'],r['spline_success'],f"{r['unique_starts']}/{r['unique_goals']}",r['pair_shortfall'],round(r['rrt_wall_median_seconds'],2) if r['rrt_wall_median_seconds'] is not None else '-',r['complete']])+'|')
    lines += ['', '## 采样盒子（world，米）', '']
    for name, spec in settings['regions'].items():
        lines.append(f'- {name}: `{spec["translation"]}`')
    lines += ['', '## 已观测到的有效目标范围', '',
        '下列范围是离散样本的轴对齐包围盒，不表示盒内任意点可行。RRT成功目标数按target索引去重。', '',
        '|区域|无碰撞目标XYZ|RRT成功目标XYZ|独立RRT成功目标|独立样条成功目标|',
        '|---|---|---|---:|---:|']
    for r in rows:
        def fmt(bounds):
            return '; '.join(f'{a}=[{lo:.4f},{hi:.4f}]' for a, (lo,hi) in bounds.items()) if bounds else '无'
        lines.append(f"|{r['cell']}|{fmt(r['collision_valid_achieved_xyz'])}|{fmt(r['rrt_success_goal_xyz'])}|{r['unique_rrt_success_goals']}|{r['unique_spline_success_goals']}|")
    (output / 'REPORT.md').write_text('\n'.join(lines)+'\n')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--source-dir', type=Path, default=ROOT/'scripts/inference/logs/marvin-workspace-dense-v3')
    parser.add_argument('--targets-per-cell', type=int, default=200)
    parser.add_argument('--ik-restarts', type=int, default=20)
    parser.add_argument('--pairs-per-cell', type=int, default=100)
    parser.add_argument('--starts-per-goal', type=int, default=5)
    parser.add_argument('--workers', type=int, choices=(1,), default=1)
    parser.add_argument('--recover', action='store_true', help='Archive corrupt records before resuming (no active workers)')
    parser.add_argument('--cell', help=argparse.SUPPRESS)
    parser.add_argument('--ik-only', action='store_true', help='Complete TCP/IK sampling without running RRT')
    parser.add_argument('--prepare-pairs-only', action='store_true', help='Freeze pair queues from complete saved IK')
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.prepare_pairs_only:
        prepare_saved_pairs(args.output_dir)
        summarize(args.output_dir)
        return
    if args.cell:
        # A queued orchestrator and a manually scheduled region can coexist.
        # Lock before touching targets/results and recheck completion after wait.
        import fcntl
        dest = args.output_dir / args.cell
        dest.mkdir(parents=True, exist_ok=True)
        with (dest / '.worker.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not (dest / 'complete.json').exists():
                worker(args)
        return
    if args.summarize_only:
        print(json.dumps(summarize(args.output_dir), indent=2))
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.recover:
        moved = recover_corrupt_records(args.output_dir)
        print(f'Preserved {len(moved)} damaged/incomplete files under recovery/', flush=True)
    settings = dict(regions=edge_regions(), targets_per_cell=args.targets_per_cell,
        ik_restarts=args.ik_restarts, pairs_per_cell=args.pairs_per_cell,
        starts_per_goal=args.starts_per_goal, rrt_seconds=10,
        source_dir=str(args.source_dir.resolve()))
    if (args.output_dir/'settings.json').exists():
        if read(args.output_dir/'settings.json') != settings:
            raise ValueError('Resume settings differ; use original settings or new output directory')
    else:
        save(args.output_dir/'settings.json', settings)
        save(args.output_dir/'source-ik.json', read(args.source_dir/'ik-reachability.json'))
        (args.output_dir/'regions-snapshot.yaml').write_text((args.source_dir/'configs/regions-snapshot.yaml').read_text())
    save(args.output_dir/'resume-events'/f'{time.time_ns()}.json',
         dict(workers=args.workers, ik_only=args.ik_only, rrt_concurrency_limit=1, started_unix_ns=time.time_ns()))
    def launch(name):
        dest = args.output_dir/name
        dest.mkdir(exist_ok=True)
        if (dest/'complete.json').exists():
            return name, 0
        if args.ik_only and len(list((dest/'ik').glob('*.json'))) == settings['targets_per_cell']:
            return name, 0
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        code = None
        for attempt in range(3):
            with (dest/'worker.log').open('a') as log:
                command = [sys.executable, str(Path(__file__).resolve()), '--output-dir', str(args.output_dir), '--cell', name]
                if args.ik_only:
                    command.append('--ik-only')
                code = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
            save(dest/'process-status.json', dict(returncode=code, attempts=attempt+1))
            if code == 0 or (dest/'complete.json').exists():
                break
        return name, code
    from concurrent.futures import as_completed
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(launch, name) for name in settings['regions']]
        for future in as_completed(futures):
            name, code = future.result()
            print(f'{name}: process exit={code}', flush=True)
            summarize(args.output_dir)
    rows = summarize(args.output_dir)
    if not args.ik_only and not all(r['complete'] for r in rows):
        raise SystemExit(2)


if __name__ == '__main__':
    main()

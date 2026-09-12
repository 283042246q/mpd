#!/usr/bin/env python3
"""Spatial re-analysis of persisted IK targets and exact paired RRT trials.

No IK/planner reruns and no changes to the endpoint pool. Cell assignments for
IK use requested TCP coordinates; RRT uses achieved FK coordinates of the exact
arm configuration recovered from that pool. Rates are conditional on the saved
orientation and other-arm distributions, not continuous reachability proofs.
"""
import argparse
import csv
from itertools import product
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def cells_for(group):
    bounds = group['bounds']
    shelf = 'cabinet' in group['region']
    edges = {}
    for axis in 'xyz':
        lo, hi = bounds[axis][0]
        split = {'x': .40 if shelf else .60,
                 'y': .75 if shelf else .25,
                 'z': (.525 if lo > .4 else .20) if shelf else 0.0}[axis]
        if axis == 'y' and hi <= 0:
            split = -split
        edges[axis] = [lo, split, hi] if lo < split < hi else [lo, hi]
    return [{axis: [edges[axis][i], edges[axis][i + 1]]
             for axis, i in zip('xyz', indices)}
            for indices in product(*(range(len(edges[a]) - 1) for a in 'xyz'))], edges


def locate(point, edges):
    # Half-open bins, inclusive outer boundary. Slight IK residual outside the
    # requested box remains visible as outside instead of being silently clipped.
    if any(p < edges[a][0] or p > edges[a][-1] for a, p in zip('xyz', point)):
        return None
    indices = [min(int(np.searchsorted(edges[a], p, side='right') - 1), len(edges[a]) - 2)
               for a, p in zip('xyz', point)]
    shape = tuple(len(edges[a]) - 1 for a in 'xyz')
    return int(np.ravel_multi_index(tuple(indices), shape))


def summarize(root):
    audit = read(root / 'ik-reachability.json')['results']
    manifest = read(root / 'pair-manifest.json')
    rows, pools, grids = {}, {}, {}
    for key, group in audit.items():
        boxes, edges = cells_for(group)
        grids[key] = edges
        for index, bounds in enumerate(boxes):
            rows[(key, index)] = dict(group=key, cell=index, bounds=bounds,
                tcp_samples=0, reference_valid=0, ik_solved=0, collision_valid=0,
                rrt_endpoint_incidence=0, rrt_raw_success=0, rrt_spline_success=0,
                unique_endpoint_keys=set())
        arm_slice = slice(0, 7) if group['arm'] == 'left' else slice(7, 14)
        pools[key] = {}
        for record in group['records']:
            index = locate(record['target_position'], edges)
            if index is None:
                raise ValueError(f'Target outside configured region: {key}')
            row = rows[(key, index)]
            row['tcp_samples'] += 1
            for field in ('reference_valid', 'ik_solved', 'collision_valid'):
                row[field] += int(record.get(field, False))
            if record['collision_valid']:
                pools[key][tuple(record['q'][arm_slice])] = record['achieved_position']
    endpoints, transitions = [], []
    for pair in manifest['pairs']:
        request = read(root / pair['request'])
        result_path = root / 'rrt' / pair['case'] / f"pair-{pair['pair_index']:03d}" / 'result.json'
        result = read(result_path) if result_path.exists() else None
        locations = {}
        for role in ('start', 'goal'):
            for arm, arm_slice in [('left', slice(0, 7)), ('right', slice(7, 14))]:
                key = f"{arm}:{pair[role][arm]}"
                q_key = tuple(request[f'q_{role}'][arm_slice])
                if q_key not in pools[key]:
                    raise ValueError(f'Cannot recover exact FK endpoint: {pair["request"]} {role} {arm}')
                point = pools[key][q_key]
                index = locate(point, grids[key])
                locations[(role, arm)] = (key, index)
                endpoints.append(dict(case=pair['case'], pair=pair['pair_index'],
                    role=role, arm=arm, group=key, cell=index, achieved_xyz=point,
                    rrt_tested=result is not None,
                    raw_success=bool(result and result['raw_success']),
                    spline_success=bool(result and result['spline_success'])))
                if index is not None and result is not None:
                    row = rows[(key, index)]
                    row['rrt_endpoint_incidence'] += 1
                    row['rrt_raw_success'] += int(result['raw_success'])
                    row['rrt_spline_success'] += int(result['spline_success'])
                    row['unique_endpoint_keys'].add(q_key)
        transitions.append(dict(case=pair['case'], pair=pair['pair_index'],
            start={a: locations[('start', a)] for a in ('left', 'right')},
            goal={a: locations[('goal', a)] for a in ('left', 'right')},
            rrt_tested=result is not None, raw_success=bool(result and result['raw_success']),
            spline_success=bool(result and result['spline_success'])))
    for row in rows.values():
        row['unique_rrt_endpoints'] = len(row.pop('unique_endpoint_keys'))
        row['ik_rate'] = row['ik_solved'] / row['tcp_samples'] if row['tcp_samples'] else None
        row['collision_valid_rate'] = row['collision_valid'] / row['tcp_samples'] if row['tcp_samples'] else None
        # No training dataset ingestion occurs in this report.
        row['dataset_ingested'] = None
    return dict(schema='marvin_workspace_cells/v1', cells=list(rows.values()),
                endpoints=endpoints, transitions=transitions)


def write_report(root, destination):
    report = summarize(root)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'cells.json').write_text(json.dumps(report, indent=2) + '\n')
    with (destination / 'cells.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report['cells'][0]))
        writer.writeheader()
        writer.writerows(report['cells'])
    lines = ['# Marvin 工作区细分统计', '', f'输入：`{root}`', '',
        '这是既有 1200 次 TCP 采样、100 组双臂 pair 的空间重统计，未新增 IK/RRT 运行。',
        'IK 使用采样目标坐标；RRT 使用对应关节状态的实际 FK 坐标。区间左闭右开，最外侧上边界包含。',
        'IK 成功率分母为 TCP 采样数；无碰撞判定包含另一臂参考姿态。失败不等于几何不可达。',
        'RRT 列为关联该 cell 的端点出现次数，start/goal 分别计数；不是独立 RRT 次数，不能跨行相加为成功 pair 数。',
        '独立端点按该臂 7D 关节状态去重。完整双臂 cell→cell 连接和起终点角色详见 cells.json 的 transitions/endpoints。',
        '原始 RRT 与样条统计沿用原 benchmark 验证；未增加动态验证、训练集写入或可达性证明。',
        '样条有效数表示已有可用路径记录，不代表已进入训练集；dataset_ingested 标记为 null。', '']
    for key in sorted({r['group'] for r in report['cells']}):
        rows = [r for r in report['cells'] if r['group'] == key]
        lines += [f'## {key}', '',
            '| cell | x | y | z | TCP | IK | 无碰撞 | 独立RRT端点 | RRT通过/关联端点 | 样条通过/关联端点 |',
            '|---|---|---|---|---:|---:|---:|---:|---:|---:|']
        for r in rows:
            bounds = [' – '.join(f'{v:.3f}' for v in r['bounds'][a]) for a in 'xyz']
            lines.append('| ' + ' | '.join(map(str, [r['cell'], *bounds, r['tcp_samples'],
                r['ik_solved'], r['collision_valid'], r['unique_rrt_endpoints'],
                f"{r['rrt_raw_success']}/{r['rrt_endpoint_incidence']}",
                f"{r['rrt_spline_success']}/{r['rrt_endpoint_incidence']}"])) + ' |')
        lines += ['']
    (destination / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(cells=len(report['cells']), tcp_samples=sum(r['tcp_samples'] for r in report['cells']),
        pairs=len(report['transitions']), outside_fk_endpoints=sum(e['cell'] is None for e in report['endpoints']),
        report=str(destination / 'REPORT.md')), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    write_report(args.input_dir, args.output_dir or args.input_dir / 'spatial-cells')

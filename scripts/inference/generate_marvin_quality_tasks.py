"""Fresh, bounded region sampling for the inference quality benchmark."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGIONS = ROOT / 'data_generation_cfgs/EnvWarehouse-RobotMarvinBimanual-independent.yaml'


def scenarios():
    table = dict(left='left_table', right='right_table')
    result = {'easy_table': dict(tier='easy', start=table, goal=table)}
    for arm in ('left', 'right'):
        for tier, suffix in [('medium', 'cabinet'), ('hard', 'cabinet_upper'),
                             ('hard', 'table_cross_x'), ('extreme', 'table_cross_y'),
                             ('extreme', 'shelf_upper_yedge')]:
            goal = dict(table, **{arm: f'{arm}_{suffix}'})
            result[f'{tier}_{arm}_{suffix}'] = dict(tier=tier, start=table, goal=goal)
    result['hard_dual_shelf'] = dict(tier='hard', start=table,
                                   goal=dict(left='left_cabinet', right='right_cabinet'))
    order = {'easy': 0, 'medium': 1, 'hard': 2, 'extreme': 3}
    return dict(sorted(result.items(), key=lambda item: order[item[1]['tier']]))


def generate(output, regions_path, tiers, count, attempts, timeout):
    from scripts.inference.benchmark_marvin_inference_quality import immutable_json
    from scripts.inference.benchmark_marvin_workspace_generalization import (
        _sample_mapping, _request_for_pair, _actual_pose, _write_json)
    from scripts.generate_data.generate_marvin_warehouse_bimanual import MarvinWarehouseGenerator
    from scripts.inference.benchmark_marvin_workspace_edges import edge_regions
    from copy import deepcopy
    config = yaml.safe_load(regions_path.read_text())
    for name, region in edge_regions().items():
        if name.endswith('shelf_upper_yedge'):
            config['placement_regions'][name] = deepcopy(config['placement_regions'][region['base_region']])
            config['placement_regions'][name]['translation'] = region['translation']
    selected = {k: v for k, v in scenarios().items() if v['tier'] in tiers}
    settings = dict(regions=config, scenarios=selected, tasks_per_scenario=count,
                    max_attempts_per_scenario=attempts, attempt_timeout_s=timeout)
    immutable_json(output / 'settings.json', settings)
    report_path = output / 'tasks.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else {
        'tasks': [], 'scenarios': {}, 'sampling': 'system_entropy', 'complete': False}
    if report['complete']:
        return
    generator = MarvinWarehouseGenerator(config, None, progress_label='quality-tasks')
    try:
        for name, spec in selected.items():
            state = report['scenarios'].setdefault(name, dict(attempts=0, accepted=0))
            seen = {tuple(t['request']['q_start'] + t['request']['q_goal']) for t in report['tasks']}
            while state['accepted'] < count and state['attempts'] < attempts:
                state['attempts'] += 1
                generator.deadline = time.perf_counter() + timeout
                reference = generator._random_valid_state()
                start = None if reference is None else _sample_mapping(generator, reference, spec['start'])
                goal = None if start is None else _sample_mapping(generator, start, spec['goal'])
                if goal is not None and all(np.linalg.norm(goal[s:s+7] - start[s:s+7]) >= .08 for s in (0, 7)):
                    key = tuple(start.tolist() + goal.tolist())
                    if key not in seen:
                        seen.add(key)
                        request = _request_for_pair(generator, name, state['accepted'],
                                                    spec, start, goal)
                        request['scene']['start_goal_source']['difficulty'] = spec['tier']
                        report['tasks'].append(dict(tier=spec['tier'], scenario=name, request=request,
                            tcp_start={arm: _actual_pose(generator, start, arm)[:, 3].tolist() for arm in ('left', 'right')},
                            tcp_goal={arm: _actual_pose(generator, goal, arm)[:, 3].tolist() for arm in ('left', 'right')}))
                        state['accepted'] += 1
                state['shortfall'] = count - state['accepted']
                report['sampling_stats'] = dict(generator.stats)
                _write_json(report_path, report)
                print(f'{name}: {state["accepted"]}/{count}, attempts={state["attempts"]}/{attempts}', flush=True)
        report['complete'] = True
        _write_json(report_path, report)
    finally:
        generator.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--regions', type=Path, default=DEFAULT_REGIONS)
    parser.add_argument('--tiers', nargs='+', required=True)
    parser.add_argument('--count', type=int, required=True)
    parser.add_argument('--attempts', type=int, required=True)
    parser.add_argument('--timeout', type=float, required=True)
    args = parser.parse_args()
    generate(args.output_dir, args.regions, args.tiers, args.count, args.attempts, args.timeout)


if __name__ == '__main__':
    main()

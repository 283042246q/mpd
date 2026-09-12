from collections import Counter

import numpy as np

from scripts.inference.benchmark_marvin_workspace_edges import edge_regions, select_pairs


def test_distinct_goals_precede_repeats_and_each_pair_is_unique():
    records = [dict(collision_valid=True, target_index=i, q_goal=[i + 20.] * 14) for i in range(40)]
    starts = [[float(i)] * 14 for i in range(10)]
    pairs = select_pairs(records, starts, 100, 5, np.random.default_rng(0))
    assert len(pairs) == 100
    assert len({r['target_index'] for r in pairs[:40]}) == 40
    assert len({(r['target_index'], r['start_index']) for r in pairs}) == 100
    assert max(Counter(r['target_index'] for r in pairs).values()) <= 5


def test_insufficient_or_duplicate_goals_do_not_inflate_pair_count():
    records = [dict(collision_valid=True, target_index=i, q_goal=[20.] * 14) for i in range(10)]
    records.append(dict(collision_valid=False, target_index=10))
    starts = [[float(i)] * 14 for i in range(8)]
    pairs = select_pairs(records, starts, 100, 5, np.random.default_rng(0))
    assert len(pairs) == 5
    assert len({r['start_index'] for r in pairs}) == 5
    assert select_pairs([], starts, 100, 5, np.random.default_rng(0)) == []


def test_twelve_edge_boxes_cover_both_arms_and_crossing_signs():
    regions = edge_regions()
    assert len(regions) == 12
    assert sum(s['arm'] == 'left' for s in regions.values()) == 6
    for name, spec in regions.items():
        for intervals in spec['translation'].values():
            assert len(intervals) == 1 and intervals[0][0] < intervals[0][1]
        lo, hi = spec['translation']['y'][0]
        positive = (spec['arm'] == 'left') != ('cross' in name)
        assert lo >= 0 if positive else hi <= 0


def test_rrt_slot_is_exclusive_for_independent_callers(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    import time
    from scripts.inference.benchmark_marvin_workspace_edges import rrt_slot
    barrier = threading.Barrier(3)
    state = {'active': 0, 'peak': 0}
    def attempt(_):
        barrier.wait()
        with rrt_slot(tmp_path) as slot:
            assert slot == 0
            state['active'] += 1
            state['peak'] = max(state['peak'], state['active'])
            time.sleep(.02)
            state['active'] -= 1
    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(attempt, range(3)))
    assert state['peak'] == 1


def test_recovery_preserves_good_targets_and_archives_empty_records(tmp_path):
    from scripts.inference.benchmark_marvin_workspace_edges import save, read, recover_corrupt_records
    save(tmp_path/'settings.json', {'regions': {'cell': {}}, 'targets_per_cell': 2})
    target = {'position': [.4, .2, .1], 'rotation': np.eye(3).tolist(), 'reference_index': 0}
    save(tmp_path/'cell/ik/0000.json', {'target': target})
    damaged = tmp_path/'cell/ik/0001.json'
    damaged.touch()
    moved = recover_corrupt_records(tmp_path)
    assert moved == ['cell/ik/0001.json']
    assert read(tmp_path/'cell/targets/0000.json') == target
    assert (tmp_path/'cell/ik/0000.json').exists()
    assert not damaged.exists()
    assert len(list((tmp_path/'recovery').glob('*/cell/ik/0001.json'))) == 1

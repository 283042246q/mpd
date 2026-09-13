from copy import deepcopy
import json

import numpy as np
import yaml

from scripts.inference.benchmark_marvin_inference_quality import (
    DEFAULT_CONFIG, build_cases, select_checkpoints, summarize,
)


def test_four_batches_keep_candidate_batch_32_and_enable_a5():
    base = yaml.safe_load(DEFAULT_CONFIG.read_text())
    assert base['collision_optimization']['pair_streaming']['enabled']
    assert base['collision_optimization']['reduced_guide_geometry']['enabled']
    assert base['gradient_pruning']['spatial']['link_broad_phase']['enabled']
    assert base['dense_validation']['chunking'] == {
        'enabled': True, 'candidate_chunk_size': 8,
        'time_chunk_size': 32, 'self_pair_chunk_size': 1024,
    }
    before = deepcopy(base)
    cases = {name: (config, batches) for name, config, batches in build_cases(base)}
    config, batches = cases['candidates-4x32']
    assert batches == 4 and config['n_trajectory_samples'] == 32
    assert config['dense_validation']['chunking']['enabled']
    assert config['collision_optimization']['reduced_guide_geometry']['enabled']
    assert base == before
    assert cases['ee-late-4x'][0]['ddim']['ee_goal_position_weight_end'] == 4.0


def test_checkpoint_selection_matches_saved_step_not_loss_array_index(tmp_path):
    (tmp_path / 'args.yaml').write_text('use_ema: true\n')
    folder = tmp_path / 'checkpoints'
    folder.mkdir()
    np.save(folder / 'val_losses.npy', np.array([
        (100, {'VALIDATION total_loss': 0.4}),
        (500, {'VALIDATION total_loss': 0.1}),
        (900, {'VALIDATION total_loss': 0.2}),
    ], dtype=object))
    for step in [100, 900]:
        (folder / f'ema_model__iter_{step:06d}.pth').touch()
    _, selected = select_checkpoints(tmp_path, 3)
    assert [item['step'] for item in selected] == [900, 100]


def test_four_batch_success_counts_once_and_adds_time():
    rows = [dict(case='4x32', request_index=0, seed=1, batches=4,
                 status='success' if batch == 2 else 'no_valid_trajectory',
                 wall_seconds=2.0) for batch in range(4)]
    assert summarize(rows[:3])['4x32']['success_rate'] is None
    result = summarize(rows)['4x32']
    assert result['successful_groups'] == 1
    assert result['mean_total_wall_s'] == 8.0


def test_tier_summary_keeps_failures_in_their_own_denominator():
    from scripts.inference.benchmark_marvin_inference_quality import summarize_by_tier
    rows = [dict(case='baseline', request_index=i, seed=1, batches=1,
                 status=status, wall_seconds=2, tier=tier)
            for i, (tier, status) in enumerate([('easy', 'success'), ('hard', 'oom')])]
    result = summarize_by_tier(rows)
    assert result['easy']['baseline']['success_rate'] == 1
    assert result['hard']['baseline']['success_rate'] == 0


def test_sampling_shortfalls_are_bounded_and_resume_does_not_resample(tmp_path, monkeypatch):
    from scripts.inference.generate_marvin_quality_tasks import generate, DEFAULT_REGIONS
    import scripts.generate_data.generate_marvin_warehouse_bimanual as generation

    instances = []
    class EmptySampler:
        def __init__(self, *args, **kwargs):
            self.stats = {}
            self.calls = 0
            instances.append(self)

        def _random_valid_state(self):
            self.calls += 1
            return None

        def close(self):
            self.closed = True

    monkeypatch.setattr(generation, 'MarvinWarehouseGenerator', EmptySampler)
    generate(tmp_path, DEFAULT_REGIONS, ['easy', 'extreme'], 1, 2, 1)
    report = json.loads((tmp_path / 'tasks.json').read_text())
    assert report['complete'] and report['tasks'] == []
    assert len(report['scenarios']) == 5
    assert all(item == dict(attempts=2, accepted=0, shortfall=1)
               for item in report['scenarios'].values())
    assert instances[0].calls == 10 and instances[0].closed
    generate(tmp_path, DEFAULT_REGIONS, ['easy', 'extreme'], 1, 2, 1)
    assert len(instances) == 1


def test_one_click_runs_all_four_batches_and_resumes(tmp_path, monkeypatch):
    import scripts.inference.benchmark_marvin_inference_quality as benchmark
    request = tmp_path / 'request.json'
    request.write_text(json.dumps({'request_id': 'test', 'seed': 9}))
    calls = []
    def execute(config, artifact, args):
        calls.append(json.loads(args.request.read_text())['seed'])
        return dict(status='success', wall_seconds=1)
    monkeypatch.setattr(benchmark, 'execute_case', execute)
    monkeypatch.setattr(benchmark, 'model_fingerprints', lambda cases: {})
    argv = ['--output-dir', str(tmp_path / 'experiment'), '--requests', str(request),
            '--skip-checkpoints', '--cases', 'candidates-4x32', '--seeds', '100', '--device', 'cpu']
    assert benchmark.main(argv) == 0
    assert calls == [100, 101, 102, 103]
    assert benchmark.main(argv) == 0
    assert calls == [100, 101, 102, 103]
    report = json.loads((tmp_path / 'experiment/summary-by-tier.json').read_text())
    assert report['external']['candidates-4x32']['successful_groups'] == 1

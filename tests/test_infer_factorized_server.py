from scripts.runtime import infer_factorized_server


def test_factorized_server_routes_runtime_options(tmp_path, monkeypatch):
    captured = {}

    class FakeEngine:
        def __init__(self, **kwargs):
            captured["engine"] = kwargs

    class FakeService:
        def __init__(self, socket, output_root, factory, **kwargs):
            captured["service"] = (socket, output_root, kwargs)
            factory(lambda *_: None)

        def serve_forever(self):
            captured["served"] = True

    monkeypatch.setattr(infer_factorized_server, "FactorizedMpdRuntimeEngine", FakeEngine)
    monkeypatch.setattr(infer_factorized_server, "DynamicResidentPlannerService", FakeService)
    checkpoint = tmp_path / "timing.pt"
    assert infer_factorized_server.main(
        [
            "--socket",
            str(tmp_path / "worker.sock"),
            "--output-root",
            str(tmp_path / "results"),
            "--timing-checkpoint",
            str(checkpoint),
            "--method",
            "f3",
            "--adapt-spatial-basis",
            "--alternating-rounds",
            "2",
            "--timing-steps-per-space-step",
            "2",
        ]
    ) == 0
    assert captured["served"] is True
    assert captured["service"][2]["trajectory_compression"] is False
    engine = captured["engine"]
    assert engine["timing_checkpoint"] == checkpoint
    assert engine["adapt_spatial_basis"] is True
    assert engine["factorized_settings"]["method"] == "f3"
    assert engine["factorized_settings"]["alternating_rounds"] == 2
    assert engine["factorized_settings"]["timing_steps_per_space_step"] == 2
    assert engine["space_time_settings"]["num_timing_control_points"] == 8
    assert engine["space_time_settings"]["duration_min"] == 2.0

import json

import pytest

from scripts.inference import infer_factorized
from scripts.inference.infer_space_time import _build_parser as old_parser
from test_infer_space_time import FakeEngine


@pytest.mark.parametrize("method", ["f1", "f2", "f3"])
def test_new_cli_routes_factorized_options_and_preserves_request_export(tmp_path, monkeypatch, method):
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"seed": 12}))
    output = tmp_path / method
    monkeypatch.setattr(infer_factorized, "FactorizedMpdRuntimeEngine", FakeEngine)
    code = infer_factorized.main(["--request", str(request), "--output-dir", str(output),
        "--method", method, "--timing-checkpoint", str(tmp_path / "timing.pt"), "--adapt-spatial-basis"])
    assert code == 0
    engine = FakeEngine.instances[-1]
    assert engine.options["adapt_spatial_basis"]
    assert engine.options["factorized_settings"]["method"] == method
    assert engine.options["space_time_settings"]["duration_min"] == 2.
    summary = json.loads((output / "summary.json").read_text())
    assert summary["schema"] == "mpd_factorized_inference"
    assert summary["timing_mode"] == method
    assert (output / "run-0000/trajectory.npz").is_file()
    # No new timing modes have been inserted into the old public interface.
    args = old_parser().parse_args(["--request", str(request), "--output-dir", str(output)])
    assert args.timing_mode == "phase5_joint" and args.duration_min == 6.


def test_cli_failure_has_explicit_json_and_nonzero_exit(tmp_path, monkeypatch):
    request = tmp_path / "request.json"
    request.write_text("{}")
    def fail(**kwargs):
        raise ValueError("timing checkpoint contains NaN/Inf")
    monkeypatch.setattr(infer_factorized, "FactorizedMpdRuntimeEngine", fail)
    output = tmp_path / "failure"
    assert infer_factorized.main(["--request", str(request), "--output-dir", str(output),
        "--timing-checkpoint", "corrupt.pt"]) == 1
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "inference_error"
    assert "NaN/Inf" in summary["error"]["message"]

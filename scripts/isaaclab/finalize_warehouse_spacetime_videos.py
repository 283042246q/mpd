"""Encode browser-compatible H.264 clips and audit every decoded video frame."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np

try:
    from .warehouse_spacetime_payload import validate_suite
except ImportError:
    from warehouse_spacetime_payload import validate_suite


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".inprogress")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def finalize(suite_path, output):
    suite = json.loads(Path(suite_path).read_text())
    validate_suite(suite)
    output = Path(output)
    rows = []
    for case in suite["cases"]:
        stats_path = output / (case["case_id"] + ".json")
        row = json.loads(stats_path.read_text())
        video = Path(row["video"])
        if not row["complete"] or not video.is_file():
            raise ValueError("incomplete case: " + case["case_id"])
        if row.get("codec") != "h264":
            encoded = video.with_name(video.stem + ".h264.inprogress.mp4")
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                            "-i", str(video), "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                            "-threads", "4", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(encoded)], check=True)
            os.replace(encoded, video)
        capture = cv2.VideoCapture(str(video))
        decoded, spread = 0, []
        while True:
            success, frame = capture.read()
            if not success:
                break
            if frame.shape[:2] != (row["identity"]["height"], row["identity"]["width"]):
                raise ValueError("video resolution changed")
            # Exclude the HUD: a uniform/black scene cannot pass this audit.
            region = frame[int(frame.shape[0] * .18):int(frame.shape[0] * .88)]
            spread.append(float(np.std(region)))
            decoded += 1
        fps = capture.get(cv2.CAP_PROP_FPS)
        capture.release()
        if decoded != row["frames"] or abs(fps - row["identity"]["fps"]) > .01 or min(spread, default=0) < 2:
            raise ValueError("bad encoded frame stream: " + case["case_id"])
        row.update(codec="h264", bytes=video.stat().st_size, decoded_frames=decoded,
                   sha256=hashlib.sha256(video.read_bytes()).hexdigest(), scene_frame_std_min=min(spread))
        atomic_json(stats_path, row)
        rows.append(row)
        print("[warehouse-video-audit] %s: %d frames, H.264" % (case["case_id"], decoded), flush=True)
    if len({row["sha256"] for row in rows}) != len(rows):
        raise ValueError("duplicate video files")
    counts = {mode: sum(row["mode"] == mode for row in rows) for mode in suite["modes"]}
    atomic_json(output / "report.json", {"expected": suite["expected_cases"], "completed": len(rows),
                                         "mode_counts": counts, "all_frames_decoded": True,
                                         "codec": "h264", "videos": rows})
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    return finalize(args.suite, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())

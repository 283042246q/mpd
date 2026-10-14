"""Render a saved trajectory and its prescribed obstacles in the Warehouse."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from scripts.spacetime_data.validate_warehouse_spacetime import read_forecast
from scripts.spacetime_data.warehouse_world import WarehouseCollision


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    index = json.loads((args.root / "index.json").read_text())
    entries = sorted([e for samples in index["splits"].values() for e in samples], key=lambda e: e["attempt_id"])
    if not 0 <= args.sample < len(entries):
        parser.error("sample index out of range")
    entry = entries[args.sample]
    with h5py.File(args.root / entry["file"], "r") as shard:
        sample = shard[entry["group"]]
        forecast = read_forecast(sample)
        times, q = np.asarray(sample["execution/time"]), np.asarray(sample["execution/q"])
    torch.set_num_threads(1)
    world = WarehouseCollision("cpu")
    robot = world.robot
    ee = robot.get_EE_pose(torch.as_tensor(q[::8], dtype=torch.float32))[:, :3, 3].cpu().numpy()
    event = (float(forecast.event_times[len(forecast.event_times) // 2])
             if len(forecast.event_times) else float(times[-1]) / 2)
    frame_times = [0., event, float(times[-1])]
    fig = plt.figure(figsize=(16, 5))
    colors = ["#d1495b", "#00798c", "#edae49", "#7b2cbf", "#3a7d44"]
    for column, instant in enumerate(frame_times, 1):
        ax = fig.add_subplot(1, 3, column, projection="3d")
        world.environment.render(ax)
        robot.render(ax, q=torch.as_tensor(q[np.argmin(abs(times - instant))], dtype=torch.float32), color="#204d8c")
        ax.plot(*ee.T, color="#204d8c", linewidth=1.8, label="EE path")
        for obj, mode in enumerate(forecast.modes):
            color = colors[obj % len(colors)]
            path = forecast.positions[forecast.times <= times[-1], obj]
            center = np.array([np.interp(instant, forecast.times, forecast.positions[:, obj, axis]) for axis in range(3)])
            ax.plot(*path.T, color=color, linewidth=1, alpha=0.6)
            size, shape = forecast.size[obj], forecast.shape[obj]
            if shape == 1:
                signs = np.array([[-1,-1,-1], [-1,-1,1], [-1,1,-1], [-1,1,1],
                                  [1,-1,-1], [1,-1,1], [1,1,-1], [1,1,1]])
                vertices = center + signs * size
                faces = [[vertices[i] for i in face] for face in
                         ([0,1,3,2], [4,5,7,6], [0,1,5,4], [2,3,7,6], [0,2,6,4], [1,3,7,5])]
                ax.add_collection3d(Poly3DCollection(faces, facecolors=color, edgecolors=color, alpha=0.3))
            else:
                azimuth = np.linspace(0, 2 * np.pi, 16)
                polar = np.linspace(0, np.pi, 10)
                x = center[0] + size[0] * np.outer(np.cos(azimuth), np.sin(polar))
                y = center[1] + size[0] * np.outer(np.sin(azimuth), np.sin(polar))
                z = size[0] * np.outer(np.ones_like(azimuth), np.cos(polar))
                if shape == 2:
                    z += np.sign(z) * size[1]
                ax.plot_wireframe(x, y, z + center[2], color=color, linewidth=0.5)
            ax.scatter(*forecast.anchors[obj], color=color, marker="x", s=18)
        ax.set(xlim=(-0.2, 1.1), ylim=(-1., 1.), zlim=(-0.1, 1.2), title="t = %.2f s" % instant,
               xlabel="x (m)", ylabel="y (m)", zlabel="z (m)")
        ax.view_init(elev=27, azim=-52)
    fig.suptitle("Warehouse / %s / %s" % (entry["attempt_id"], ", ".join(forecast.modes) or "no dynamic obstacles"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

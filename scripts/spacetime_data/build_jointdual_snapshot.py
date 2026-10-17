"""Freeze converted P21/EE data and train-only statistics for JointDual."""
import argparse
from mpd.datasets.warehouse_joint_dataset import build_snapshot


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", required=True, action="append")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    snapshot = build_snapshot(args.dataset_root, args.output)
    print({"identity": snapshot["identity"], "splits": snapshot["split_counts"],
           "rejected": len(snapshot["rejected"])})


if __name__ == "__main__":
    main()

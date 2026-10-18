"""Build the shared, immutable P/z/task/world tensor cache once per snapshot."""
import argparse
from mpd.datasets.warehouse_joint_cache import build_tensor_cache


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--output", help="Default: snapshot directory/.jointdual_tensor_cache/<identity>")
    args = parser.parse_args(argv)
    print(build_tensor_cache(args.snapshot, args.output))


if __name__ == "__main__":
    main()

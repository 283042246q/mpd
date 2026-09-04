#!/usr/bin/env python3
"""Line-delimited JSON dynamic Marvin worker."""
import argparse
import json
import sys
from mpd.bimanual.runtime_contract import BimanualRequest
from scripts.inference.inference_marvin_bimanual import plan


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = BimanualRequest.from_dict(json.loads(line))
            print(json.dumps(plan(request), separators=(",", ":")), flush=True)
        except Exception as error:
            print(json.dumps({"schema": "marvin_bimanual_result/v1", "status": "error", "error": str(error)}), flush=True)


if __name__ == "__main__":
    main()

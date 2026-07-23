import os
import subprocess
import sys
import time
import unittest

from scripts.isaaclab.subprocess_utils import _process_group_exists, _terminate_process


class IsaacLabSubprocessCleanupTest(unittest.TestCase):
    def test_terminate_process_stops_entire_process_group(self):
        child_code = (
            "import subprocess, sys, time; "
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
            "time.sleep(60)"
        )
        process = subprocess.Popen([sys.executable, "-c", child_code], start_new_session=True)

        try:
            time.sleep(0.2)
            self.assertTrue(_process_group_exists(process.pid))
            returncode = _terminate_process(process, terminate_timeout_s=1, kill_timeout_s=1)

            self.assertIsNotNone(returncode)
            self.assertFalse(_process_group_exists(process.pid))
        finally:
            if _process_group_exists(process.pid):
                os.killpg(process.pid, 9)
            if process.poll() is None:
                process.wait(timeout=1)


if __name__ == "__main__":
    unittest.main()

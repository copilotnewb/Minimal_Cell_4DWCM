import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / 'tools' / 'benchmark_run.py'


class BenchmarkRunTests(unittest.TestCase):
    def test_gpu_sampling_reports_each_device_without_combining_vram(self):
        with tempfile.TemporaryDirectory(prefix='gpu sampler ') as directory:
            root = Path(directory)
            executable = root / 'nvidia-smi'
            executable.write_text('#!' + sys.executable + '\n'
                                  'print("0, GPU-one, Sample GPU, 16384, 4096, 73")\n'
                                  'print("1, GPU-two, Second GPU, 8192, 1024, N/A")\n')
            executable.chmod(0o755)
            report = root / 'result.json'
            run = subprocess.run([sys.executable, str(SCRIPT), '--output', str(report),
                                  '--sample-interval', '.02', '--', sys.executable, '-c',
                                  'import time; time.sleep(.15)'], capture_output=True, text=True,
                                 env=dict(os.environ, PATH=str(root)))
            self.assertEqual(run.returncode, 0, run.stderr)
            devices = json.loads(report.read_text())['gpu_sampling']['devices']
            self.assertEqual(devices['GPU-one']['peak_memory_used_mib'], 4096)
            self.assertEqual(devices['GPU-two']['peak_memory_used_mib'], 1024)
            self.assertIsNone(devices['GPU-two']['peak_utilization_percent'])
            self.assertGreater(devices['GPU-one']['samples'], 0)
            self.assertTrue(report.with_suffix('.gpu.csv').exists())

    def test_records_real_command_output_stages_and_exit_status_without_gpu(self):
        with tempfile.TemporaryDirectory(prefix='benchmark run ') as directory:
            report = Path(directory) / 'result.json'
            code = ('print("Current biological time: 0.0"); '
                    'print("ODE time: 0.5"); print("ODE time: 1.5"); '
                    'print("Current biological time: 4.0"); raise SystemExit(7)')
            # Force the documented no-nvidia-smi path; Python uses an absolute path.
            env = dict(os.environ, PATH='')
            run = subprocess.run([sys.executable, str(SCRIPT), '--output', str(report), '--',
                                  sys.executable, '-c', code], capture_output=True, text=True, env=env)
            self.assertEqual(run.returncode, 7, run.stderr)
            result = json.loads(report.read_text())
            self.assertEqual(result['returncode'], 7)
            self.assertEqual(result['biological_seconds_observed'], 4.0)
            self.assertEqual(result['stages']['ode']['calls'], 2)
            self.assertEqual(result['stages']['ode']['total_seconds'], 2.0)
            self.assertEqual(result['stages']['ode']['mean_seconds'], 1.0)
            self.assertEqual(result['gpu_sampling']['available'], False)
            self.assertGreater(result['wall_seconds'], 0)
            self.assertIn('ODE time: 1.5', report.with_suffix('.log').read_text())

    def test_missing_command_is_a_recorded_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / 'result.json'
            run = subprocess.run([sys.executable, str(SCRIPT), '--output', str(report), '--',
                                  '/nonexistent/mc4d-executable'], capture_output=True, text=True)
            self.assertNotEqual(run.returncode, 0)
            result = json.loads(report.read_text())
            self.assertIsNotNone(result['launch_error'])


if __name__ == '__main__':
    unittest.main()

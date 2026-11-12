#!/usr/bin/env python3
"""Record wall time, existing hook timings, and sampled GPU memory for a command.

Example (run from the model directory):
  python tools/benchmark_run.py --output benchmarks/run.json -- \
      python Whole_Cell_Minimal_Cell.py -od perf1 -t 20 -cd 0 -dsd /path/to/software

GPU memory is device-wide and sampled, not an allocator-exact peak. Keep other
GPU workloads idle for a useful comparison. This wrapper never changes physics.
"""
import argparse
import csv
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import threading
import time


STAGES = {
    'Time between hook steps': 'rdme_between_hooks',
    'DNA time': 'dna',
    'Ribo time': 'ribosomes',
    'CME time': 'cme',
    'ODE time': 'ode',
    'Communication and file write time': 'checkpoint',
}
TIMING = re.compile(r'^(' + '|'.join(re.escape(key) for key in STAGES) + r'):\s*([0-9.eE+-]+)\s*$')
BIOLOGICAL_TIME = re.compile(r'^Current biological time:\s*([0-9.eE+-]+)\s*$')


def numeric(value):
    try:
        result = float(value.strip())
        return result if math.isfinite(result) else None
    except ValueError:
        return None


class GpuSampler(threading.Thread):
    def __init__(self, path, interval):
        super().__init__(daemon=True)
        self.path = path
        self.interval = interval
        self.stop_event = threading.Event()
        self.executable = shutil.which('nvidia-smi')
        self.devices = {}
        self.error = None

    def run(self):
        if self.executable is None:
            return
        with self.path.open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['unix_seconds', 'gpu_index', 'uuid', 'name', 'memory_total_mib', 'memory_used_mib', 'utilization_percent'])
            while not self.stop_event.is_set():
                try:
                    result = subprocess.run(
                        [self.executable, '--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu',
                         '--format=csv,noheader,nounits'],
                        capture_output=True, text=True, check=True, timeout=5)
                    timestamp = time.time()
                    for row in csv.reader(io.StringIO(result.stdout)):
                        if len(row) != 6:
                            continue
                        index, uuid, name = [value.strip() for value in row[:3]]
                        total, used, utilization = map(numeric, row[3:])
                        writer.writerow([timestamp, index, uuid, name, total, used, utilization])
                        record = self.devices.setdefault(uuid, {'index': index, 'name': name,
                            'memory_total_mib': total, 'first_memory_used_mib': used,
                            'peak_memory_used_mib': used, 'peak_utilization_percent': utilization, 'samples': 0})
                        record['samples'] += 1
                        if used is not None:
                            previous = record['peak_memory_used_mib']
                            record['peak_memory_used_mib'] = used if previous is None else max(previous, used)
                        if utilization is not None:
                            previous = record['peak_utilization_percent']
                            record['peak_utilization_percent'] = utilization if previous is None else max(previous, utilization)
                    handle.flush()
                except (OSError, subprocess.SubprocessError) as error:
                    self.error = str(error)
                    return
                self.stop_event.wait(self.interval)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True, help='JSON report; sibling .log and .gpu.csv contain raw evidence.')
    parser.add_argument('--sample-interval', type=float, default=1.0, help='GPU sampling interval in seconds (default: 1).')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('Supply a simulation command after --.')
    if not math.isfinite(args.sample_interval) or args.sample_interval <= 0:
        parser.error('--sample-interval must be finite and positive.')
    if args.output.suffix.lower() != '.json':
        parser.error('--output must end in .json.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.output.with_suffix('.log')
    gpu_path = args.output.with_suffix('.gpu.csv')
    sampler = GpuSampler(gpu_path, args.sample_interval)
    start_utc = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    sampler.start()
    stages = {}
    first_bio = last_bio = None
    launch_error = None
    returncode = 127
    process = None
    environment = dict(os.environ)
    environment.setdefault('PYTHONUNBUFFERED', '1')
    try:
        with log_path.open('w') as log:
            try:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, errors='replace', bufsize=1, env=environment)
                for line in process.stdout:
                    log.write(line)
                    sys.stdout.write(line)
                    match = TIMING.match(line.strip())
                    if match:
                        value = float(match.group(2))
                        if math.isfinite(value):
                            record = stages.setdefault(STAGES[match.group(1)], {'calls': 0, 'total_seconds': 0., 'max_seconds': 0.})
                            record['calls'] += 1
                            record['total_seconds'] += value
                            record['max_seconds'] = max(record['max_seconds'], value)
                    match = BIOLOGICAL_TIME.match(line.strip())
                    if match:
                        value = float(match.group(1))
                        if math.isfinite(value):
                            first_bio = value if first_bio is None else min(first_bio, value)
                            last_bio = value if last_bio is None else max(last_bio, value)
                returncode = process.wait()
            except OSError as error:
                launch_error = str(error)
                log.write(launch_error + '\n')
            except KeyboardInterrupt:
                if process is not None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                returncode = 130
    finally:
        elapsed = time.perf_counter() - start
        sampler.stop_event.set()
        sampler.join(timeout=6)
        for record in stages.values():
            record['mean_seconds'] = record['total_seconds'] / record['calls']
        duration = None if first_bio is None else last_bio - first_bio
        report = {
            'command': command, 'working_directory': os.getcwd(), 'started_utc': start_utc,
            'python': sys.version, 'platform': platform.platform(), 'returncode': returncode,
            'launch_error': launch_error, 'wall_seconds': elapsed,
            'biological_seconds_observed': duration,
            'wall_seconds_per_biological_second': elapsed / duration if duration else None,
            'stages': stages, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'gpu_sampling': {'available': sampler.executable is not None, 'error': sampler.error,
                             'interval_seconds': args.sample_interval, 'devices': sampler.devices,
                             'scope': 'device-wide sampled usage; includes other processes and can miss brief peaks'},
            'log_file': str(log_path), 'gpu_csv': str(gpu_path) if sampler.executable else None,
        }
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print('\nBenchmark report: {}'.format(args.output))
    return returncode if returncode >= 0 else 128 - returncode


if __name__ == '__main__':
    sys.exit(main())

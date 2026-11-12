"""Own chromosome workers outside the serializable simulation state.

Lattice Microbes and LAMMPS use separate CUDA contexts. Pinning the child to a
single visible device makes its hard-coded Kokkos device zero refer to the
chosen GPU, including inside a scheduler's CUDA_VISIBLE_DEVICES namespace.
"""
import math
import os
from functools import wraps
from pathlib import Path
import subprocess


_jobs = {}


class SimulationTimeLimit(RuntimeError):
    """Controlled wall-time stop that must preserve pending checkpoint DNA."""


def addArguments(parser):
    parser.add_argument('--dnaCudaDevice', type=int, default=None,
                        help='DNA GPU ordinal in the parent visible-device list (default: --cudaDevices).')
    parser.add_argument('--dnaMode', choices=('overlap', 'serial'), default='overlap',
                        help='Overlap chromosome and RDME work, or wait for DNA before resuming RDME.')
    parser.add_argument('--dnaTimeout', type=float, default=300.0,
                        help='Maximum seconds to wait for a chromosome worker at a synchronization point.')


def configure(sim_properties, args):
    device = args.cudaDevices if args.dnaCudaDevice is None else args.dnaCudaDevice
    if device < 0 or args.cudaDevices < 0:
        raise ValueError('GPU ordinals must be nonnegative.')
    if not math.isfinite(args.dnaTimeout) or args.dnaTimeout <= 0:
        raise ValueError('--dnaTimeout must be finite and positive.')
    sim_properties.update(rdme_cuda_device=args.cudaDevices, dna_cuda_device=device, dna_mode=args.dnaMode,
                          dna_timeout=args.dnaTimeout)
    # Validate scheduler mapping before constructing the large simulation.
    dnaEnvironment(sim_properties)


def dnaEnvironment(sim_properties, environ=None):
    env = dict(os.environ if environ is None else environ)
    device = sim_properties.get('dna_cuda_device')
    if device is None:
        return env  # Compatibility for standalone callers without CLI settings.
    if device < 0:
        raise ValueError('DNA GPU ordinal must be nonnegative.')
    if 'CUDA_VISIBLE_DEVICES' in env:
        visible = [value.strip() for value in env['CUDA_VISIBLE_DEVICES'].split(',') if value.strip()]
        if device >= len(visible) or visible[device] == '-1':
            raise ValueError('DNA GPU ordinal {} is unavailable in CUDA_VISIBLE_DEVICES={!r}.'.format(device, env['CUDA_VISIBLE_DEVICES']))
        env['CUDA_VISIBLE_DEVICES'] = visible[device]
    else:
        env['CUDA_VISIBLE_DEVICES'] = str(device)
    return env


def btreeDirectory(sim_properties):
    root = Path(sim_properties['dna_software_directory']).expanduser().resolve()
    for candidate in (root / 'btree_chromo_gpu', root / 'btree_chromo', root):
        if (candidate / 'build/apps/btree_chromo').is_file():
            return candidate
    raise FileNotFoundError('Build btree_chromo_gpu (or btree_chromo) under {} first; expected build/apps/btree_chromo.'.format(root))


def chainExecutable(sim_properties):
    root = Path(sim_properties['dna_software_directory']).expanduser().resolve()
    for directory in ('src', 'fortran'):
        candidate = root / 'sc_chain_generation' / directory / 'gen_sc_chain'
        if candidate.is_file():
            return candidate
    raise FileNotFoundError('Build sc_chain_generation/src/gen_sc_chain under {} first.'.format(root))


def _key(sim_properties):
    return str(Path(sim_properties['working_directory']).resolve())


def hasRunningDna(sim_properties):
    job = _jobs.get(_key(sim_properties))
    return job is not None and job[0].poll() is None


def startDna(sim_properties, command, background=True):
    """Start exactly one worker per simulation, with argv-safe paths and logging."""
    key = _key(sim_properties)
    if key in _jobs:
        # Reap/check an existing worker before replacing its handle or log.
        waitForDna(sim_properties)
    log_path = Path(key) / 'DNA' / 'worker.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('w') as log:
        process = subprocess.Popen([str(arg) for arg in command],
                                   env=dnaEnvironment(sim_properties),
                                   stdout=log, stderr=subprocess.STDOUT)
    _jobs[key] = (process, log_path)
    if not background or sim_properties.get('dna_mode') == 'serial':
        waitForDna(sim_properties)


def cancelDna(sim_properties):
    """Terminate and reap only the worker owned by this simulation."""
    job = _jobs.pop(_key(sim_properties), None)
    if job is None:
        return
    process, _ = job
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    else:
        process.wait()


def cleanupOnHookError(hook):
    """Reap DNA before LM's SWIG director turns a hook error into Py_Exit.

    The native director exits the interpreter instead of unwinding to the
    entrypoint's finally block, so cleanup must happen inside the Python hook.
    A controlled wall-time stop completes the DNA referenced by the checkpoint;
    other hook failures cancel the worker before exiting.
    """
    @wraps(hook)
    def guarded(self, *args, **kwargs):
        try:
            return hook(self, *args, **kwargs)
        except SimulationTimeLimit:
            try:
                waitForLastDna(self.sim_properties)
            except BaseException:
                cancelDna(self.sim_properties)
                raise
            raise
        except BaseException:
            cancelDna(self.sim_properties)
            raise
    return guarded


def waitForDna(sim_properties, expected_file=None, topology_file=None):
    """Wait for process completion, not the first appearance of a partial file.

    On restart there is no inherited process handle; the saved coordinate file
    must already be complete. A failed worker is an error, never a reason to
    silently substitute a previous chromosome state.
    """
    key = _key(sim_properties)
    job = _jobs.get(key)
    if job is not None:
        process, log_path = job
        timeout = float(sim_properties.get('dna_timeout', 300.0))
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as error:
            cancelDna(sim_properties)
            raise TimeoutError('Chromosome worker did not finish within {} seconds; see {}. Increase --dnaTimeout for a slow run.'.format(timeout, log_path)) from error
        _jobs.pop(key, None)
        if returncode:
            raise RuntimeError('Chromosome worker exited with status {}; see {}.'.format(returncode, log_path))
    if expected_file is not None:
        path = Path(expected_file)
        size = path.stat().st_size
        if size == 0 or size % 24:
            raise ValueError('Chromosome coordinate file {} is empty or truncated ({} bytes; expected float64 xyz triples).'.format(path, size))
        if topology_file is not None:
            with Path(topology_file).open() as topology:
                header = topology.readline().strip()
            try:
                label, count = header.split('=', 1)
                monomers = int(count)
                if label != 'size' or monomers <= 0:
                    raise ValueError('invalid monomer count')
            except ValueError as error:
                raise ValueError('Invalid chromosome topology header in {}: {!r}.'.format(topology_file, header)) from error
            if size != monomers * 24:
                raise ValueError('Chromosome coordinate file {} has {} bytes, but topology {} requires {} monomers ({} bytes).'.format(
                    path, size, topology_file, monomers, monomers * 24))


def waitForLastDna(sim_properties):
    """Finish and validate the complete chromosome referenced by the state."""
    step = sim_properties.get('last_DNA_step')
    if step is None:
        waitForDna(sim_properties)
    else:
        directory = Path(sim_properties['working_directory']) / 'DNA'
        waitForDna(sim_properties, directory / 'dna_monomers_{:d}.bin'.format(step),
                   directory / 'chromo_topo_{:d}.dat'.format(step))


def runBtree(sim_properties, directives, background=True):
    startDna(sim_properties, [btreeDirectory(sim_properties) / 'build/apps/btree_chromo', directives],
             background=background)

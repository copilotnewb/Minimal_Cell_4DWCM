"""Real subprocess tests; no LAMMPS or GPU is required."""
import argparse
import ast
from datetime import datetime
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import time
import types
import unittest

import DnaRuntime as runtime


class DnaRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='dna runtime ')
        self.root = Path(self.temp.name)
        (self.root / 'DNA').mkdir()
        self.state = {'working_directory': str(self.root) + os.sep,
                      'dna_software_directory': str(self.root), 'dna_timeout': 3.0}

    def tearDown(self):
        runtime.cancelDna(self.state)
        self.temp.cleanup()

    def test_default_maps_dna_to_the_rdme_device(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--cudaDevices', type=int, default=0)
        runtime.addArguments(parser)
        args = parser.parse_args(['--cudaDevices', '1'])
        runtime.configure(self.state, args)
        env = runtime.dnaEnvironment(self.state, {'CUDA_VISIBLE_DEVICES': 'GPU-first,GPU-second', 'KEEP': 'yes'})
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], 'GPU-second')
        self.assertEqual(env['KEEP'], 'yes')
        self.assertEqual(args.dnaMode, 'overlap')

    def test_explicit_device_and_scheduler_ordinals(self):
        self.state['dna_cuda_device'] = 0
        self.assertEqual(runtime.dnaEnvironment(self.state, {'CUDA_VISIBLE_DEVICES': '3,5'})['CUDA_VISIBLE_DEVICES'], '3')
        self.state['dna_cuda_device'] = 1
        self.assertEqual(runtime.dnaEnvironment(self.state, {})['CUDA_VISIBLE_DEVICES'], '1')
        with self.assertRaises(ValueError):
            runtime.dnaEnvironment(self.state, {'CUDA_VISIBLE_DEVICES': '3'})
        with self.assertRaises(ValueError):
            runtime.dnaEnvironment(self.state, {'CUDA_VISIBLE_DEVICES': ''})

    def test_waits_until_all_coordinates_written_and_keeps_checkpoints_serializable(self):
        output = self.root / 'DNA' / 'dna_monomers_250.bin'
        command = [sys.executable, '-c',
                   'import pathlib,time,sys; p=pathlib.Path(sys.argv[1]); p.write_bytes(b"x"*24); time.sleep(.15); p.write_bytes(b"x"*48)',
                   str(output)]
        runtime.startDna(self.state, command)
        pickle.dumps(self.state)  # Process handles must never enter checkpoints.
        runtime.waitForDna(self.state, output)
        self.assertEqual(output.stat().st_size, 48)
        self.assertFalse(runtime.hasRunningDna(self.state))

    def test_finished_worker_has_no_polling_delay_and_zero_exit_is_required(self):
        runtime.startDna(self.state, [sys.executable, '-c', 'raise SystemExit(7)'])
        with self.assertRaisesRegex(RuntimeError, '7'):
            runtime.waitForDna(self.state)
        self.assertFalse(runtime.hasRunningDna(self.state))

    def test_timeout_stops_child_instead_of_reusing_stale_chromosome(self):
        self.state['dna_timeout'] = .05
        runtime.startDna(self.state, [sys.executable, '-c', 'import time; time.sleep(60)'])
        with self.assertRaises(TimeoutError):
            runtime.waitForDna(self.state)
        self.assertFalse(runtime.hasRunningDna(self.state))

    def test_restart_accepts_complete_output_and_rejects_truncation(self):
        output = self.root / 'DNA' / 'dna_monomers_250.bin'
        with self.assertRaises(FileNotFoundError):
            runtime.waitForDna(self.state, output)
        output.write_bytes(b'x' * 25)
        with self.assertRaises(ValueError):
            runtime.waitForDna(self.state, output)
        output.write_bytes(b'x' * 24)
        runtime.waitForDna(self.state, output)

    def test_last_chromosome_rejects_aligned_truncation_against_topology(self):
        self.state['last_DNA_step'] = 250
        output = self.root / 'DNA' / 'dna_monomers_250.bin'
        topology = self.root / 'DNA' / 'chromo_topo_250.dat'
        topology.write_text('size=2\nm(2),1,1,1,2,2\n')
        output.write_bytes(b'x' * 24)
        with self.assertRaisesRegex(ValueError, 'topology'):
            runtime.waitForLastDna(self.state)
        output.write_bytes(b'x' * 48)
        runtime.waitForLastDna(self.state)

    def test_last_chromosome_requires_its_topology_even_without_a_live_worker(self):
        self.state['last_DNA_step'] = 250
        (self.root / 'DNA' / 'dna_monomers_250.bin').write_bytes(b'x' * 48)
        with self.assertRaises(FileNotFoundError):
            runtime.waitForLastDna(self.state)

    def test_hook_exception_reaps_real_worker_before_native_director_exits(self):
        # Load the actual hook classes without importing their GPU dependencies.
        # LM's director calls Py_Exit on an exception, bypassing caller finally.
        for filename in ('Hook.py', 'Restart_Hook.py'):
            for error in (ValueError('hook failed'), SystemExit(7)):
                with self.subTest(hook=filename, error=type(error).__name__):
                    source = Path(__file__).resolve().parents[1] / filename
                    hook_class = next(node for node in ast.parse(source.read_text()).body
                                      if isinstance(node, ast.ClassDef) and node.name == 'MyOwnSolver')
                    def fail(*args):
                        raise error
                    namespace = {'DnaRuntime': runtime, 'TIME': time,
                                 'communicate': types.SimpleNamespace(updateCountsRDME=fail)}
                    exec(compile(ast.Module(body=[hook_class], type_ignores=[]), str(source), 'exec'), namespace)
                    solver = namespace['MyOwnSolver'].__new__(namespace['MyOwnSolver'])
                    solver.sim_properties = self.state
                    solver.endLastHook = time.time()
                    solver.restart_time = 0
                    solver.rdme = None
                    runtime.startDna(self.state, [sys.executable, '-c', 'import time; time.sleep(60)'])
                    process = runtime._jobs[runtime._key(self.state)][0]
                    try:
                        with self.assertRaises(type(error)):
                            solver.hookSimulation(0, None)
                        self.assertIsNotNone(process.poll(), 'worker must be stopped before the hook exception escapes')
                        self.assertFalse(runtime.hasRunningDna(self.state))
                    finally:
                        runtime.cancelDna(self.state)

    def test_controlled_time_limit_completes_and_validates_pending_checkpoint_dna(self):
        self.state['last_DNA_step'] = 250
        output = self.root / 'DNA' / 'dna_monomers_250.bin'
        topology = self.root / 'DNA' / 'chromo_topo_250.dat'
        ready = self.root / 'worker-ready'
        topology.write_text('size=2\nm(2),1,1,1,2,2\n')
        for filename in ('Hook.py', 'Restart_Hook.py'):
            for final_size in (48, 24):
                with self.subTest(hook=filename, final_size=final_size):
                    source = Path(__file__).resolve().parents[1] / filename
                    hook_class = next(node for node in ast.parse(source.read_text()).body
                                      if isinstance(node, ast.ClassDef) and node.name == 'MyOwnSolver')
                    namespace = {'DnaRuntime': runtime, 'TIME': time, 'datetime': datetime}
                    exec(compile(ast.Module(body=[hook_class], type_ignores=[]), str(source), 'exec'), namespace)
                    solver = namespace['MyOwnSolver'].__new__(namespace['MyOwnSolver'])
                    solver.sim_properties = self.state
                    solver.endLastHook = time.time()
                    solver.restart_time = 0 if filename == 'Hook.py' else 100
                    solver.termination_time = 0
                    solver.start_datetime = datetime.now()
                    solver.final_simulation_time = solver.restart_time + 1
                    if ready.exists():
                        ready.unlink()
                    command = [sys.executable, '-c',
                               'import pathlib,sys,time; p=pathlib.Path(sys.argv[1]); '
                               'p.write_bytes(b"x"*24); pathlib.Path(sys.argv[2]).touch(); '
                               'time.sleep(.2); p.write_bytes(b"x"*int(sys.argv[3]))',
                               str(output), str(ready), str(final_size)]
                    runtime.startDna(self.state, command)
                    process = runtime._jobs[runtime._key(self.state)][0]
                    try:
                        deadline = time.monotonic() + 3
                        while not ready.exists():
                            if time.monotonic() > deadline:
                                self.fail('worker did not write its partial chromosome')
                            time.sleep(.005)
                        if final_size == 48:
                            with self.assertRaisesRegex(Exception, 'Reached end of allowed simulation time'):
                                solver.hookSimulation(1, None)
                            self.assertEqual(output.stat().st_size, 48, 'controlled stop must finish checkpoint DNA')
                        else:
                            with self.assertRaisesRegex(ValueError, 'topology'):
                                solver.hookSimulation(1, None)
                        self.assertEqual(process.returncode, 0, 'controlled stop must wait rather than terminate')
                        self.assertFalse(runtime.hasRunningDna(self.state))
                    finally:
                        runtime.cancelDna(self.state)

    def test_division_validates_coordinates_before_loading_the_new_chromosome(self):
        source = Path(__file__).resolve().parents[1] / 'SpatialDnaDynamics.py'
        names = {'updateChromosomeDivision', 'checkLastChromosome'}
        functions = [node for node in ast.parse(source.read_text()).body
                     if isinstance(node, ast.FunctionDef) and node.name in names]
        self.state.update(last_DNA_step=None, DNAcoords={}, genome={}, total_time=10,
                          timestep=0.5, divH=100)
        def write_truncated(*args):
            (self.root / 'DNA' / 'chromo_topo_2.dat').write_text('size=2\n')
            (self.root / 'DNA' / 'dna_monomers_2.bin').write_bytes(b'x' * 24)
        def disallow_placement(*args):
            self.fail('division consumed the aligned truncated chromosome')
        namespace = {'DnaRuntime': runtime, 'np': __import__('numpy'),
                     'writeDivisionChromosomeInputFile': lambda *args: None,
                     'runDivChromosome': write_truncated,
                     'placeNewChromosome': disallow_placement}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), namespace)
        with self.assertRaisesRegex(ValueError, 'topology'):
            namespace['updateChromosomeDivision'](1, None, self.state, {}, {})

    def test_partitioning_rejects_truncated_daughter_before_replacing_parent(self):
        import numpy as np
        source = Path(__file__).resolve().parents[1] / 'SpatialDnaDynamics.py'
        function = next(node for node in ast.parse(source.read_text()).body
                        if isinstance(node, ast.FunctionDef) and node.name == 'partitionChromosomes')
        self.state.update(last_DNA_step=250, divH_Prev=10, divR=20, lattice_spacing=10e-9)
        parent = self.root / 'DNA' / 'dna_monomers_250.bin'
        np.array([[0., 0., -100.], [0., 0., 100.]]).tofile(parent)
        original = parent.read_bytes()
        (self.root / 'DNA' / 'chromo_topo_250.dat').write_text('size=2\nm(1),1,1,1,1,1\nd(1),2,2,2,2,2\n')
        def write_truncated(side, *args):
            (self.root / 'DNA' / ('chromo_topo_' + side + '.dat')).write_text('size=2\n')
            (self.root / 'DNA' / ('dna_monomers_' + side + '.bin')).write_bytes(b'x' * 24)
        namespace = {'DnaRuntime': runtime, 'np': np, 'os': os,
                     'writePartitioningChromosomeInputFile': lambda state, side, *args: side,
                     'runPartitioningChromosome': write_truncated}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
        with self.assertRaisesRegex(ValueError, 'topology'):
            namespace['partitionChromosomes'](self.state)
        self.assertEqual(parent.read_bytes(), original)

    def test_serial_mode_finishes_before_return_and_routes_child_gpu(self):
        self.state.update(dna_mode='serial', dna_cuda_device=0)
        output = self.root / 'gpu.txt'
        runtime.startDna(self.state, [sys.executable, '-c',
            'import os,pathlib,sys; pathlib.Path(sys.argv[1]).write_text(os.environ["CUDA_VISIBLE_DEVICES"])', str(output)])
        self.assertTrue(output.exists())
        self.assertFalse(runtime.hasRunningDna(self.state))

    def test_resolves_fork_and_legacy_directory_names(self):
        fork = self.root / 'btree_chromo_gpu'
        exe = fork / 'build' / 'apps' / 'btree_chromo'
        exe.parent.mkdir(parents=True)
        exe.write_text('placeholder')
        self.assertEqual(runtime.btreeDirectory(self.state), fork)
        chain = self.root / 'sc_chain_generation' / 'src' / 'gen_sc_chain'
        chain.parent.mkdir(parents=True)
        chain.write_text('placeholder')
        self.assertEqual(runtime.chainExecutable(self.state), chain)
        fork.rename(self.root / 'btree_chromo')
        self.assertEqual(runtime.btreeDirectory(self.state), self.root / 'btree_chromo')

    def test_fork_binary_takes_precedence_over_a_legacy_checkout(self):
        for name in ('btree_chromo', 'btree_chromo_gpu'):
            executable = self.root / name / 'build' / 'apps' / 'btree_chromo'
            executable.parent.mkdir(parents=True)
            executable.write_text('placeholder')
        self.assertEqual(runtime.btreeDirectory(self.state), self.root / 'btree_chromo_gpu')


if __name__ == '__main__':
    unittest.main()

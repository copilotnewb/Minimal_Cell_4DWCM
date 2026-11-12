"""Detect native LM failures that return normally but leave incomplete HDF5."""
import ast
import importlib
from pathlib import Path
import tempfile
import types
import unittest

import numpy as np
try:
    import h5py
except ImportError:
    h5py = None

import DnaRuntime


@unittest.skipIf(h5py is None, 'h5py is required for LM result files')
class RdmeResultsTests(unittest.TestCase):
    def make_file(self, directory, duration=2., times=(0., 1., 2.)):
        path = Path(directory) / 'MinCell_restart_100.lm'
        with h5py.File(path, 'w') as handle:
            params = handle.create_group('Parameters')
            for key, value in {'maxTime': duration, 'timestep': 50e-6,
                               'writeInterval': 20000, 'latticeWriteInterval': 20000}.items():
                params.attrs[key] = np.bytes_(str(value))
            replica = handle.create_group('Simulations/0000001')
            replica.create_dataset('SpeciesCountTimes', data=times)
            replica.create_dataset('SpeciesCounts', data=np.ones((len(times), 2), dtype=np.int32))
            replica.create_dataset('LatticeTimes', data=times)
            lattices = replica.create_group('Lattice')
            for index in range(len(times)):
                lattices.create_dataset('{:010d}'.format(index), data=np.zeros((2, 3, 4, 16), dtype=np.uint32))
        return path

    def run_entrypoint_tail(self, filename, output, state):
        # Execute the real run/finally/completion portion without loading LM.
        source = Path(__file__).resolve().parents[1] / filename
        body = ast.parse(source.read_text()).body
        start = next(i for i, node in enumerate(body) if isinstance(node, ast.Try)
                     and any(isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
                             and child.func.attr == 'run' for child in ast.walk(node)))
        try:
            results = importlib.import_module('RdmeResults')
        except ModuleNotFoundError as error:
            if error.name != 'RdmeResults':
                raise
            results = None
        sim = types.SimpleNamespace(filename=output, simulationTime=2., run=lambda **kwargs: None)
        namespace = {'sim': sim, 'solver': object(), 'sim_properties': state,
                     'args': types.SimpleNamespace(cudaDevices=0),
                     'DnaRuntime': DnaRuntime, 'RdmeResults': results}
        exec(compile(ast.Module(body=body[start:], type_ignores=[]), str(source), 'exec'), namespace)

    def test_both_entrypoints_reject_silent_native_early_return(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.make_file(directory, times=(0., 1.))
            state = {'working_directory': directory, 'last_DNA_step': None, 'time': 101.}
            for filename in ('Whole_Cell_Minimal_Cell.py', 'Restart_Whole_Cell_Minimal_Cell.py'):
                with self.subTest(entrypoint=filename), self.assertRaisesRegex(ValueError, 'incomplete'):
                    self.run_entrypoint_tail(filename, path, state)

    def test_restart_completion_uses_local_hdf_time(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.make_file(directory)
            self.run_entrypoint_tail('Restart_Whole_Cell_Minimal_Cell.py', path,
                                     {'working_directory': directory, 'last_DNA_step': None, 'time': 102.})

    def test_fractional_final_interval_uses_last_scheduled_frame(self):
        module = importlib.import_module('RdmeResults')
        with tempfile.TemporaryDirectory() as directory:
            for duration, times in ((1.25, (0., 1.)), (.5, (0.,)), (.99999, (0., 1.))):
                with self.subTest(duration=duration):
                    module.validateCompletion(self.make_file(directory, duration, times))

    def test_missing_final_lattice_frame_is_rejected(self):
        module = importlib.import_module('RdmeResults')
        with tempfile.TemporaryDirectory() as directory:
            path = self.make_file(directory)
            with h5py.File(path, 'a') as handle:
                del handle['Simulations/0000001/Lattice/0000000002']
            with self.assertRaisesRegex(ValueError, 'lattice frame'):
                module.validateCompletion(path)

    def test_count_time_shape_mismatch_is_rejected(self):
        module = importlib.import_module('RdmeResults')
        with tempfile.TemporaryDirectory() as directory:
            path = self.make_file(directory)
            with h5py.File(path, 'a') as handle:
                del handle['Simulations/0000001/SpeciesCounts']
                handle['Simulations/0000001'].create_dataset('SpeciesCounts', data=[[1, 2], [3, 4]])
            with self.assertRaisesRegex(ValueError, 'dimensions'):
                module.validateCompletion(path)

    def test_invalid_final_time_is_rejected(self):
        module = importlib.import_module('RdmeResults')
        with tempfile.TemporaryDirectory() as directory:
            for end_time in (float('nan'), 2.5):
                path = self.make_file(directory)
                with h5py.File(path, 'a') as handle:
                    handle['Simulations/0000001/SpeciesCountTimes'][-1] = end_time
                with self.subTest(end_time=end_time), self.assertRaises(ValueError):
                    module.validateCompletion(path)


if __name__ == '__main__':
    unittest.main()

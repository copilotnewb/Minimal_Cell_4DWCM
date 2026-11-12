"""Read real HDF5 fixtures with both LM species-name encodings."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
try:
    import h5py
except ImportError:
    h5py = None

import CmeResults


@unittest.skipIf(h5py is None, 'h5py is required for LM result files')
class CmeResultsTests(unittest.TestCase):
    def make_file(self, directory, new_names=True, end_time=1.0):
        path = Path(directory) / 'cme.lm'
        with h5py.File(path, 'w') as handle:
            params = handle.create_group('Parameters')
            if new_names:
                params.create_dataset('SpeciesNames', data=np.array([[b'RNA'], [b'TRNA']]))
            else:
                params.attrs['speciesNames'] = np.bytes_('RNA,TRNA')
            replicate = handle.create_group('Simulations/0000001')
            replicate.create_dataset('SpeciesCountTimes', data=[0., .5, end_time])
            replicate.create_dataset('SpeciesCounts', data=[[9, 12], [8, 11], [5, 14]], dtype=np.int32)
        return path

    def test_final_row_and_decoded_species_for_both_formats(self):
        with tempfile.TemporaryDirectory() as directory:
            for new_names in (True, False):
                path = self.make_file(directory, new_names)
                names, counts = CmeResults.readFinalCounts(path)
                self.assertEqual(names, ['RNA', 'TRNA'])
                np.testing.assert_array_equal(counts, [5, 14])
                # The file is closed on return, so another writer can open it.
                with h5py.File(path, 'a') as handle:
                    handle.attrs['checked'] = True

    def test_incomplete_native_solver_result_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.make_file(directory, end_time=.9)
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                CmeResults.readFinalCounts(path)

    def test_shape_mismatch_is_rejected_without_partial_update(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.make_file(directory)
            with h5py.File(path, 'a') as handle:
                del handle['Simulations/0000001/SpeciesCounts']
                handle['Simulations/0000001'].create_dataset('SpeciesCounts', data=[[1], [2], [3]])
            with self.assertRaises(ValueError):
                CmeResults.readFinalCounts(path)


if __name__ == '__main__':
    unittest.main()

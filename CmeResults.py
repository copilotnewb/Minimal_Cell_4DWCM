"""Read the final global-CME state in one HDF5 hyperslab."""
import numpy as np


def readFinalCounts(filename, expected_time=1.0):
    # h5py is already a dependency of Lattice Microbes; keep other CPU helpers
    # importable without loading the simulator's optional dependencies.
    import h5py

    with h5py.File(filename, 'r') as handle:
        params = handle['Parameters']
        if 'SpeciesNames' in params:
            raw_names = np.asarray(params['SpeciesNames']).reshape(-1)
            names = [name.decode('utf-8') if isinstance(name, bytes) else str(name)
                     for name in raw_names]
        else:
            raw_names = params.attrs['speciesNames']
            if isinstance(raw_names, bytes):
                raw_names = raw_names.decode('utf-8')
            names = str(raw_names).split(',')
        replicate = handle['Simulations']['0000001']
        counts = replicate['SpeciesCounts']
        times = replicate['SpeciesCountTimes']
        if (counts.ndim != 2 or counts.shape[1] != len(names)
                or counts.shape[0] == 0 or times.shape != (counts.shape[0],)):
            raise ValueError('Global CME output has inconsistent species/count/time dimensions: {}.'.format(filename))
        if not np.isfinite(times[-1]) or times[-1] < expected_time - 1e-9:
            raise ValueError('Global CME output is incomplete: {} did not reach {} seconds.'.format(filename, expected_time))
        return names, counts[-1, :]

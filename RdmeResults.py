"""Check persisted RDME output after LM's native solver returns."""
import math


def validateCompletion(filename):
    """Require the final scheduled counts and lattice for this local run.

    LM catches native solver exceptions without propagating them to Python.
    Its IntMpdRdmeSolver records times in local elapsed seconds, including on a
    restart, and does not append an extra frame for a shortened final interval.
    Check the schedule stored in this file instead of the cumulative cell time.
    A failure after the last scheduled output in a fractional remainder cannot
    be detected from HDF5 alone; current entrypoints request integer seconds.
    """
    import h5py

    with h5py.File(filename, 'r') as handle:
        parameters = handle['Parameters'].attrs
        timestep = float(parameters['timestep'])
        duration = float(parameters['maxTime'])
        if (not math.isfinite(timestep) or timestep <= 0
                or not math.isfinite(duration) or duration < 0):
            raise ValueError('Invalid RDME time parameters in {}.'.format(filename))

        # Match the native loop: run integer steps until step*tau >= maxTime.
        final_step = math.ceil(duration / timestep)
        if final_step and (final_step - 1) * timestep >= duration:
            final_step -= 1
        if final_step * timestep < duration:
            final_step += 1
        replica = handle['Simulations']['0000001']

        for dataset, parameter in (('SpeciesCountTimes', 'writeInterval'),
                                   ('LatticeTimes', 'latticeWriteInterval')):
            interval = int(parameters[parameter])
            if interval <= 0:
                raise ValueError('Invalid RDME {} in {}.'.format(parameter, filename))
            times = replica[dataset]
            scheduled_step = (final_step // interval) * interval
            expected = scheduled_step * timestep
            if times.ndim != 1 or times.shape[0] == 0:
                raise ValueError('RDME output is incomplete: {} has no {}.'.format(filename, dataset))
            actual = float(times[-1])
            if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=0., abs_tol=1e-9):
                raise ValueError('RDME output is incomplete or inconsistent: {} ends {} at {}, expected {} seconds. Check the native LM log for a solver failure.'.format(
                    filename, dataset, actual, expected))

        counts = replica['SpeciesCounts']
        if (counts.ndim != 2 or counts.shape[0] != replica['SpeciesCountTimes'].shape[0]
                or counts.shape[1] == 0):
            raise ValueError('RDME output has inconsistent count/time dimensions: {}.'.format(filename))
        # Read the small final row to surface any HDF5 data error before success.
        counts[-1, :]
        final_frame = '{:010d}'.format(replica['LatticeTimes'].shape[0] - 1)
        if final_frame not in replica['Lattice'] or replica['Lattice'][final_frame].size == 0:
            raise ValueError('RDME output is missing its final lattice frame: {}.'.format(filename))

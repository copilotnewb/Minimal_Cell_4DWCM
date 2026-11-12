#!/usr/bin/env python3
"""CPU microbenchmarks at the default lattice dimensions; no simulator needed."""
import argparse
import copy
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from LatticeFunctions import particleLocations
import RibosomesRDME


class ArrayLattice:
    def __init__(self, sites):
        self.sites = sites

    def getSiteLatticeView(self):
        return self.sites

    def setSiteType(self, x, y, z, value):
        self.sites[z, y, x] = value


def measure(function, repeats):
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        value = function()
        samples.append(time.perf_counter() - start)
    return value, statistics.median(samples), samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--species', type=int, default=200)
    parser.add_argument('--baseline-ref', default='08f5f2080dc5d766448b02e20c74a6d04c737f94')
    args = parser.parse_args()
    if args.repeats < 1 or not 1 <= args.species <= 500:
        parser.error('Use positive repeats and 1..500 species.')
    rng = np.random.default_rng(13)
    shape = (64, 64, 128)
    particles = np.zeros((16,) + shape + (1,), dtype=np.uint32)
    flat = particles.reshape(-1)
    flat[rng.choice(flat.size, 100000, replace=False)] = rng.integers(1, 501, size=100000)
    ids = list(range(1, args.species + 1))
    reference, old_seconds, old_samples = measure(lambda: {pid: np.argwhere(particles == pid) for pid in ids}, args.repeats)
    actual, new_seconds, new_samples = measure(lambda: particleLocations(particles, ids), args.repeats)
    for pid in ids:
        np.testing.assert_array_equal(actual[pid], reference[pid])
    result = {'scope': 'synthetic CPU routines, not a whole-cell or GPU benchmark',
              'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform(),
              'lattice_shape': particles.shape, 'occupied_slots': 100000, 'selected_species': len(ids),
              'particle_locations': {'baseline_median_seconds': old_seconds, 'optimized_median_seconds': new_seconds,
                                     'ratio': old_seconds / new_seconds, 'exact_coordinates': True,
                                     'baseline_samples': old_samples, 'optimized_samples': new_samples}}

    source = subprocess.check_output(['git', 'show', args.baseline_ref + ':RibosomesRDME.py'], cwd=ROOT, text=True)
    legacy = types.ModuleType('baseline_ribosomes')
    exec(compile(source, 'baseline_RibosomesRDME.py', 'exec'), legacy.__dict__)
    names = ('cytoplasm', 'outer_cytoplasm', 'DNA', 'extracellular', 'membrane', 'center', 'cross')
    regions = {name: {'shape': np.zeros(shape, dtype=bool), 'index': idx + 1} for idx, name in enumerate(names)}
    regions['cytoplasm']['shape'][:] = True
    for name, count in [('DNA', 40000), ('center', 3000), ('cross', 18000)]:
        regions[name]['shape'].reshape(-1)[rng.choice(np.prod(shape), count, replace=False)] = True
    centers = np.zeros(shape, dtype=bool)
    crosses = np.zeros(shape, dtype=bool)
    centers.reshape(-1)[rng.choice(np.prod(shape), 3000, replace=False)] = True
    crosses.reshape(-1)[rng.choice(np.prod(shape), 18000, replace=False)] = True
    ribosomes = {'ribos': {'center_idx': 'center', 'cross_idx': 'cross', 'centers': centers, 'crosses': crosses}}
    def site_run(function):
        lattice = ArrayLattice(np.zeros(shape, dtype=np.uint8))
        function(lattice, ribosomes, copy.deepcopy(regions))
        return lattice.sites
    before, old_seconds, old_samples = measure(lambda: site_run(legacy.updateRiboSites), args.repeats)
    after, new_seconds, new_samples = measure(lambda: site_run(RibosomesRDME.updateRiboSites), args.repeats)
    np.testing.assert_array_equal(after, before)
    result['ribosome_site_updates'] = {'baseline_median_seconds': old_seconds, 'optimized_median_seconds': new_seconds,
                                     'ratio': old_seconds / new_seconds, 'exact_sites': True,
                                     'baseline_samples': old_samples, 'optimized_samples': new_samples,
                                     'note': 'Includes equal array-copy setup; baseline setters are Python stand-ins for SWIG.'}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

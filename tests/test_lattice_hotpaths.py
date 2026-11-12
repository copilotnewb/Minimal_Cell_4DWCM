"""CPU equivalence checks for the array operations used at every hook."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

import LatticeFunctions as lattice_functions
import RibosomesRDME


class SiteLattice:
    def __init__(self, sites):
        self.sites = sites

    def getSiteLatticeView(self):
        return self.sites

    def setSiteType(self, x, y, z, site_type):
        self.sites[z, y, x] = site_type


def reference_site_update(lattice, ribosomes, regions):
    """Original ordered setter algorithm, independent of the vectorized path."""
    for details in ribosomes.values():
        for part, bases in (
            ('center_idx', ('cytoplasm', 'outer_cytoplasm', 'DNA')),
            ('cross_idx', ('cytoplasm', 'outer_cytoplasm', 'DNA', 'extracellular')),
        ):
            for point in np.argwhere(regions[details[part]]['shape']):
                point = tuple(point)
                for base in bases:
                    if regions[base]['shape'][point]:
                        lattice.setSiteType(*point[::-1], regions[base]['index'])
                        break
        regions[details['center_idx']]['shape'] = details['centers']
        regions[details['cross_idx']]['shape'] = details['crosses']
    for details in ribosomes.values():
        for point in np.argwhere(details['crosses']):
            point = tuple(point)
            if not (regions['membrane']['shape'][point] or regions['DNA']['shape'][point]):
                lattice.setSiteType(*point[::-1], regions[details['cross_idx']]['index'])
    for details in ribosomes.values():
        for point in np.argwhere(details['centers']):
            lattice.setSiteType(*point[::-1], regions[details['center_idx']]['index'])
    return regions


class ParticleLocationTests(unittest.TestCase):
    def test_matches_argwhere_order_with_duplicates_missing_and_zero(self):
        # A transposed, non-cubic lattice also catches accidentally flattened axes.
        rng = np.random.default_rng(46188)
        particles = rng.integers(0, 21, size=(4, 7, 3, 9, 4), dtype=np.uint32)
        particles = particles.transpose(0, 3, 2, 1, 4)
        actual = lattice_functions.particleLocations(particles, [17, 2, 17, 99, 0])
        self.assertEqual(set(actual), {17, 2, 99, 0})
        for pid in actual:
            np.testing.assert_array_equal(actual[pid], np.argwhere(particles == pid))

    def test_empty_input_and_no_matches(self):
        particles = np.zeros((1, 2, 3, 4, 1), dtype=np.uint32)
        self.assertEqual(lattice_functions.particleLocations(particles, []), {})
        actual = lattice_functions.particleLocations(particles, [8])
        self.assertEqual(actual[8].shape, (0, 5))


class RibosomeSiteTests(unittest.TestCase):
    def test_interior_ribosomes_do_not_build_an_unused_shell_tree(self):
        shape = (7, 5, 11)
        particles = np.zeros((16,) + shape + (1,), dtype=np.uint32)
        particles[0, 2, 2, 3, 0] = 3
        lattice = SiteLattice(np.zeros(shape, dtype=np.uint8))
        lattice.getParticleLatticeView = lambda: particles
        lattice.getOccupancy = lambda x, y, z: np.count_nonzero(particles[:, z, y, x, :])
        regions = {name: {'shape': np.zeros(shape, dtype=bool)}
                   for name in ('membrane', 'extracellular', 'outer_cytoplasm')}
        regions['outer_cytoplasm']['shape'][1, :, :] = True
        ribosomes = {'ribos': {'center_idx': 'center', 'cross_idx': 'cross',
                              'centers': np.zeros(shape, dtype=bool), 'crosses': np.zeros(shape, dtype=bool)}}
        state = {'lattice_edges': shape, 'riboIDs': [False, False, True],
                 'name_to_index': {'ribosomeP': 3}}
        with patch.object(RibosomesRDME.spatial, 'KDTree', wraps=RibosomesRDME.spatial.KDTree) as trees:
            result = RibosomesRDME.placeRibosomes(lattice, state, regions, ribosomes)
        self.assertEqual(trees.call_count, 0)
        np.testing.assert_array_equal(np.argwhere(result['ribos']['centers']), [[2, 2, 3]])
        np.testing.assert_array_equal(np.argwhere(result['ribos']['crosses']),
                                      [[1, 2, 3], [2, 1, 3], [2, 2, 2], [2, 2, 4], [2, 3, 3], [3, 2, 3]])

    def test_matches_original_priority_for_overlaps_and_multiple_types(self):
        rng = np.random.default_rng(13)
        shape = (7, 5, 11)
        for _ in range(20):
            regions = {}
            names = ('cytoplasm', 'outer_cytoplasm', 'DNA', 'extracellular', 'membrane', 'c1', 'x1', 'c2', 'x2')
            for idx, name in enumerate(names):
                regions[name] = {'shape': rng.random(shape) < 0.25, 'index': idx + 1}
            ribosomes = {
                'first': {'center_idx': 'c1', 'cross_idx': 'x1', 'centers': rng.random(shape) < .1, 'crosses': rng.random(shape) < .25},
                'second': {'center_idx': 'c2', 'cross_idx': 'x2', 'centers': rng.random(shape) < .1, 'crosses': rng.random(shape) < .25},
            }
            initial = rng.integers(0, 10, size=shape, dtype=np.uint8)
            expected_lattice = SiteLattice(initial.copy())
            actual_lattice = SiteLattice(initial.copy())
            expected = reference_site_update(expected_lattice, copy.deepcopy(ribosomes), copy.deepcopy(regions))
            actual = RibosomesRDME.updateRiboSites(actual_lattice, ribosomes, regions)
            np.testing.assert_array_equal(actual_lattice.sites, expected_lattice.sites)
            for name in names:
                np.testing.assert_array_equal(actual[name]['shape'], expected[name]['shape'])
            self.assertIs(actual['c1']['shape'], ribosomes['first']['centers'])

    def test_bulk_view_update_marks_cuda_lattice_dirty_once(self):
        # CudaIntLattice.copyFromGPU marks host/device synchronized. Merely
        # changing the inherited IntLattice view does not clear that flag.
        class CudaSiteLattice(SiteLattice):
            def __init__(self, sites):
                super().__init__(sites)
                self.device = sites.copy()
                self.synchronized = True
                self.setter_calls = 0

            def setSiteType(self, x, y, z, value):
                super().setSiteType(x, y, z, value)
                self.synchronized = False
                self.setter_calls += 1

            def copyToGPU(self):
                if not self.synchronized:
                    self.device[:] = self.sites
                    self.synchronized = True

        shape = (3, 4, 7)
        regions = {name: {'shape': np.zeros(shape, bool), 'index': i + 1}
                   for i, name in enumerate(('cytoplasm', 'outer_cytoplasm', 'DNA', 'extracellular', 'membrane', 'center', 'cross'))}
        regions['cytoplasm']['shape'][:] = True
        regions['center']['shape'][1, 2, 5] = True
        centers = np.zeros(shape, bool)
        centers[2, 1, 6] = True
        ribosomes = {'ribos': {'center_idx': 'center', 'cross_idx': 'cross', 'centers': centers, 'crosses': np.zeros(shape, bool)}}
        lattice = CudaSiteLattice(np.zeros(shape, np.uint8))
        RibosomesRDME.updateRiboSites(lattice, ribosomes, regions)
        lattice.copyToGPU()
        self.assertEqual(lattice.device[1, 2, 5], 1)
        self.assertEqual(lattice.device[2, 1, 6], 6)
        self.assertEqual(lattice.setter_calls, 1)


if __name__ == '__main__':
    unittest.main()

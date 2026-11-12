"""Exercise real cost accounting using the LM particle array layout."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np


# These two external packages are not needed by the cost functions themselves.
_spec = importlib.util.spec_from_file_location('cost_communication', Path(__file__).resolve().parents[1] / 'Communicate.py')
communicate = importlib.util.module_from_spec(_spec)
with patch.dict(sys.modules, {'pySTDLM': types.ModuleType('pySTDLM'),
                             'pySTDLM.PostProcessing': types.ModuleType('pySTDLM.PostProcessing'),
                             'MC_CME': types.ModuleType('MC_CME')}):
    _spec.loader.exec_module(communicate)


class ParticleLattice:
    def __init__(self):
        self.particles = np.zeros((16, 3, 4, 7, 1), dtype=np.uint32)

    def getParticleLatticeView(self):
        return self.particles

    def addParticle(self, x, y, z, pid):
        site = self.particles[:, z, y, x, 0]
        site[np.flatnonzero(site == 0)[0]] = pid


def properties():
    names = {'P_0001_TC': 11, 'P_0002_TC': 12, 'P_0003_TC': 13,
             'P_0001': 21, 'P_0002': 22, 'C_P_0002': 23,
             'DT_0001': 31, 'DT_0002': 32, 'DT_0003': 33, 'decay': 40}
    counts = {name: 0 for name in names}
    for amino_acid in ('ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY', 'HIS', 'ILE',
                       'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'):
        counts[amino_acid + '_cost'] = 0
        counts[amino_acid + '_cost_second'] = 0
    for key in ('GTP_translat_cost', 'FMET_cost', 'ATP_mRNAdeg_cost', 'AMP_mRNAdeg_cost',
                'CMP_mRNAdeg_cost', 'GMP_mRNAdeg_cost', 'UMP_mRNAdeg_cost'):
        counts[key] = counts[key + '_second'] = 0
    for key in ('PM_0001', 'PM_0002', 'PM_0003', 'DM_0001', 'DM_0002', 'DM_0003', 'total_proteins_made'):
        counts[key] = 0
    return {'name_to_index': names, 'counts': counts,
            'genome': {'JCVISYN3A_0001': {'Type': 'protein', 'AAsequence': 'AAG', 'RNAsequence': 'AACGU'},
                       'JCVISYN3A_0002': {'Type': 'protein', 'AAsequence': 'MF', 'RNAsequence': 'CCU'},
                       'JCVISYN3A_0003': {'Type': 'protein', 'AAsequence': 'W', 'RNAsequence': 'G'},
                       'JCVISYN3A_0004': {'Type': 'tRNA', 'RNAsequence': 'ACGU'}}}


class CommunicationCostTests(unittest.TestCase):
    def test_translation_multiple_species_one_scan_preserves_compaction_and_costs(self):
        state = properties()
        lattice = ParticleLattice()
        lattice.particles[:4, 1, 2, 4, 0] = [11, 12, 11, 90]
        lattice.particles[0, 2, 0, 5, 0] = 12
        state['counts'].update(P_0001_TC=2, P_0002_TC=2, P_0003_TC=1)
        with patch.object(communicate.np, 'argwhere', wraps=np.argwhere) as scans:
            communicate.calculateTranslationCosts(state, lattice)
        self.assertEqual(scans.call_count, 1, 'selected species must share one full-lattice scan')
        np.testing.assert_array_equal(lattice.particles[:4, 1, 2, 4, 0], [90, 21, 21, 23])
        self.assertEqual(lattice.particles[0, 2, 0, 5, 0], 23)
        self.assertEqual(np.count_nonzero(lattice.particles), 5)
        for key, expected in {'GTP_translat_cost': 20, 'ALA_cost': 4, 'GLY_cost': 2,
                              'MET_cost': 2, 'PHE_cost': 2, 'FMET_cost': 4}.items():
            self.assertEqual(state['counts'][key], expected)
            self.assertEqual(state['counts'][key + '_second'], expected)
        self.assertEqual(state['counts']['PM_0001'], 2)
        self.assertEqual(state['counts']['PM_0002'], 2)
        self.assertEqual(state['counts']['PM_0003'], 0)
        self.assertEqual(state['counts']['total_proteins_made'], 4)

    def test_degradation_multiple_species_one_scan_preserves_stoichiometry(self):
        state = properties()
        lattice = ParticleLattice()
        lattice.particles[:4, 1, 2, 4, 0] = [31, 32, 31, 90]
        state['counts'].update(DT_0001=2, DT_0002=1)
        with patch.object(communicate.np, 'argwhere', wraps=np.argwhere) as scans:
            communicate.calculateDegradationCosts(state, lattice)
        self.assertEqual(scans.call_count, 1)
        np.testing.assert_array_equal(lattice.particles[:4, 1, 2, 4, 0], [90, 40, 40, 40])
        for key, expected in {'ATP_mRNAdeg_cost': 13, 'AMP_mRNAdeg_cost': 4,
                              'CMP_mRNAdeg_cost': 4, 'GMP_mRNAdeg_cost': 2, 'UMP_mRNAdeg_cost': 3}.items():
            self.assertEqual(state['counts'][key], expected)
            self.assertEqual(state['counts'][key + '_second'], expected)
        self.assertEqual(state['counts']['DM_0001'], 2)
        self.assertEqual(state['counts']['DM_0002'], 1)


if __name__ == '__main__':
    unittest.main()

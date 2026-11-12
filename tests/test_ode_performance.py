"""CPU-only regressions for ODE sampling and static metabolic input reuse."""

import importlib.util
import copy
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import pandas as pd
from scipy import integrate as scipy_integrate


ROOT = Path(__file__).resolve().parents[1]


def load_module(name):
    # These optional packages are not used by the sampling and spreadsheet
    # paths under test. Real NumPy, Pandas, Excel parsing and SciPy are used.
    cvodes = types.ModuleType("pycvodes")
    cvodes.integrate_predefined = None
    cvodes.integrate_adaptive = None
    imports = {
        "pycvodes": cvodes,
        "odecell": types.ModuleType("odecell"),
        "libsbml": types.ModuleType("libsbml"),
        "pySTDLM": types.ModuleType("pySTDLM"),
        "pySTDLM.PostProcessing": types.ModuleType("pySTDLM.PostProcessing"),
        "LatticeFunctions": types.ModuleType("LatticeFunctions"),
    }
    with mock.patch.dict(sys.modules, imports):
        spec = importlib.util.spec_from_file_location(name + "_under_test", ROOT / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class InitialModel:
    def getInitVals(self):
        return [2.0, 1.0]


class RecordingModel:
    def __init__(self):
        self.metabolites = {}

    def addMetabolite(self, identifier, name, concentration):
        self.metabolites[identifier] = concentration

    def getRxnList(self):
        return []


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module("Integrate")

    @staticmethod
    def rhs(t, y):
        rate = 0.7 * y[0] + 0.02 * np.sin(t)
        return [-rate, rate]

    def test_lsoda_samples_match_the_existing_requests_and_defaults(self):
        # Reference the original repeated-target protocol: replacing it with
        # linspace or changing tolerances changes this numerical trajectory.
        reference = scipy_integrate.ode(self.rhs).set_integrator("lsoda")
        reference.set_initial_value(InitialModel().getInitVals())
        expected = []
        while reference.successful() and reference.t < 1.0:
            expected.append(reference.integrate(reference.t + 0.01).copy())
        actual = self.module.runODE([99.0, 99.0], self.rhs, InitialModel())
        self.assertEqual(actual.shape, (100, 2))
        np.testing.assert_array_equal(actual, np.asarray(expected))

    def test_result_collection_does_not_repeatedly_copy_prior_samples(self):
        with mock.patch.object(self.module.np, "append", wraps=np.append) as append:
            result = self.module.runODE(None, self.rhs, InitialModel())
        self.assertEqual(append.call_count, 0)
        self.assertEqual(result.shape, (100, 2))
        np.testing.assert_allclose(result.sum(axis=1), 3.0, rtol=0, atol=3e-15)

    def test_early_failure_returns_only_written_samples(self):
        class FailingIntegrator:
            def __init__(self, callback):
                self.calls = 0
                self.t = 0.0
                self.buffer = np.zeros(2)

            def set_integrator(self, name):
                self.name = name

            def set_initial_value(self, initial):
                self.initial = initial

            def successful(self):
                return self.calls < 3

            def integrate(self, target):
                self.calls += 1
                self.t = target
                self.buffer[:] = [self.calls, self.calls + 10]
                return self.buffer

        with mock.patch.object(self.module.integrate, "ode", FailingIntegrator):
            result = self.module.runODE(None, self.rhs, InitialModel())
        np.testing.assert_array_equal(result, [[1.0, 11.0], [2.0, 12.0], [3.0, 13.0]])


class MetabolicInputTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module("Rxns_ODE")
        self.directory = tempfile.TemporaryDirectory(dir=ROOT / "tests")
        self.addCleanup(self.directory.cleanup)
        self.head = Path(self.directory.name)
        (self.head / "input_data").mkdir()
        self.workbook = self.head / "input_data" / "protein_metabolites.xlsx"
        self.write_workbook("M_base,M_modified")

    def write_workbook(self, metabolites):
        pd.DataFrame({"Protein": ["P1"], "Metabolite IDs": [metabolites]}).to_excel(
            self.workbook, sheet_name="protein metabolites", index=False
        )

    def properties(self, protein_count=100, modified_count=20, volume=1e-15):
        return {
            "head_directory": str(self.head) + "/",
            "counts": {"P1": protein_count, "M_modified": modified_count, "M_other": 7},
            "volume_L": volume,
        }

    def test_cached_input_does_not_freeze_counts_or_volume(self):
        first = RecordingModel()
        second = RecordingModel()
        with mock.patch.object(self.module.pd, "read_excel", wraps=pd.read_excel) as reads:
            self.module.addProteinMetabolites(first, self.properties())
            self.module.addProteinMetabolites(second, self.properties(150, 30, 2e-15))
        self.assertEqual(reads.call_count, 1)
        factor = 1000.0 / (6.022e23 * 1e-15)
        self.assertEqual(first.metabolites, {"M_base": 80 * factor, "M_modified": 20 * factor})
        self.assertAlmostEqual(second.metabolites["M_base"], 60 * factor)
        self.assertAlmostEqual(second.metabolites["M_modified"], 15 * factor)

    def test_modified_workbook_is_read_again(self):
        first = RecordingModel()
        second = RecordingModel()
        self.module.addProteinMetabolites(first, self.properties())
        self.write_workbook("M_base,M_other")
        self.module.addProteinMetabolites(second, self.properties())
        self.assertIn("M_modified", first.metabolites)
        self.assertNotIn("M_modified", second.metabolites)
        self.assertIn("M_other", second.metabolites)

    def test_returned_sheet_cannot_mutate_later_models(self):
        first = self.module._read_model_sheet(self.workbook, "protein metabolites")
        first.loc[0, "Protein"] = "changed"
        second = self.module._read_model_sheet(self.workbook, "protein metabolites")
        self.assertEqual(second.loc[0, "Protein"], "P1")

    def test_random_binding_model_reuses_sbml_parse(self):
        # libSBML is an optional compiled dependency on CPU-only test machines.
        # The stand-in exposes only the document/model APIs used by an empty
        # reaction table, so parsing reuse is exercised through the real caller.
        class Document:
            def clone(self):
                return copy.deepcopy(self)

            def getModel(self):
                return self

            def getListOfSpecies(self):
                return []

            def getListOfReactions(self):
                return []

        kinetic = self.head / "input_data" / "kinetic_params.xlsx"
        with pd.ExcelWriter(kinetic) as writer:
            for sheet in ("Central", "Nucleotide", "Lipid", "Cofactor", "Transport"):
                pd.DataFrame({"Reaction Name": []}).to_excel(writer, sheet_name=sheet, index=False)
        (self.head / "input_data" / "Syn3A_updated.xml").write_text("test document")
        reader = mock.Mock(return_value=Document())
        self.module.libsbml.readSBMLFromFile = reader
        self.module.defineRandomBindingRxns(RecordingModel(), self.properties())
        self.module.defineRandomBindingRxns(RecordingModel(), self.properties())
        self.assertEqual(reader.call_count, 1)

    def test_sbml_documents_are_isolated_and_reloaded_after_changes(self):
        class Document:
            def __init__(self, value):
                self.value = value

            def clone(self):
                return copy.deepcopy(self)

        path = self.head / "input_data" / "Syn3A_updated.xml"
        path.write_text("first input")
        self.module.libsbml.readSBMLFromFile = lambda filename: Document(Path(filename).read_text())
        first = self.module._read_sbml_file(path)
        first.value = "caller mutation"
        second = self.module._read_sbml_file(path)
        self.assertEqual(second.value, "first input")
        path.write_text("changed input with a different size")
        third = self.module._read_sbml_file(path)
        self.assertEqual(third.value, "changed input with a different size")


class FluxSavingTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module("FileSaving")
        self.directory = tempfile.TemporaryDirectory(dir=ROOT / "tests")
        self.addCleanup(self.directory.cleanup)
        self.folder = Path(self.directory.name)
        self.properties = {
            "working_directory": str(self.folder) + "/",
            "counts": {"M_A": 40, "M_B": 50},
        }
        self.model = types.SimpleNamespace(getRxnList=lambda: [
            types.SimpleNamespace(getID=lambda: "R1"),
            types.SimpleNamespace(getID=lambda: "R2"),
            types.SimpleNamespace(getID=lambda: "R3"),
        ])
        self.solver = types.SimpleNamespace(calcFlux=lambda t, y: [2.5 * y[0], -0.75 * y[1], y[0] + y[1]])
        self.results = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        self.module.saveCountsAndFluxes(0, self.properties, self.results, self.model, self.solver)
        self.properties["counts"] = {"M_A": 45, "M_B": 55}

    def test_flux_collection_does_not_repeatedly_copy_prior_samples(self):
        # Keep Pandas' internal NumPy calls outside this assertion: this checks
        # only the output-buffer behavior in FileSaving itself.
        numpy = types.SimpleNamespace(**vars(np))
        numpy.append = mock.Mock(wraps=np.append)
        with mock.patch.object(self.module, "np", numpy):
            self.module.saveCountsAndFluxes(1, self.properties, self.results, self.model, self.solver)
        self.assertEqual(numpy.append.call_count, 0)

    def test_saved_fluxes_and_csv_format_are_unchanged(self):
        self.module.saveCountsAndFluxes(1, self.properties, self.results, self.model, self.solver)
        expected = (
            "Time,0.0,1.0\n"
            "M_A,40.0,45.0\n"
            "M_B,50.0,55.0\n"
            "F_R1,2.5,7.5\n"
            "F_R2,-1.5,-3.0\n"
            "F_R3,3.0,7.0\n"
        )
        self.assertEqual((self.folder / "counts_and_fluxes.csv").read_text(), expected)


if __name__ == "__main__":
    unittest.main()

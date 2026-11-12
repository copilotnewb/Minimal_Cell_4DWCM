"""Compare the original and optimized metabolic path with the shipped inputs.

Example (run from the Minimal_Cell_4DWCM repository):
    python benchmarks/benchmark_ode.py --baseline-main-ref 08f5f20 --baseline-odecell-ref b19413f

Requires NumPy, SciPy, Pandas, openpyxl and python-libsbml. The sibling odecell
checkout supplies its real modelbuilder/solver; optional FBA/optimizer imports
are isolated by its CPU test loader. This is a deterministic metabolic fixture,
not a full spatial simulation or a measurement of GPU runtime/VRAM.
"""

import argparse
import copy
import json
from pathlib import Path
import platform
from statistics import median
import subprocess
import sys
from time import perf_counter
import types
from unittest import mock

import libsbml
import numpy as np
import pandas as pd
import scipy


ROOT = Path(__file__).resolve().parents[1]


def initial_fixture():
    """Read a deterministic 200 nm cell initialization from the input tables."""
    volume = (4.0 / 3.0) * np.pi * (200e-9) ** 3 * 1000.0
    tables = pd.read_excel(
        ROOT / "input_data" / "initial_concentrations.xlsx",
        sheet_name=["Simulation Medium", "Intracellular Metabolites", "Comparative Proteomics"],
    )
    properties = {
        "head_directory": str(ROOT) + "/", "volume_L": volume,
        "SA_total": 4.0 * np.pi * 200.0 ** 2, "counts": {}, "medium": {},
    }
    for _, row in tables["Simulation Medium"].iterrows():
        properties["medium"]["M_" + row["Met ID"]] = row["Conc (mM)"]
    for _, row in tables["Intracellular Metabolites"].iterrows():
        properties["counts"]["M_" + row["Met ID"]] = int(round(
            row["Init Conc (mM)"] / 1000.0 * 6.022e23 * volume
        ))
    for _, row in tables["Comparative Proteomics"].iloc[1:].iterrows():
        count = row["Sim. Initial Ptn Cnt"]
        if "S ribosomal" in row["Gene Product"]:
            count = max(25, int(count - 500))
        if row["Localization"] in ("trans-membrane", "peripheral membrane"):
            count = int(count)
        else:
            count = int(count / 2) + int(count / 6) + int(count / 3)
        properties["counts"]["P_" + row["Locus Tag"].split("_")[1]] = count
    protein_forms = pd.read_excel(
        ROOT / "input_data" / "protein_metabolites.xlsx", sheet_name="protein metabolites"
    )
    for _, row in protein_forms.iterrows():
        for metabolite in row["Metabolite IDs"].split(","):
            properties["counts"][metabolite] = 0
    return properties


def load_main_module(name, package, revision=None):
    path = ROOT / (name + ".py")
    source = path.read_text() if revision is None else subprocess.check_output(
        ["git", "show", revision + ":" + name + ".py"], cwd=ROOT, text=True
    )
    module = types.ModuleType(name + "_benchmark")
    module.__file__ = str(path)
    cvodes = types.ModuleType("pycvodes")
    cvodes.integrate_predefined = None
    cvodes.integrate_adaptive = None
    with mock.patch.dict(sys.modules, {"odecell": package, "pycvodes": cvodes}):
        exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def run_metabolism(rxns, integration, properties, backend="python", cache_dir=None):
    start = perf_counter()
    model = rxns.initModel(properties)
    model_time = perf_counter() - start
    start = perf_counter()
    if hasattr(integration, "buildSolver"):
        solver = integration.buildSolver(model, backend=backend, cache_dir=cache_dir)
    else:
        solver = integration.noCythonSetSolver(model)
    solver_time = perf_counter() - start
    start = perf_counter()
    results = integration.runODE(model.getInitVals(), solver, model)
    integration_time = perf_counter() - start
    timings = {
        "model_seconds": model_time, "solver_seconds": solver_time,
        "integration_seconds": integration_time,
        "total_seconds": model_time + solver_time + integration_time,
    }
    if results.shape != (100, len(model.getInitVals())) or not np.isfinite(results).all():
        raise AssertionError("The metabolic fixture did not produce 100 finite samples")
    fluxes = np.asarray([solver.calcFlux(0, row) for row in results])
    return timings, model, solver, results, fluxes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--odecell-dir", type=Path, default=ROOT.parent / "odecell")
    parser.add_argument("--baseline-main-ref", required=True)
    parser.add_argument("--baseline-odecell-ref", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--backend", choices=("python", "cython"), default="python")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output", type=Path, help="Write the JSON report as well as printing it")
    args = parser.parse_args()
    sys.path.insert(0, str(args.odecell_dir / "tests"))
    from solver_test_support import load_solver
    compiled_module = None
    compiler_calls = []
    if args.backend == "cython":
        builder, new_solver, compiled_module = load_solver(include_compiled=True)
        original_compile = compiled_module._compile_extension

        def record_compile(*arguments, **keywords):
            compiler_calls.append(arguments[2])
            return original_compile(*arguments, **keywords)

        compiled_module._compile_extension = record_compile
    else:
        builder, new_solver = load_solver()
    baseline_source = subprocess.check_output(
        ["git", "show", args.baseline_odecell_ref + ":odecell/solver.py"],
        cwd=args.odecell_dir, text=True,
    )
    _, old_solver = load_solver(baseline_source)
    packages = []
    for name, solver_module in (("before", old_solver), ("after", new_solver)):
        package = types.ModuleType("odecell_" + name)
        package.modelbuilder = builder
        package.solver = solver_module
        packages.append(package)
    old_rxns = load_main_module("Rxns_ODE", packages[0], args.baseline_main_ref)
    old_integration = load_main_module("Integrate", packages[0], args.baseline_main_ref)
    new_rxns = load_main_module("Rxns_ODE", packages[1])
    new_integration = load_main_module("Integrate", packages[1])
    if compiled_module is not None:
        # This benchmark owns its process. Register the already-loaded package
        # for Integrate's lazy import without removing native modules afterward.
        packages[1].__path__ = [str(args.odecell_dir / "odecell")]
        packages[1].compiled = compiled_module
        sys.modules["odecell"] = packages[1]
        sys.modules["odecell.compiled"] = compiled_module
    fixture = initial_fixture()
    start = perf_counter()
    warmup_model = new_rxns.initModel(fixture)
    cold_setup = perf_counter() - start
    start = perf_counter()
    warmup_solver = new_integration.buildSolver(warmup_model, backend=args.backend, cache_dir=args.cache_dir)
    cold_solver_setup = perf_counter() - start
    before_times, after_times = [], []
    previous_rhs = None
    changed_parameters_observed = True
    bit_identical = True
    max_state_error = 0.0
    max_state_error_per_tolerance = 0.0
    max_flux_error = 0.0
    max_scalar_rhs_error = 0.0
    max_scalar_flux_error = 0.0
    for repeat in range(args.repeats):
        properties = copy.deepcopy(fixture)
        properties["volume_L"] *= 1.0 + repeat * 0.01
        for key in properties["counts"]:
            if key.startswith("P_"):
                properties["counts"][key] += repeat
        medium_key = next(iter(properties["medium"]))
        properties["medium"][medium_key] *= 1.0 + repeat * 0.02
        before = run_metabolism(old_rxns, old_integration, properties)
        after = run_metabolism(new_rxns, new_integration, properties, args.backend, args.cache_dir)
        before_times.append(before[0])
        after_times.append(after[0])
        np.testing.assert_array_equal(before[1].getInitVals(), after[1].getInitVals())
        bit_identical &= np.array_equal(before[3], after[3]) and np.array_equal(before[4], after[4])
        max_state_error = max(max_state_error, float(np.max(np.abs(after[3] - before[3]))))
        max_state_error_per_tolerance = max(max_state_error_per_tolerance, float(np.max(
            np.abs(after[3] - before[3]) / (1e-12 + 1e-6 * np.abs(before[3]))
        )))
        max_flux_error = max(max_flux_error, float(np.max(np.abs(after[4] - before[4]))))
        if args.backend == "python":
            np.testing.assert_array_equal(before[3], after[3])
            np.testing.assert_array_equal(before[4], after[4])
            for callback in ("compiledCall", "calcFlux"):
                if before[2].localDic[callback].__code__ != after[2].localDic[callback].__code__:
                    raise AssertionError("Generated callback changed: " + callback)
        else:
            np.testing.assert_allclose(after[3], before[3], rtol=1e-6, atol=1e-12)
            if type(after[2]) is not type(warmup_solver):
                raise AssertionError("Numeric parameter updates changed the compiled class")
            # Compare the mathematical callbacks at identical states separately
            # from accumulated adaptive-integrator roundoff.
            for state in before[3]:
                reference_rhs = np.asarray(before[2](0.0, state))
                compiled_rhs = np.asarray(after[2](0.0, state))
                reference_flux = np.asarray(before[2].calcFlux(0.0, state))
                compiled_flux = np.asarray(after[2].calcFlux(0.0, state))
                np.testing.assert_allclose(compiled_rhs, reference_rhs, rtol=1e-12, atol=1e-14)
                np.testing.assert_allclose(compiled_flux, reference_flux, rtol=1e-12, atol=1e-14)
                max_scalar_rhs_error = max(max_scalar_rhs_error, float(np.max(np.abs(compiled_rhs - reference_rhs))))
                max_scalar_flux_error = max(max_scalar_flux_error, float(np.max(np.abs(compiled_flux - reference_flux))))
        rhs = after[2](0.0, np.asarray(after[1].getInitVals(), dtype=np.double))
        if previous_rhs is not None:
            changed_parameters_observed &= not np.array_equal(rhs, previous_rhs)
        previous_rhs = rhs
    report = {
        "python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__, "libsbml": libsbml.getLibSBMLDottedVersion(),
        "baseline_main_ref": args.baseline_main_ref,
        "baseline_odecell_ref": args.baseline_odecell_ref,
        "backend": args.backend,
        "metabolites": len(after[1].getMetList()), "reactions": len(after[1].getRxnList()),
        "samples_per_integration": 100, "repeats": args.repeats,
        "cold_after_model_seconds": cold_setup,
        "first_after_solver_seconds": cold_solver_setup,
        "first_solver_used_disk_cache": args.backend == "cython" and not compiler_calls,
        "before_median": {key: median(row[key] for row in before_times) for key in before_times[0]},
        "after_median": {key: median(row[key] for row in after_times) for key in after_times[0]},
        "trajectories_and_fluxes_bit_identical": bool(bit_identical),
        "generated_callbacks_identical": args.backend == "python",
        "state_comparison_rtol": 1e-6,
        "state_comparison_atol": 1e-12,
        "scalar_comparison_rtol": 1e-12,
        "scalar_comparison_atol": 1e-14,
        "max_state_absolute_error": max_state_error,
        "max_state_error_per_tolerance": max_state_error_per_tolerance,
        "max_trajectory_flux_absolute_error": max_flux_error,
        "max_scalar_rhs_absolute_error": max_scalar_rhs_error,
        "max_scalar_flux_absolute_error": max_scalar_flux_error,
        "compiler_invocations": len(compiler_calls),
        "changed_parameters_observed": changed_parameters_observed,
    }
    report["total_speedup"] = report["before_median"]["total_seconds"] / report["after_median"]["total_seconds"]
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()

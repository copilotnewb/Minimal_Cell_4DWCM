# Performance branch: one 16 GB GPU

This branch reduces repeated CPU work and changes the largest avoidable GPU
reaction table from a dense species-by-species array to a sparse pair table.
It also removes a device heap allocation from every firing lattice site.
Use the matching `perf/single-gpu-16gb` branches of **Minimal_Cell_4DWCM,
Lattice_Microbes, odecell, and btree_chromo_gpu** together.

The complete coupled simulation has **not** run on a GPU in the development
environment. A 16 GB fit and an end-to-end speedup are therefore unmeasured.
The CPU measurements below are reproducible component measurements. They must
not be multiplied together or interpreted as a whole-cell speedup.

## What changed

| Component | Change | Expected benefit |
| --- | --- | --- |
| LM integer RDME solver | Sparse second-order propensity table, with the old dense representation available for comparison | Removes the quadratic species table from host and device memory |
| LM integer reaction kernel | Occupancy-bounded local accounting replaces a species-sized `malloc`, copy, and `free` for each firing site | Removes device heap contention and temporary copying |
| Python lattice communication | Find all required cost particles in one lattice scan and group them in original coordinate order | Avoids hundreds of repeated scans per communication interval |
| Ribosome exclusion sites | Bulk Boolean writes and one native setter to mark the CUDA lattice dirty | Avoids thousands of Python/SWIG calls while ensuring upload |
| DNA loop movement | Evaluate only the consecutive candidate path for a bound hinge | At most 30 distance evaluations instead of a 54,338-monomer scan for the initial model's loop setting |
| Metabolic setup | Skip unused Jacobian construction; cache immutable workbook/SBML inputs | Avoids repeated parsing and quadratic work while refreshing live parameters |
| ODE and flux outputs | Preallocate the same output samples | Avoids growing and copying arrays on every sample |
| CME results | Read the final counts row once and validate the completed time | Avoids repeated trajectory reads and surfaces failed native runs |
| Process coordination | Explicit GPU selection, owned chromosome workers, checked output and failure cleanup | Makes the one-GPU configuration explicit and avoids consuming incomplete DNA output |

The physical lattice spacing, RDME timestep, coupling schedule, particle
capacity, Brownian dynamics parameters, and default LSODA tolerances are
unchanged. The default Python metabolic callbacks and tested trajectories
match their original outputs exactly. The optional Cython backend is discussed
separately below.

No source changes were needed in `sc_chain_generation` or `FreeDTS` for the
validated optimizations. The driver now accepts both
`sc_chain_generation/src/gen_sc_chain` and the legacy `fortran/` location,
and both `btree_chromo_gpu/` and `btree_chromo/` directory names.
When both built checkouts are present, `btree_chromo_gpu/` takes precedence.
The worker launches use argument lists, but unrelated legacy shell file moves
still exist elsewhere in the model; use paths without spaces for a full run.

## Where the GPU memory saving comes from

Let `S` be the model's number of species, `T` its number of site types, and `P`
the number of distinct unordered reactive species pairs. The second-order
propensity table uses the following storage on the device (and corresponding
host storage):

| Representation | Bytes |
| --- | ---: |
| Original dense | `4 * T * S * S` |
| Sparse values plus pair indices | `4 * T * P + 4 * P + 4 * (S + 1)` |

Initially zero-rate reactions are included in the sparse topology, so later
rate updates remain valid. Duplicate and self-reaction aggregation, particle
iteration, and floating-point accumulation order are preserved. Sparse lookup
adds a short indexed search; the GPU throughput tradeoff needs measurement.

The initial `64 x 64 x 128` lattice at 16 four-byte particle slots uses about
64 MiB for its two particle buffers and 0.5 MiB for the site map. The complete
budget also includes the remaining reaction/diffusion tables, random-number
state, CUDA contexts, overflow/profiling buffers, LAMMPS allocations, and any
other GPU applications. The lattice dimensions alone do not predict peak VRAM.

After the simulation has finalized `MinCell.lm`, inspect its actual counts:

```sh
python ../Lattice_Microbes/utils/int_mpd_memory.py Data/perf_python/MinCell.lm
python ../Lattice_Microbes/utils/int_mpd_memory.py Data/perf_python/MinCell.lm --json
```

This read-only tool does not need a GPU. Its known-array estimates are not an
allocator-exact peak. The rebuilt solver also prints its actual table sizes
during initialization. See the
[LM performance guide](https://github.com/copilotnewb/Lattice_Microbes/blob/perf/single-gpu-16gb/docs/int_mpd_performance.md)
for formulas, kernel details, and standalone checks.

## Rebuild the changed dependencies

Activate the environment that already contains the simulation's dependencies.
The following are **rebuild** instructions; they do not replace the projects'
CUDA, HDF5, protobuf, MPI, LAMMPS/Kokkos, and Python installation instructions.
Use a CUDA toolkit and host compiler compatible with your GPU and existing
environment. A GPU's VRAM capacity does not identify its CUDA architecture.

Fetch and check out `perf/single-gpu-16gb` in each of the four changed forks.
If using existing clones, preserve any local changes first. For example, in
each clean checkout:

```sh
git fetch origin
git switch perf/single-gpu-16gb
```

### Lattice Microbes

Rebuild and install the native library and Python bindings. From its checkout,
use a separate build directory and your GPU's architecture, for example:

```sh
cmake -S src -B build-perf \
  -D CMAKE_BUILD_TYPE=Release \
  -D MPD_GLOBAL_R_MATRIX=ON \
  -D MPD_GLOBAL_S_MATRIX=ON \
  -D MPD_GLOBAL_T_MATRIX=ON \
  -D CUDA_ARCHITECTURES=YOUR_GPU_ARCH
cmake --build build-perf --parallel
cmake --install build-perf
python -c 'import lm; print(lm.__file__)'
```

Replace `YOUR_GPU_ARCH` with the correct architecture value; do not run that
placeholder literally. LM's CMake configuration derives its install prefix
from the active Python environment. Retain any additional options your
existing working installation requires. Merely updating the checkout does
not update an already installed `_lm` library.

### odecell

Install the changed package in the **same environment** as LM. From the
`odecell` checkout, an environment whose dependencies are already satisfied
can use:

```sh
python -m pip install --no-deps --no-build-isolation -e .
python -c 'import odecell; print(odecell.__file__)'
```

The package's existing native extension still needs its original build
dependencies. The optional runtime compilation described below additionally
uses Cython, NumPy headers, setuptools, and a working Unix C compiler.

### btree_chromo_gpu

First configure the Makefile's MPI, fmt, and modified LAMMPS library/header
paths for your installation. This repository contains old compiled files, so
clean them before rebuilding:

```sh
make clean
make release
```

The resulting executable is `build/apps/btree_chromo`. Release builds now
use `-O2` and C++17 without fast-math flags. The custom LAMMPS/Kokkos installation
and its DNA model additions remain required. The CPU search optimization does
not replace those dependencies.

The directory passed to `--dnaSoftwareDirectory` should contain both
`btree_chromo_gpu/` (or `btree_chromo/`) and `sc_chain_generation/`. Build the
latter with its existing instructions if its executable is absent.

## First hardware run

Run from the `Minimal_Cell_4DWCM` checkout. Use a new output name for each
comparison. This eight-second smoke run includes several metabolic intervals;
initial chromosome generation can still take substantial wall time:

```sh
python tools/benchmark_run.py --output benchmarks/single_gpu_python.json -- \
  python Whole_Cell_Minimal_Cell.py \
    -od perf_python -t 8 -cd 0 -drs 13 \
    -dsd /absolute/path/to/software \
    --dnaTimeout 1800
```

The Python ODE backend is the default. The DNA worker and the separate CME
process now use the same selected GPU as RDME unless a separate DNA GPU is
requested. If a scheduler/container already sets `CUDA_VISIBLE_DEVICES`, the
ordinals refer to that visible list. For example, with only physical GPU 3
visible, `-cd 0` selects it; do not replace the scheduler's visibility mask.

The wrapper writes:

- `benchmarks/single_gpu_python.json`: total wall time, observed biological
  duration, per-stage timing summaries, exit status, and per-device sampled
  memory peaks.
- `benchmarks/single_gpu_python.log`: combined simulation output.
- `benchmarks/single_gpu_python.gpu.csv`: raw `nvidia-smi` samples, when available.

Memory samples are device-wide, include other processes, and can miss brief
peaks between samples. Close other GPU workloads for comparisons. Startup
time is included in total wall time; the stage summaries help distinguish
startup cost from repeated hooks. The `rdme_between_hooks` timer also includes
native output work between callbacks and should not be read as kernel time.

An eight-second success does not exercise late replication or division.
Before a long production run, test a restart and a longer run through the
memory-heavy phases relevant to your experiment. Preserve the original
installation or another environment for a baseline comparison.

### DNA overlap and separate devices

`--dnaMode overlap` is the default. `--dnaMode serial` waits for the chromosome
worker before RDME continues, which is useful for measuring contention on one
GPU. **Serial mode does not release the parent's RDME CUDA allocations.** It
cannot by itself guarantee that two contexts fit in memory, and may be slower.

`--dnaCudaDevice N` optionally selects another GPU from the parent's visible
list. The child is given a one-device visibility mask; its internal Kokkos
device zero then refers to that selected device. The original single-GPU
Kokkos launch and CME random-number generator are retained.

`--dnaTimeout` sets how long a synchronization point waits, rather than how
long the entire simulation may run. A timeout or nonzero worker exit is an
error. The latest worker output is in `Data/<outputDir>/DNA/worker.log` or
`Data/<outputDir>/CME/worker.log`. Failed DNA updates no longer silently reuse
an old chromosome configuration.

Before consuming DNA, the driver checks the exact coordinate-file size against
the topology's monomer count, including on restart and in division. Ordinary
hook errors cancel and reap an outstanding worker before LM's native callback
exits Python. The existing `--maximumHours` controlled stop instead lets the
pending checkpoint chromosome finish and validates it before stopping. After
a normal RDME return, a final HDF5 schedule/frame check catches native errors
that LM logs without raising into Python.

An intentional `--maximumHours` stop still uses LM's existing nonzero exit
status (normally 1); check the time-limit message and checkpoint results when
interpreting the timing report.

### Dense table comparison

The old dense propensity representation remains available:

```sh
LM_INTMPD_SECOND_ORDER_MODE=dense \
python tools/benchmark_run.py --output benchmarks/single_gpu_dense.json -- \
  python Whole_Cell_Minimal_Cell.py \
    -od perf_dense -t 8 -cd 0 -drs 13 \
    -dsd /absolute/path/to/software --dnaTimeout 1800
```

Only try this if the model audit indicates the dense allocation will fit.
This compares table representations in the patched solver; the shared
reaction-kernel improvements still apply in dense mode. For a full baseline,
use the original commits listed below and separately rebuilt dependencies.

## Optional compiled metabolic callbacks

`--odeBackend cython` generates a parameterized Cython functor, compiles it on
first use, and reuses the binary when its generated source and ABI/build key
match. Every metabolic interval receives fresh parameter values in its own
array. Structure, reaction formulas, stoichiometry, species order, unlifted
constants, Python/NumPy/Cython identity, and compile options influence reuse.
The cache is outside the installed package, allowing read-only installations.

In the real-network fixture, median model setup, callback preparation and
integration together fell from **3.972 s to 0.7085 s** after warm-up (5.61x).
Integration alone fell from **3.150 s to 0.2304 s** (13.68x). All three tested
parameter sets produced bit-identical 100-sample trajectories and flux tables.
Flux validation and CSV I/O are outside these timed sections. These results
measure the metabolic component, not a complete whole-cell interval.

First callback compilation took **72.47 s** on the development host. It has an
upfront cost that must be amortized across subsequent intervals. The default cache is
`Data/<outputDir>/ode_cache`; use `--odeCacheDirectory /path/to/cache` to share
it across runs. Compilation errors include the saved build-log path. When
the environment's configured compiler is unavailable, select an installed,
compatible compiler through `CC`, for example `CC=gcc` on the tested Linux
development host. No `-ffast-math` is enabled; contraction is disabled for
these callbacks. LSODA's method, tolerances, and requested output times remain
the same.

Share a cache only across compatible host/toolchain configurations. The key
records configured compiler strings and flags, but does not fingerprint the
actual compiler binary or all CPU features. Use separate cache directories
after toolchain changes or when using native CPU flags on different machines.

Compare it in a separate output directory after the Python smoke run. This
example uses a shared local cache so subsequent fresh-output runs can reuse
the first compilation:

```sh
python tools/benchmark_run.py --output benchmarks/single_gpu_cython.json -- \
  python Whole_Cell_Minimal_Cell.py \
    -od perf_cython -t 8 -cd 0 -drs 13 \
    -dsd /absolute/path/to/software --dnaTimeout 1800 \
    --odeBackend cython --odeCacheDirectory "$PWD/ode_cache"
```

For steady-state timing, reuse that cache with another output name and a
longer run. An eight-interval cold run may spend more time compiling than it
saves; compare warm runs separately from first-use wall time.

### Restart smoke test

After the compiled smoke run completes, continue it for eight more biological
seconds, repeating the intended runtime settings:

```sh
python tools/benchmark_run.py --output benchmarks/single_gpu_restart.json -- \
  python Restart_Whole_Cell_Minimal_Cell.py \
    -od perf_cython -t 8 -cd 0 -drs 13 \
    -dsd /absolute/path/to/software --dnaTimeout 1800 \
    --odeBackend cython --odeCacheDirectory "$PWD/ode_cache"
```

The restart CLI applies its current defaults rather than inheriting the saved
ODE backend, cache location, DNA mode, timeout, and device choices. Repeat any
nondefault settings you want to keep. The `-t` value is additional elapsed
simulation time, not the desired cumulative endpoint.

The exact agreement above applies to the tested fixtures and toolchain.
Compiled floating-point evaluation on another toolchain can change adaptive
solver decisions, so this backend is opt-in. Its numerical comparison and
measured build/runtime costs are recorded in the ODE benchmark results. Do not use the DNA seed as
a claim of a fixed seed for the entire coupled stochastic simulation: it
controls the DNA component, while RDME/CME have their own random states.

## Reproducing the available checks

From this checkout, with NumPy, SciPy, pandas, h5py and the existing model
dependencies available:

```sh
python -m unittest discover -s tests -v
python benchmarks/benchmark_lattice.py
python benchmarks/benchmark_ode.py --help
```

The lattice tests cover exact coordinate ordering, duplicates and missing
species, non-cubic axes, final reaction-cost counters/particles, overlapping
region priorities, and invalidating the CUDA copy after bulk site writes.
Process tests use actual short-lived subprocesses. HDF5 fixtures exercise
final counts and incomplete-output handling.

Additional component checks:

```sh
# In odecell:
python -m unittest discover -s tests -v

# In Lattice_Microbes:
python -m unittest discover -s testing/host -v

# In btree_chromo_gpu:
python tests/run_cpu_tests.py
python tests/run_cpu_tests.py --benchmark
```

The btree suite compares 18,000 randomized candidate cases and 38,416 complete
loop states against the original implementation. LM tests compile the actual
host/device helper logic and test sparse/dense float equality, rate refresh,
buffer lifecycle, overflow extent, and 5,000 randomized valid stoichiometries
in each stoichiometric-storage mode.

The changed integer reaction kernel also compiled and assembled with NVIDIA
NVRTC/ptxas 12.9 for `compute_75`/`sm_75`, in both global and constant reaction
table configurations. Both used 72 registers and no stack or spills in that
check. This is a focused device compilation check, **not** a full LM
build/link, a GPU execution test, or certification of another CUDA version.
The LM guide gives the reproduction command and hardware validation steps.

### Saved CPU measurements

Raw measurements, settings, and scope are in
[`benchmarks/results/`](../benchmarks/results/). Lattice results use a
synthetic default-size lattice with 100,000 occupied slots and 200 queried
species; the site-setter baseline is a Python stand-in for SWIG. DNA search
results use a 54,338-monomer region, 500 queries, a 30-step extension bound,
and median-of-five timing. Timings vary on the shared CPU host.

| CPU routine in the saved fixture | Original | Changed | Local ratio |
| --- | ---: | ---: | ---: |
| Locate particles for 200 species | 8.242 s | 0.0780 s | 105.7x |
| Update ribosome site masks | 137.8 ms | 4.82 ms | 28.6x |
| DNA candidate queries, 500 hinges | 449.3 ms | 0.758 ms | 592.5x |
| Metabolic setup and integration, Python | 4.528 s | 3.840 s | 1.18x |
| Metabolic setup and integration, warm Cython | 3.972 s | 0.7085 s | 5.61x |

The DNA result is only its candidate-search routine. The lattice rows include
the specified synthetic data and Python-call baseline; none measures a GPU
kernel or complete cell trajectory.
The two metabolic rows are separate benchmark runs; their original timings
differ with CPU load. Each ratio compares the before/after pair in its own
saved report.

The real metabolic fixture uses the repository's input workbooks and SBML
network: 149 metabolites and 175 reactions. The Python-path comparison checks
generated callbacks, 100-sample LSODA trajectories, and 100-sample flux tables
under three parameter sets. See the saved result for model setup, callback
construction, integration, and cold-compilation timings. Flux tables are
validated outside the timed region. These are metabolic component measurements
rather than entire whole-cell runs.

### Baseline revisions

| Repository | Baseline commit |
| --- | --- |
| Minimal_Cell_4DWCM | `08f5f2080dc5d766448b02e20c74a6d04c737f94` |
| Lattice_Microbes | `bc58524c2d76c1b84313672b72b140f7f40b7819` |
| odecell | `b19413f2f8c1a88d1fe619e09ad9beb7953ccb96` |
| btree_chromo_gpu | `fbc6f01b10a2c169e439c541f71e5b2a413fd0c9` |

## What the hardware results should answer

Record the GPU model, VRAM, CPU, CUDA version, installed library locations,
command line, JSON/log/CSV files, and worker logs on a failure. The next
optimization target should follow the observed stage times and memory peak:
RDME kernels, repeated CME setup, DNA/LAMMPS, metabolic callbacks, or output.
Useful follow-up work includes sparse product stoichiometry and launch/output
profiling, but those changes need real GPU timing and ordering validation.
No numerical coarsening or unsupported whole-model speedup is assumed here.

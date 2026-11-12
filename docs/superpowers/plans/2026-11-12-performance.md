# Single GPU Performance Implementation Plan

**Goal:** Reduce whole-cell runtime and GPU memory for a 16 GB device, preserving scientific equations, time steps, tolerances, and reaction ordering.

**Architecture:** Optimize representations and repeated setup in the existing Python, CUDA, and C++ components. Keep GPU-dependent validation explicitly pending the user's machine. Work on `perf/single-gpu-16gb` in each affected fork; publish reviewable branches and draft pull requests.

**Authorization:** The user requested autonomous implementation in all six forks and will perform the full simulation test. Implementation decisions below are engineering choices, not measured GPU results.

## Constraints and alternatives

- Preserve the 10 nm lattice, 50 microsecond RDME step, 250-step hooks, solver tolerances, physical parameters, and random draw ordering.
- Prefer sparse reaction data and bounded exact searches over coarser simulation or lower precision.
- Keep model parameter refresh each metabolic interval; caching a fully parameterized solver would freeze live enzyme/cost inputs.
- Separate CPU microbenchmarks from end-to-end GPU performance. No claim of a 500x whole-model speedup or guaranteed 16 GB fit without a hardware run.
- Leave sc_chain_generation and FreeDTS unchanged if inspection reveals no safely verifiable hot-path improvement.

## Work

- [x] Lattice Microbes: remove per-reacting-site device heap allocation with equivalent local accounting; replace quadratic second-order propensity storage for IntMpdRdmeSolver with exact sparse pair lookup, keep dense fallback, fix redundant model uploads, test shared host/device lookup and legacy aggregation semantics.
- [x] btree_chromo_gpu: replace whole-chromosome scans for bound loop movement with the exact bounded candidate search; differential tests must preserve candidates and RNG sequence; compile optimized release by default without fast math.
- [x] odecell / Integrate.py / Rxns_ODE.py: skip unused Jacobian construction, avoid repeated trajectory reallocations, cache immutable file inputs while refreshing all live model parameters. Compare RHS/fluxes and real LSODA trajectories.
- [x] Main lattice communication: group selected particle locations in one lattice scan; vectorize ribosome site writes through the existing writable NumPy view. Test duplicates, absent species, non-cubic axes, overlapping region precedence, multiple ribosome types, and final particle/cost states. Mark the CUDA lattice dirty with one native setter after bulk edits.
- [x] Main process coordination: explicit DNA GPU mapping through CUDA_VISIBLE_DEVICES; retain/reap child processes, wait for actual completion, check failures and output completeness; resolve both fork and legacy executable directory layouts. Test real short-lived subprocesses, partial output, nonzero exit, timeout cleanup, and worker argv paths containing spaces. Validate topology counts and persisted RDME output; handle native callback exits and controlled wall-time stops.
- [x] Add an opt-in Cython metabolic backend with source-keyed compilation and fresh full parameter vectors; keep Python as the default and compare real-network trajectories/fluxes under changed parameters. Preserve numeric rendering, unary-sign precedence, and derivative accumulation order.
- [x] Add repeatable CPU benchmarks and hardware-run instructions with an opt-in timing/VRAM recorder. Review cross-repository compatibility, run available checks, and prepare reviewable changes on the performance branches.
- [ ] Target-machine follow-up: rebuild/link the full native stack, execute the coupled simulation, measure peak VRAM and end-to-end throughput, test restart and late replication/division.

## Review focus

Zero/one/multiple reactions per pair; homodimer aggregation; exact floating point assignment order; CUDA resource cleanup and indexing; checkpoint serialization excludes process handles; restart without a live child; CUDA ordinal remapping under scheduler visibility; complete DNA output before consumption; lattice axis order and compaction with duplicate particles; cache invalidation and fresh ODE parameters.

## Validation record

See `docs/PERFORMANCE.md`, the raw `benchmarks/results/` JSON files and the
component pull requests. Available validation includes 44 main CPU tests,
6 interpreted and 11 compiled odecell tests, 8 LM host tests, NVIDIA
NVRTC/ptxas compilation of the changed kernel, and exact btree candidate and
trajectory comparisons. GPU execution remains explicitly pending.

# MC4D

This repository contains the 4D whole-cell model for the genetically minimal cell, JCVI-syn3A. Below, you will find instructions on required programs that need to be installed, instructions on how to run the model, and descriptions of individual simulation files.

## Performance branch and one-GPU testing

The coordinated `perf/single-gpu-16gb` branches of this fork, Lattice Microbes,
odecell, and btree_chromo_gpu include sparse GPU reaction storage, faster
lattice communication and DNA loop searches, and an optional compiled
metabolic backend. See [the performance and rebuild guide](docs/PERFORMANCE.md)
for measured CPU results, validation limits, single-GPU commands, and the
timing/VRAM recorder. A full GPU run and a 16 GB fit still require testing on
the target machine.

## Required Programs

- [Lattice Microbes](https://github.com/copilotnewb/Lattice_Microbes) — use branch `perf/single-gpu-16gb` and rebuild the native library and Python bindings.
- [odecell](https://github.com/copilotnewb/odecell) — use branch `perf/single-gpu-16gb`; install in the same conda environment as Lattice Microbes after building LM.
- [btree_chromo_gpu](https://github.com/copilotnewb/btree_chromo_gpu) — use branch `perf/single-gpu-16gb`; requires the modified Kokkos-enabled LAMMPS installation.
- [sc_chain_generation](https://github.com/copilotnewb/sc_chain_generation) — existing default branch; unchanged by this performance work.
- [FreeDTS](https://github.com/copilotnewb/FreeDTS) — optional; existing default branch, unchanged by this performance work.

## Running the Model

The Python model runs directly from this checkout. After installing and rebuilding the required dependencies, activate the conda environment in which you built Lattice Microbes:

```
conda activate envName
```

Then, you must make sure that your LAMMPS installation for ```btree_chromo``` is in your path. You can check this by running:

```
lammps -h
```

Once your environment is ready, you can now run the model.

The python file ```Whole_Cell_Minimal_Cell.py``` is the main executable for the model. The executable has the following user input variables:

|  Variable  | Shorthand | Description |
|------------|-----------|-------------|
| --outputDir | -od | Name of directory that will be created to store trajectory files. DO NOT INCLUDE A '.' |
| --simTime | -t | Amount of biological time to be simulated in units of seconds. |
| --cudaDevices | -cd | Integer index within the visible GPU list for RDME and CME; also the default DNA GPU. |
| --dnaSoftwareDirectory | -dsd | Directory containing `btree_chromo_gpu/` (preferred, or legacy `btree_chromo/`) and `sc_chain_generation/`. |
| --dnaRngSeed | -drs | Integer RNG seed that will be used for the chromosome programs. |
| --workingDirectory | -wd | Directory where the simulation is being run in. Defaults to current directory, no input is needed. Only necessary to change on clusters. |
| --dnaCudaDevice | | Optional DNA GPU ordinal within the parent's visible device list. |
| --dnaMode | | `overlap` (default) or `serial`; serial waits for DNA but retains RDME device allocations. |
| --dnaTimeout | | Seconds to wait for a DNA worker at synchronization (default 300). |
| --odeBackend | | `python` (default) or opt-in `cython` callbacks. |
| --odeCacheDirectory | | Optional shared compiled ODE cache; default is the output directory's `ode_cache/`. |

Example executable:

```
python Whole_Cell_Minimal_Cell.py -od replicate1 -t 1200 -cd 0 -drs 13 -dsd /absolute/path/to/software/
```

## Restarting a simulation

If you want to run more time for a simulation or you need to restart a simulations because of a crash, you can run the available restart python script ```Restart_Whole_Cell_Minimal_Cell.py```

This python executable has the same input variables as the original script. This will only run if ```outputDir``` is a directory that contains all associated files with a previously run simulation from the main script. For example, if you provided the input ```-od replicate1``` previously, giving that same argument to the restart script will take the simulation state from the replicate1 directory.

The ```simTime``` variable is how much more biological time you want to run. If you have simulated 3600 seconds and want to reach a total of 6000 seconds, you would give an input of 2400 for this variable.

## Descriptions of Simulation Files

| File Name | Description |
|-----------|-------------|
|```input_data/``` | Directory containing initial conditions, kinetic parameters, and other data used to initialize the whole-cell model. |
| ``` Communicate.py ``` | Procedures used to communicate counts of molecules between different methodologies. For example, extraction of enzyme counts from RDME to use in rates of ODE reactions. |
| ``` Diffusion.py ``` | Sets the diffudion rules for all molecules that are put into the RDME lattice. |
| ``` Division.py ``` | Updates cell morphology and particle positions on the RDME lattice during cell division. |
| ``` FileSaving.py ``` | Records the cell state and generates restart files. |
| ```FreeDTS_functions.py``` | Defines set of functions used to interpret output files of membrane shapes from FreeDTS. |
| ``` GIP_rates.py ``` | Contains functions for rate equations of genetic information processing reactions. |
| ``` Growth.py  ``` | Updates cell morphology and particle positions on the RDME lattice as the cell grows spherically. |
| ``` Hook.py ``` | IMPORTANT Defines main algorithm used when interrupting the RDME solver. Includes executing all other simulations methods and calling functions to communicate betwwen the methods. |
| ``` ImportInitialConditions.py ``` | Loads initial condition data into initialize the system. |
| ``` InitRdmeDNA.py ``` | Creates an initial condition chromosome configuration using ```sc_chain_generation``` and communicates the structure to the RDME lattice. |
| ``` Integrate.py ``` | Runs the ODE integrator for metabolism. |
| ``` LatticeFunctions.py ``` | Defines functions used in communication to manipulate the RDME lattice. |
| ``` MC_CME.py ``` | Creates and runs the global CME simulation for tRNA charging and transcription. |
| ``` MC_RDME_initialization.py ``` | Initializes RDME simulation including site types and reactions. |
| ``` RegionsAndComplexes.py ``` | Initializes shapes of cell regions (e.g. membrane and cytoplasm) onto the RDME lattice. |
| ```Restart_Hook.py``` | Modified version of ```Hook.py``` that is used for simulations that have been restarted. |
| ```Restart_MC_RDME_initialization.py``` | Constructs a whole-cell simulation from the last recorded cell state of a preexisting simulation. |
| ```Restart_Whole_Cell_Minimal_Cell.py``` | Executable for restarting simulations. |
| ``` RibosomesRDME.py ``` | Updates excluded volume of ribosomes so that the ribosomes lattice sites follow the center of mass particle. |
| ``` Run_CME.py ``` | Executable file for global CME. |
| ``` Rxns_CME.py ``` | Defines set of reactions simulated in the global CME. |
| ``` Rxns_ODE.py ``` | Defines set of reactions simulated using deterministic ODEs. |
| ``` Rxns_RDME.py ``` | Defines set of reactions simulated on the RDME lattice. |
| ``` SpatialDnaDynamics.py ``` | Procedures for running ```btree_chromo``` and updating the chromosome state on the RDME lattice. |

## Simulation Files and Input Data by Model Component

| Model Component | Simulation Files | Input Data |
|---------------|-------------|-------------|
|RDME - Reaction-Diffusion Master Equaion| ``` MC_RDME_initialization.py ```, ``` Rxns_RDME.py ```, ``` Diffusion.py ```, ```RegionsAndComplexes.py```, ```RibosomesRDME.py```, ```InitRdmeDna.py``` | ``` oneParamMulder-local_min.json ```, ``` LargeSubunit.xlsx ```|
|CME - Chemical Master Equation|``` MC_CME.py ```, ``` Run_CME.py ```, ``` Rxns_CME.py ``` | ``` kinetic_params.xlsx ```|
|ODE - Ordinary Differential Equations| ``` Rxns_ODE.py ```, ```integrate.py```| ``` kinetic_params.xlsx ```, ``` protein_metabolites.xlsx ```, ``` Syn3A_updated.xml ```|
|BD (DNA) - Brownian Dynamics of DNA| ``` InitRdmeDNA.py ```, ``` SpatialDnaDynamics.py ```| ``` in_BD_lengths_LAMMPS_test.txt ```, ``` loop_params.txt ```, ``` syn3A.gb ```|


## Connectivity of 4DWCM Python Modules

![plot](./4DWCM_Dependency.png)


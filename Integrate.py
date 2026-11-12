"""
Authors: Zane Thornburg, David Biancho

Integrate system of ODE reactions
"""

from pycvodes import integrate_predefined
from pycvodes import integrate_adaptive
from scipy import integrate
import odecell
import numpy as np


### Constants
step = 0.1 # s
atol = 1e-6 # tolerance
rtol = 1e-6 # tolerance


def buildSolver(model, backend='python', cache_dir=None):
    """Select the reference Python callback or the opt-in cached Cython one."""
    if backend == 'python':
        return noCythonSetSolver(model)
    if backend == 'cython':
        from odecell.compiled import buildSolver as buildCompiledSolver
        return buildCompiledSolver(model, cache_dir=cache_dir)
    raise ValueError("Unknown ODE backend: {!r}".format(backend))


#########################################################################################
def setSolver(model):
    """
    Set the solver for the model

    Parameters:
         
        (odecell.model) - the model object

    Returns:

        solvFunctor - a functor for the solver

    """

    ## We are NOT building for odeint (gives us more room to chose between CVODES and SciPy-ODE).
    ## We are NOT using a jacobian, since we do not have the partial derivatives for all rate forms.
    ## We are building with Cython for speed, this is a big model.

    # Builds the solver using a *Functor* interface
    solvFunctor = odecell.solver.ModelSolver(model) 
    solvFunctor.prepareFunctor() #OG uncomment

    # Set verbosity to 0 for now, below uncomment OG
    rxnIdList = solvFunctor.buildCall(odeint=False, useJac=False, cythonBuild=True, functor=True, verbose=0)
    #rxnIdList = solvFunctor.buildCall(odeint=True, useJac=False, cythonBuild=False, functor=False, transpJac=False, verbose=0, noBuild=True)

    # Sets up the actual solver, with updated parameter values
    #modelOptSpace, initParamVals=model.getOptSpace()
    initParamVals = model.getInitVals()
    solvFunctor = solvFunctor.functor( np.asarray(initParamVals, dtype=np.double) )

    return solvFunctor
#########################################################################################


#########################################################################################
### NOTE: Have to get a callable f(y,t) for scipy.ode without creating the functor
def noCythonSetSolver(model):
    """
    Set the solver without compiling via Cython

    Parameters:

    model (odecell Model object): The model object

    Returns:

    solver (odecell Solver object): The Solver object, to solve the system of ODEs representing metabolic reactions
    """

    # Construct a Model Solver Object
    solver = odecell.solver.ModelSolver(model)

    rxnIdList =solver.buildCall(verbose=0, useJac=False, transpJac=0, nocheck=False, odeint=False, cythonBuild=False, functor=False, noBuild=True)

    return solver 
#########################################################################################


#########################################################################################
def f_wrap(solv, t, y, dydt):

    #solv = setSolver(model)
    dydt[:] = solv(0,np.asarray(y))[:]
#########################################################################################


#########################################################################################
def getInitVals(model):
    y0=model.getInitVals()
#     print(y0)
    return y0
#########################################################################################

# def noCythonRunODE():
#     """
#     Run the ODE Model without compiling via Cython

#     Parameters:

#     Returns:
#     None
#     """
#     return 0

#########################################################################################
def runODE(y0, solv, model):
    """
    Run the ODE Model after getting initial conditions

    Parameters:

    y0 (seems non-necessary) - can remove
    time (float): the current hybrid simulation time
    delt (float): the communication timestep between stochastic and deterministic simulation
    ts (float): the timestep for the adaptive ODE Solver
    solv (odecell Solver object): The solver object, with call built
    model (odecell Model object): The model object

    Returns:

    results (np.array): the array containing ODE Simulation Results (Maybe only the last time should be passed?)
    """

    #y0 = model.getInitVals()
    #print("shape: ",len(y0))
    #y0 = np.asarray(y0,dtype=np.double)

    #modelOptSpace,initParamVals=model.getOptSpace()
    #dydt = np.zeros(len(y0))
    #tout, results, info = integrate_adaptive(f_wrap(solv,time+delt,y0,dydt),None,y0,time,time+delt,atol,rtol,dx0=ts,nsteps=10000)
    #tout, results, info = integrate_adaptive(f_wrap(solv,time,y0,dydt),None,y0,time,time+delt,atol,rtol,dx0=ts,nsteps=10000)

    #tout = 0.0
    #info = "place holder"

    #solv = solv.ModelSolver(model)
    #solv.buildCall(odeint=False, useJac=False, verbose=2)
    integrator = integrate.ode(solv)#, solv.calcJac)

    # TODO: Set which integrator to use (Appears vode is default), lsoda adaptive
#     lsodaBool = True
#     if (lsodaBool):
    integrator.set_integrator("lsoda")

    initial_values = model.getInitVals()
    integrator.set_initial_value(initial_values)

    ### With fixed timestepping
    step = 0.01
    totalTime = 1.0
    # Keep the existing target times (including their floating-point rounding)
    # while avoiding a copy of every prior sample at each output step. The spare
    # row covers a final step caused by accumulated floating-point error.
    results = np.empty((int(np.ceil(totalTime / step)) + 1, len(initial_values)), float)
    sample_count = 0

    while integrator.successful() and integrator.t < totalTime:
        currConcentration = integrator.integrate(integrator.t + step)
#         print(integrator.t)
        # Silence integrator output for now
        #print(integrator.t, currConcentration)
        results[sample_count] = currConcentration
        sample_count += 1

#     results = results[-1,:]

    # Return only the last timestep for the results, this is all thats really needed
    return results[:sample_count]
#########################################################################################

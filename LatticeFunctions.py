"""
Authors: Zane Thornburg

General lattice functions
"""

import numpy as np


def particleLocations(particles, particle_ids):
    """Locate several species with one membership scan of the particle lattice.

    Each result has the same coordinates and C-order as ``argwhere(particles ==
    pid)``. Stable ordering matters when consumers remove/reinsert particles at
    a site. Callers may reuse spatial coordinates after compaction, but must not
    treat the stored word/slot coordinates as live references after a mutation.
    """
    particle_ids = list(dict.fromkeys(int(pid) for pid in particle_ids))
    if not particle_ids:
        return {}
    locations = {pid: np.empty((0, particles.ndim), dtype=np.intp)
                 for pid in particle_ids}
    coordinates = np.argwhere(np.isin(particles, particle_ids))
    if not len(coordinates):
        return locations
    labels = particles[tuple(coordinates.T)]
    order = np.argsort(labels, kind='stable')
    labels = labels[order]
    coordinates = coordinates[order]
    starts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1]
    stops = np.r_[starts[1:], len(labels)]
    for start, stop in zip(starts, stops):
        locations[int(labels[start])] = coordinates[start:stop]
    return locations

# From Tyler Biopolymers
# def deleteParticle(particles, x, y, z, pid):
#     ps = np.array([p for p_ in particles[:,z,y,x,:] for p in p_ if p != pid])
#     pps = 16 # Particles Per Site
# #     pps = cfg.particlesPerSite # Don't know how Tyler added pps as argument to his config
#     ps.resize(pps)
#     ps = ps.reshape((particles.shape[0], particles.shape[4]))
#     particles[:,z,y,x,:] = ps
    
#     return None


from math import floor
from math import log10
def round_sig(x, sig=2):
    """
    Inputs:
    Returns:
    Called by:
    Description:
    """

    negative = False
    if x < 0:
        negative = True
    x = abs(x)
    if negative:
        return -1*round(x, sig-int(floor(log10(abs(x))))-1)
    elif x==0.0:
        return 0.0
    else:
        return round(x, sig-int(floor(log10(abs(x))))-1)


def deleteParticle(particles, x, y, z, pid):
    """
    Inputs:
    Returns:
    Called by:
    Description:
    """

    ps = [p for p_ in particles[:,z,y,x,:] for p in p_ if p != 0]
    if pid in ps:
        ps.remove(pid)
        ps = np.array(ps)
        pps = 16 # Particles Per Site
        ps.resize(pps)
        ps = ps.reshape((particles.shape[0], particles.shape[4]))
        particles[:,z,y,x,:] = ps
    
    return None


def checkParticle(particles, x, y, z, pid):
    """
    Inputs:
    Returns:
    Called by:
    Description:
    """

#     print(x,y,z)
    ps = np.array([p for p_ in particles[:,z,y,x,:] for p in p_ if p == pid])
    if pid in ps:
#         print(ps)
        return True
    else:
        return False
    
    
def getParticlesInSite(particles, x, y, z):
    """
    Inputs:
    Returns:
    Called by:
    Description:
    """
    
    ps = np.array([p for p_ in particles[:,z,y,x,:] for p in p_ if p != 0])
    
#     print(ps)
    
    return ps

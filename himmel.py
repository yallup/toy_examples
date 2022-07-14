import numpy as np
from plotting.plots import make_animation_frames,plot_dead
from pypolychord import priors,run_polychord
import pypolychord.settings as settings
import os
import anesthetic

def himmel(x,scale):
    #vectorized himmelblau function with scaling paramter
    x1=x[...,0]
    x2=x[...,1]
    return -((x1**2+x2-11)**2 + (x1+x2**2-7)**2)/(2*scale**2)

#make a grid in mpl
grid = np.mgrid[-5:5:.1, -5:5:.1]
gridstack= np.dstack(grid)
scale=10
#we want the himmelblau function to be the loglikelihood so exp it to get the true target function
z=np.exp(himmel(gridstack,scale))

target_directory="himmelblau"

#wrap up likelihood and prior, plys set polychord settings
prior = priors.UniformPrior(-5,5)
def loglike(theta):
    return float(himmel(theta,scale)), []

settings=settings.PolyChordSettings(nDims=2,nDerived=0)
settings.nlive=100
settings.base_dir=os.path.join(target_directory,"chains")
run_polychord(loglike,prior=prior,nDims=2,nDerived=0,settings=settings)

#read in nested samples
ns=anesthetic.NestedSamples(root=os.path.join(target_directory,"chains/test"))

#make the live points animation
make_animation_frames(grid,z,ns,target_directory)
#plot the dead points 
plot_dead(grid,z,ns,target_directory)
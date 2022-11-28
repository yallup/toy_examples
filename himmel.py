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
# run_polychord(loglike,prior=prior,nDims=2,nDerived=0,settings=settings)

#read in nested samples
# ns=anesthetic.NestedSamples(root=os.path.join(target_directory,"chains/test"))
ns=anesthetic.read_chains(root=os.path.join(target_directory,"chains/test"))
#make the live points animation
make_animation_frames(grid,z,ns,target_directory)
#plot the dead points 
plot_dead(grid,z,ns,target_directory)

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.style.use("prb")
mpl.rcParams['grid.alpha'] = 0.2
mpl.rcParams['grid.color'] = "gray"
X,Y=grid
fig = plt.figure(figsize=[5,4])
ax = plt.axes(projection='3d')
# ax.contour3D(X,Y, z, 10, cmap='magma')
ax.plot_surface(X, Y, z, rstride=1, cstride=1,
                cmap='magma', edgecolor='none')
# plt.axis('off')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
ax.set_zlabel(r'$f(x)$')
ax.grid(alpha=0.2)
# plt.grid(b=None)
ax.view_init(45, 35)
fig.savefig(os.path.join(target_directory,"himmelblau.pdf"))
fig.savefig(os.path.join(target_directory,"himmelblau.png"),dpi=250)
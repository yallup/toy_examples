import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess

# plt.style.use(os.path.dirname(os.path.realpath(__file__))+'/polynet.mplstyle')
plt.style.use("prb")

def make_animation_frames(grid,z,samples,filename,frames_dir="frames"):
    """Make animation frames then animate them with subprocess call to convert

    Args:
        grid (np.array): x,y np.mgrid
        z (np.array): target function
        samples (anesthetic.NestedSamples): instance of NS chain file read with anesthetic
        filename (str): name of animation files
        frames_dir (str, optional): Defaults to "frames".
    """
    anim_dir=os.path.join(filename,frames_dir)
    gs = {"wspace": 0.15, "hspace": 0.03}
    n_frames=20
    # n_samp=500
    os.makedirs(anim_dir,exist_ok=True)

    for i in np.arange(0, n_frames, 1):
        uw = samples.logL_birth[~(samples.logL_birth == -np.inf)]
        points = samples.live_points(uw.min() + i / n_frames * (uw.max() - uw.min()))

        f, a = plt.subplots(nrows=1, ncols=2, sharey=True,sharex=True, gridspec_kw=gs, figsize=[6, 3])
        scatterps=points #points.sample(100,replace=True)
        a[0].contour(*grid, z, levels=10, alpha=1.,cmap="magma")
        a[0].contour(*grid, z, levels=[np.exp(scatterps.logL.min())], colors="black", alpha=.6, linewidths=3)
        
        a[0].set_title("Likelihood",fontsize=6)
        a[1].set_title("Live points",fontsize=6)
        
        a[1].scatter(scatterps.to_numpy()[...,0],scatterps.to_numpy()[...,1],marker="o",color="C0",edgecolor="black",rasterized=True)
        a[1].contour(*grid, z, levels=[np.exp(scatterps.logL.min())], colors="black", alpha=.6, linewidths=3)
        a[0].set_xlim(-5 , 5)
        a[0].set_ylim(-5, 5)

        # f.tight_layout()
        f.savefig(os.path.join(anim_dir, str(i) + ".png"),dpi=300)

    
    proc=subprocess.Popen(["convert", "-delay", "100",*[os.path.join(anim_dir,str(x)+".png") for x in np.arange(n_frames-1)],"-delay","300",os.path.join(anim_dir,str(n_frames-1)+".png"),os.path.join(filename,filename)+".gif"])
    proc.communicate()
    proc=subprocess.Popen(["mogrify", "-layers", "optimize", "-fuzz", "7%", os.path.join(filename,filename)+".gif"])
    proc.communicate()


def plot_dead(grid,z,samples,filename):
    gs = {"wspace": 0.05, "hspace": 0.03}
    f, a = plt.subplots(nrows=1, ncols=2, sharey=True,sharex=True, gridspec_kw=gs, figsize=[6, 3])
    
    samp=samples[[0,1]].to_numpy()
    a[0].contour(*grid, z, levels=10, alpha=1.,cmap="magma")
    a[0].set_title("Target Function")

    a[1].set_title("Nested Sampling dead points")
    a[1].scatter(samp[...,0],samp[...,1],marker="o",color="C1",edgecolor="black",rasterized=True)
    
    a[0].set_xlim(-5,5)
    a[0].set_ylim(-5, 5)
    f.savefig(os.path.join(filename, filename+"_dead.png"),dpi=300)
# ====================================================================================== #
# Code for producing a random Voronoi cell grid across the African continent.
# Author : Eddie Lee, edlee@csh.ac.at
# ====================================================================================== #
import sys
import os
from multiprocess import Pool
from workspace.utils import load_pickle, save_pickle
import numpy as np
import dill as pickle
from warnings import warn
from itertools import product

from voronoi_globe import *


# These bounds have been tuned to cover the African continent. They would have to be
# changed for other areas.
WIDTH = (0.05235987755982988, 1.6406094968746698)
HEIGHT = (-0.7853981633974483, 0.8203047484373349)

LARGEWIDTH = (np.pi*2-.31, 1.92)
LARGEHEIGHT = (-1, 1.1)

def create_one_grid(gridix, dx):
    assert 100>=gridix>=0
    assert dx in [20,40,80,160,320,640,1280,28,57,113,226,453,905]

    if dx==20 or dx==28 or dx==40:
        # generate centers of voronoi cells
        poissd = PoissonDiscSphere(np.pi/dx,
                                   width_bds=LARGEWIDTH,
                                   height_bds=LARGEHEIGHT)
    else:
        # load coarse grid that will be used to group cells in finer grid
        fname = f'voronoi_grids/{dx//2}/{str(gridix).zfill(2)}.p'
        if not os.path.isfile(fname):
            # try to find something close to a factor of 2
            for i in os.listdir('voronoi_grids'):
                if abs((int(i)-dx//2)) <= dx/2*.02:
                    fname = f'voronoi_grids/{i}/{str(gridix).zfill(2)}.p'
        poissd = pickle.load(open(fname,'rb'))['poissd']
        coarsegrid = poissd.samples

        # generate centers of voronoi cells
        poissd = PoissonDiscSphere(np.pi/dx,
                                   width_bds=WIDTH,
                                   height_bds=HEIGHT,
                                   coarse_grid=coarsegrid,
                                   k_coarse=15)
    poissd.sample()

    fname = f'voronoi_grids/{dx}/{str(gridix).zfill(2)}.p'
    if os.path.isfile(fname):
        warn("Overwriting existing file.")
    save_pickle(['poissd'], fname, True)

def main(gridix):
    """Loop thru Voronoi tessellations starting with coarsest grid of solid angle
    pi/20, which then forms the basis for a coarse grid to speed up finer grids in
    nested chain.
    """
    for dx in [20,40,80,160,320,640,1280,28,57,113,226,453,905]:
        try:
            os.makedirs(f'./voronoi_grids/{dx}')
        except OSError:
            pass

        create_one_grid(gridix, dx)
        print(f"Done with grid {gridix} and dx {dx}.\n")

if __name__=='__main__':
    # parallelized iteration over indicated grids
    gridix = [int(i) for i in sys.argv[1:]]
    
    # first, use Poisson disc sampling to generate the centers of the Voronoi tiles
    print("Generating cell centers...")
    with Pool() as pool:
        pool.map(main, gridix)
    print("Done.")
    
    print("Drawing boundaries around Voronoi cells...")
    polygonize(product([int(i) for i in os.listdir('./voronoi_grids')], gridix))
    print("Done.")


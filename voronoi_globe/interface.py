# ====================================================================================== #
# Interface functions.
# Author: Eddie Lee, edlee@csh.ac.at
# ====================================================================================== #
from .utils import *



def load_voronoi(dx, gridix=0, prefix='.'):
    """Load GeoPandas DataFrame and apply proper index before returning.

    Parameters
    ----------
    dx : int
    gridix : int, 0
    prefix : str, ''
        Directory prefix.

    Returns
    -------
    gpd.GeoDataFrame
    """

    gdf = gpd.read_file(f'{prefix}/voronoi_grids/{dx}/borders{str(gridix).zfill(2)}.shp')
    with open(f'{prefix}/voronoi_grids/{dx}/borders_ix{str(gridix).zfill(2)}.p', 'rb') as f:
        ix = pickle.load(f)['selectix']
    gdf.set_index(ix, inplace=True)
    
    # turn neighbor strings into lists
    gdf['neighbors'] = gdf['neighbors'].apply(lambda x: [int(i) for i in x.split(', ')])

    return gdf

def load_centers(dx, gridix=0, prefix='.'):
    """Load voronoi centers as ndarray, but put them into same reference frame as
    polygons.

    Parameters
    ----------
    dx : int
    gridix : int, 0
    prefix : str, '.'

    Returns
    -------
    np.ndarray
    """
    
    with open(f'{prefix}/voronoi_grids/{dx}/{str(gridix).zfill(2)}.p', 'rb') as f:
        poissd = pickle.load(f)['poissd']

    # must unwrap centers to test for presence in cell
    centers = gpd.GeoSeries([Point(unwrap_lon(xy[0]), xy[1]) for xy in transform(poissd.samples)])
    return centers

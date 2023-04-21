# ====================================================================================== #
# Interface functions.
# Author: Eddie Lee, edlee@csh.ac.at
# ====================================================================================== #
from fiona.errors import DriverError

from .utils import *



def africa_gdf():
    """Load polygons for main African continent landmass and Madagascar.
    """
	world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
	world = world[['continent', 'geometry']]
	continents = world.dissolve(by='continent')
	return gpd.GeoDataFrame(continents["geometry"]["Africa"] , geometry=0)

def load_voronoi(dx, gridix=0, prefix='.', exclude_boundary=False, exclude_center=False):
    """Load GeoPandas DataFrame and apply proper index before returning.

    Parameters
    ----------
    dx : int
    gridix : int, 0
    prefix : str, ''
        Directory prefix.
    exclude_boundary : bool, False
    exclude_center : bool, False

    Returns
    -------
    gpd.GeoDataFrame
    """
    assert not (exclude_center and exclude_boundary)

    gdf = gpd.read_file(f'{prefix}/voronoi_grids/{dx}/borders{str(gridix).zfill(2)}.shp')
    with open(f'{prefix}/voronoi_grids/{dx}/borders_ix{str(gridix).zfill(2)}.p', 'rb') as f:
        ix = pickle.load(f)['selectix']
    gdf.set_index(ix, inplace=True)
    
    # turn neighbor strings into lists
    gdf['neighbors'] = gdf['neighbors'].apply(lambda x: [int(i) for i in x.split(', ')])
    
    try:
        if exclude_boundary:
            af = africa_gdf() 
            contained_ix = [(af.geometry.iloc[0].contains(p.buffer(1)) or
                             af.geometry.iloc[1].contains(p.buffer(1)))
                            for p in gdf['geometry'].values]
            gdf = gdf.loc[contained_ix]
        elif exclude_center:
            af = africa_gdf() 
            contained_ix = [not (af['geometry'].iloc[0].contains(p.buffer(1)) or
                                 af.geometry.iloc[1].contains(p.buffer(1)))
                            for p in gdf['geometry'].values]
            gdf = gdf.loc[contained_ix]
    except DriverError:
        warn("Africa shapefile cannot be found. Boundary filtering not done.")
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

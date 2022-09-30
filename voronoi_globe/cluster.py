# ====================================================================================== #
# Module for clustering routines used to generate avalanches.
# Author: Eddie Lee, edlee@csh.ac.at
# ====================================================================================== #
from itertools import product
from functools import partial

from .classes import SphereCoordinate, GreatCircle, GreatCircleIntersect, VoronoiCell, Quaternion
from .utils import *



def polygonize(iter_pairs=None):
    """Create polygons denoting boundaries of Voronoi grid.

    Parameters
    ----------
    iter_pairs : list of twoples, None
        Can be specified to direct polygonization for particular combinations of dx
        and grids {dx as int}, {gridix as int}. When None, goes through preset list
        of all combos from dx=80 up to dx=1280 (about 35km).
    """

    from numpy import pi

    def loop_wrapper(args):
        dx, gridix = args
        poissd = pickle.load(open(f'voronoi_grids/{dx}/{str(gridix).zfill(2)}.p', 'rb'))['poissd']

        # identify polygons that are within interesting boundaries
        lonlat = poissd.samples.copy()
        for i in range(len(lonlat)):
            lonlat[i] = unwrap_lon((lonlat[i,0]/pi*180 + 330)%360), lonlat[i,1]/pi*180
        if dx<=20:
            selectix = np.where((lonlat[:,0]>-37.2) & (lonlat[:,0]<70.5) &
                                (lonlat[:,1]>-54) & (lonlat[:,1]<57))[0]
        elif dx<=40:
            selectix = np.where((lonlat[:,0]>-22.2) & (lonlat[:,0]<55.5) &
                                (lonlat[:,1]>-39) & (lonlat[:,1]<42))[0]
        else:
            selectix = np.where((lonlat[:,0]>-19.7) & (lonlat[:,0]<53.5) &
                                (lonlat[:,1]>-37) & (lonlat[:,1]<40))[0]
        
        # create bounding polygons, the "Voronoi cells"
        polygons = []
        for i in selectix:
            try:
                polygons.append(create_polygon(poissd, i))
            except Exception:
                raise Exception(f"Problem with {i}.")
        polygons = gpd.GeoDataFrame({'index':list(range(len(polygons)))},
                                    geometry=polygons,
                                    crs='EPSG:4326',
                                    index=selectix)
        
        # identify all neighbors of each polygon
        neighbors = []
        sindex = polygons.sindex
        scaled_polygons = polygons['geometry'].scale(1.01,1.01)
        for i, p in polygons.iterrows():
            # scale polygons by a small factor to account for precision error in determining
            # neighboring polygons; especially important once dx becomes large, say 320
            # first lines look right but seem to involve some bug in detecting intersections
            #pseries = gpd.GeoSeries(p.geometry, crs=polygons.crs).scale(1.001, 1.001)
            #neighborix = sindex.query_bulk(pseries)[1].tolist()
            neighborix = polygons.index[polygons.intersects(scaled_polygons.loc[i])].tolist()

            # remove self
            try:
                neighborix.pop(neighborix.index(i))
            except ValueError:
                pass
            assert len(neighborix)

            # must save list as string for compatibility with Fiona pickling
            neighbors.append(str(sorted(neighborix))[1:-1])
        polygons['neighbors'] = neighbors

        # correct errors
        polygons, n_inconsis = check_voronoi_tiles(polygons)
        check_overlap(polygons)

        # save
        polygons.to_file(f'voronoi_grids/{dx}/borders{str(gridix).zfill(2)}.shp')
        with open(f'voronoi_grids/{dx}/borders_ix{str(gridix).zfill(2)}.p', 'wb') as f:
            pickle.dump({'selectix':selectix}, f)
 
    if iter_pairs is None:
        # iterate over all preset combinations of dx and dt
        iter_pairs = product([40, 80, 160, 320, 640, 1280], range(10))

    with mp.Pool() as pool:
        pool.map(loop_wrapper, iter_pairs)

def revise_neighbors(dx, gridix, write=True):
    """Re-calculate neighbors and save to shapefile. Used for a one-time fix, but
    saved here for later reference.

    Parameters
    ----------
    dx : int
        Inverse solid angle.
    gridix : int
        Grid index.
    write : bool, True
        If True, write to shapely file.
    """

    assert os.path.isfile(f'voronoi_grids/{dx}/borders{str(gridix).zfill(2)}.shp')
    polygons = gpd.read_file(f'voronoi_grids/{dx}/borders{str(gridix).zfill(2)}.shp')
    
    # identify all neighbors of each polygon
    neighbors = []
    sindex = polygons.sindex
    for i, p in polygons.iterrows():
        # scale polygons by a small factor to account for precision error in determining
        # neighboring polygons; especially important once dx becomes large, say 320
        pseries = gpd.GeoSeries(p.geometry, crs=polygons.crs).scale(1.001, 1.001)
        neighborix = sindex.query_bulk(pseries)[1].tolist()

        # remove self
        try:
            neighborix.pop(neighborix.index(i))
        except ValueError:
            pass
        assert len(neighborix)

        # must save list as string for compatibility with Fiona pickling
        neighbors.append(str(sorted(neighborix))[1:-1])
    polygons['neighbors'] = neighbors

    polygons, n_inconsis = check_voronoi_tiles(polygons)
    check_overlap(polygons)
    if n_inconsis:
        print(f"There are {n_inconsis} asymmetric pairs that were corrected.")
    
    if write:
        polygons.to_file(f'voronoi_grids/{dx}/borders{str(gridix).zfill(2)}.shp')
    return polygons

def create_polygon(poissd, centerix):
    """Construct polygon about specified point in PoissonDiscSphere.

    Parameters
    ----------
    poissd : PoissonDiscSphere
    centerix : int
        Construct polygon about specified point in PoissonDiscSphere.

    Returns
    -------
    shapely.geometry.Polygon
    """
    
    center = poissd.samples[centerix]

    neighborsix = poissd.neighbors(center)
    neighborsix.pop(neighborsix.index(centerix))
    assert len(neighborsix)>=3, "Provided point has less than three neighbors."

    center = SphereCoordinate(center[0], center[1]+pi/2)
    neighbors = [SphereCoordinate(s[0], s[1]+pi/2) for s in poissd.samples[neighborsix]]
    
    try:
        precision = 1e-7
        cell = VoronoiCell(center, rng=np.random.RandomState(0), precision=precision)
        triIx = cell.initialize_with_tri(neighbors)
    except AssertionError:
        # try reducing precision
        precision = 5e-8
        cell = VoronoiCell(center, rng=np.random.RandomState(0), precision=precision)
        triIx = cell.initialize_with_tri(neighbors)

    for i, ix in enumerate(triIx):
        neighbors.pop(ix-i)
    
    # iterate thru all neighbors and try to add them to convex hull, most won't be added
    for n in neighbors:
        cell.add_cut(GreatCircle.bisector(n, center))
    
    poly = Polygon([(unwrap_lon((v.phi/pi*180+330)%360), (v.theta-pi/2)/pi*180) for v in cell.vertices])
    return poly


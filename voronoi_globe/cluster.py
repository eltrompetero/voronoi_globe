# ====================================================================================== #
# Module for clustering routines used to generate avalanches.
# Author: Eddie Lee, edlee@csh.ac.at
# ====================================================================================== #
from itertools import product
from functools import partial
from scipy.spatial import SphericalVoronoi

from .classes import SphereCoordinate, GreatCircle, GreatCircleIntersect, VoronoiCell, Quaternion
from .utils import *


def polygonize(iter_pairs=None, region=None ,iprint=False):
    """Create polygons denoting boundaries of Voronoi grid.

    Parameters
    ----------
    iter_pairs : list of twoples, None
        Can be specified to direct polygonization for particular combinations of dx
        and grids {dx as int}, {gridix as int}. When None, goes through preset list
        of all combos from dx=80 up to dx=1280 (about 35km).
    iprint : bool, False
    """
    from numpy import pi

    def loop_wrapper(args):
        dx, gridix = args
        poissd = pickle.load(open(f'voronoi_grids/{dx}/{str(gridix).zfill(2)}.p', 'rb'))['poissd']
        
        # set up SphericalVoronoi
        xy = poissd.samples.copy()
        xy[:,1] += pi/2

        centers = []
        for xy_ in xy:
            centers.append(SphereCoordinate(*xy_).vec)
        centers = np.vstack(centers)

        sv = SphericalVoronoi(centers)
        # sort vertices (optional, helpful for plotting)
        sv.sort_vertices_of_regions()

        # identify polygons that are within interesting boundaries
        lonlat = poissd.samples.copy()
        for i in range(len(lonlat)):
            lonlat[i] = unwrap_lon((lonlat[i,0]/pi*180 + 330)%360), lonlat[i,1]/pi*180

        #### ADD NEW BOUNDS FOR NEW REGIONS HERE #####

        if region=="africa":
            if dx<=28:
                lat_bounds = (-30,62)
                lng_bounds = (-45,50)
            elif dx<=57:
                lat_bounds = (-22.2,55.5)
                lng_bounds = (-39,42)
            else:
                lat_bounds = (-19.7,53.5)
                lng_bounds = (-37,40)
        elif region=="mexico":
            if dx<=28:
                lat_bounds = (-125,-75)
                lng_bounds = (5,40)
            elif dx<=57:
                lat_bounds = (-125,-75)
                lng_bounds = (5,40)
            else:
                lat_bounds = (-125,-75)
                lng_bounds = (5,40)
        elif region=="latin_america":
            if dx<=28:
                lat_bounds = (-125,-25)
                lng_bounds = (-70,40)
            elif dx<=57:
                lat_bounds = (-125,-25)
                lng_bounds = (-70,40)
            else:
                lat_bounds = (-125,-25)
                lng_bounds = (-70,40)
        elif region=="india":
            if dx<=28:
                lat_bounds = (60,105)
                lng_bounds = (0,45)
            elif dx<=57:
                lat_bounds = (60,105)
                lng_bounds = (0,45)
            else:
                lat_bounds = (60,105)
                lng_bounds = (0,45)
        ####

        selectix = np.where((lonlat[:,0]>lat_bounds[0]) & (lonlat[:,0]<lat_bounds[1]) &
                            (lonlat[:,1]>lng_bounds[0]) & (lonlat[:,1]<lng_bounds[1]))[0]
        # selectix = np.array([i for i in range(len(lonlat))])    ## instead of setting bounds for centers around which boundaries will be made, all the centers will get boundary
        
        # create bounding polygons, the "Voronoi cells"
        polygons = []
        for r in [sv.regions[i] for i in selectix]:
            coords = [SphereCoordinate(*xyz) for xyz in sv.vertices[r]]
            polygons.append(Polygon([(unwrap_lon((v.phi/pi*180+330)%360), (v.theta-pi/2)/pi*180) for v in coords]))

        polygons = gpd.GeoDataFrame({'index':list(range(selectix.size))},
                                    geometry=polygons,
                                    crs='EPSG:4326',
                                    index=selectix)
        if iprint: print("Done making polygons.")

        # identify all neighbors of each polygon
        neighbors = []
        sindex = polygons.sindex
        scaled_polygons = polygons['geometry'].scale(1.001,1.001)
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
            #assert len(neighborix)
            if len(neighborix)==0: print(i)

            # must save list as string for compatibility with Fiona pickling
            neighbors.append(str(sorted(neighborix))[1:-1])
        polygons['neighbors'] = neighbors
        if iprint: print("Done finding neighbors.")

        # correct errors
        polygons, n_inconsis = check_voronoi_tiles(polygons)
        if iprint: print("Done checking neighbors.")
        check_overlap(polygons, region=region)
        if iprint: print("Done checking overlap.")

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
        # try changing precision
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


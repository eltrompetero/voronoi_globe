# ====================================================================================== #
# Utilities for handling and testing Voronoi tiling.
# Author: Eddie Lee, edlee@csh.ac.at
# ====================================================================================== #
from itertools import product
from numpy import pi
from shapely.geometry import Point, Polygon
from numba import njit
import dill as pickle
import numpy as np
import geopandas as gpd
import multiprocess as mp
from numpy import cos, sin, arctan2, arccos, arcsin, pi, arctan
from numba import float64, njit, jit
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform



def transform(phitheta):
    """From angles to lon, lat coordinates accounting for the longitudinal shift necessary
    to get to Africa.
    
    Parameters
    ----------
    phitheta : ndarray
    
    Returns
    -------
    ndarray
        Lonlat.
    """
    
    if phitheta.ndim==1:
        newcoord = phitheta / pi * 180
        newcoord[0] += 330
        newcoord[0] %= 360
        return newcoord
        
    newcoord = phitheta / pi * 180
    newcoord[:,0] += 330
    newcoord[:,0] %= 360
    return newcoord

def unwrap_lon(x):
    """Transform longitude from (0,360) to (-180,180).
    
    Parameters
    ----------
    x : ndarray or float
    
    Returns
    -------
    ndarray or float
    """
    
    if isinstance(x, np.ndarray):
        x = x.copy()
        ix = x>180
        x[ix] -= 180
        x[ix] *= -1
        return x
    
    if x>180:
        return -(360-x)
    return x

def check_voronoi_tiles(gdf, iprint=False, parallel=True):
    """Check Voronoi tiles to make sure that they are consistent.

    This will take any asymmetric pair of tiles (where one considers the other to be
    a neighbor but not vice versa) and make sure that both neighbors lists are
    consistent with one another.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
    iprint : bool, False
    parallel : bool, False

    Returns
    -------
    geopandas.GeoDataFrame
    int
    """
    
    from shapely import wkt
    from shapely.errors import TopologicalError
    
    assert (np.diff(gdf['index'])==1).all()

    assert gdf.geometry.is_valid.all()
    if iprint: print("All geometries valid.")
    
    n_inconsis = 0
    for i, row in gdf.iterrows():
        for n in row['neighbors'].split(', '):
            n = int(n)
            if not str(i) in gdf.loc[n]['neighbors'].split(', '):
                new_neighbors = sorted(gdf.loc[n]['neighbors'].split(', ') + [str(i)])
                gdf.loc[n,'neighbors'] = ', '.join(new_neighbors)
                n_inconsis += 1
    if iprint: print("Done with correcting asymmetric neighbors.")

    # check overlap with all of africa
    # load africa
    africa = gpd.read_file(f'./africa_countries/afr_g2014_2013_0.shp')
    assert africa.crs.name=='WGS 84'
    
    # drop island countries
    countries_to_drop = ['Cape Verde', 'Mauritius', 'Seychelles']
    keepix = np.ones(len(africa), dtype=bool)
    for c in countries_to_drop:
        keepix[africa['ADM0_NAME']==c] = False
    africa = africa.loc[keepix]

    # for each country check that intersection w/ voronoi area is very
    # close to total country, but these will not always be the same
    # b/c of precision error creating gaps or overlap btwn voronoi cells
    def loop_wrapper(args):
        i, country = args
        assert np.isclose(voronoi_cov.intersection(country.geometry).area, country.geometry.area,
                          rtol=1e-3), (i, country)
    
    if parallel:
        with Pool() as pool:
            # union voronoi cells
            voronoi_cov = gdf.iloc[0].geometry
            for i in range(1, len(gdf)):
                voronoi_cov = voronoi_cov.union(gdf.iloc[i].geometry)
            voronoi_cov = gpd.GeoSeries(voronoi_cov)
            try:
                pool.map(loop_wrapper, africa.iterrows())
            except TopologicalError:
                gdf['geometry'] = gdf['geometry'].apply(lambda x:wkt.loads(wkt.dumps(x,
                                                                  rounding_precision=8)))
                # union voronoi cells
                voronoi_cov = gdf.iloc[0].geometry
                for i in range(1, len(gdf)):
                    voronoi_cov = voronoi_cov.union(gdf.iloc[i].geometry)
                voronoi_cov = gpd.GeoSeries(voronoi_cov)
                pool.map(loop_wrapper, africa.iterrows())
    else:
        try:
            # union voronoi cells
            voronoi_cov = gdf.iloc[0].geometry
            for i in range(1, len(gdf)):
                voronoi_cov = voronoi_cov.union(gdf.iloc[i].geometry)
            voronoi_cov = gpd.GeoSeries(voronoi_cov)
            for args in africa.iterrows():
                loop_wrapper(args)
        except TopologicalError:
            gdf['geometry'] = gdf['geometry'].apply(lambda x:wkt.loads(wkt.dumps(x,
                                                                rounding_precision=8)))
            # union voronoi cells
            voronoi_cov = gdf.iloc[0].geometry
            for i in range(1, len(gdf)):
                voronoi_cov = voronoi_cov.union(gdf.iloc[i].geometry)
            voronoi_cov = gpd.GeoSeries(voronoi_cov)
            for args in africa.iterrows():
                loop_wrapper(args)
 
    if iprint: print("Done with checking overlap with Africa.")

    return gdf, n_inconsis

def check_poisson_disc(poissd, min_dx):
    """Check PoissonDiscSphere grid.

    Parameters
    ----------
    poissd : PoissonDiscSphere
    min_dx : float
    """
    
    # min distance surpasses min radius
    for xy in poissd.samples:
        neighbors, dist = poissd.neighbors(xy, return_dist=True)
        zeroix = dist==0
        assert zeroix.sum()==1
        assert dist[~zeroix].min()>=min_dx, (min_dx, dist[~zeroix].min())

def extend_poissd_coarse_grid(dx):
    """Redefine PoissonDiscSphere to consider an extended number of coarse grid
    neighbors to avoid boundary artifacts (that though uncommon) would manifest from
    not considering neighbors because of thin bounds on polygons.

    This increases the default number of coarse neighbors 9 used in PoissonDiscSphere
    to 15.

    Parameters
    ----------
    dx : int
    """
    
    for i in range(10):
        with open(f'voronoi_grids/{dx}/{str(i).zfill(2)}.p','rb') as f:
            poissd = pickle.load(f)['poissd']
            poissd = _extend_poissd_coarse_grid(poissd)
            pickle.dump({'poissd':poissd}, open(f'voronoi_grids/{dx}/{str(i).zfill(2)}.p.new','wb'))

def _extend_poissd_coarse_grid(poissd):
    """Main part of .extend_poissd_coarse_grid()

    Parameters
    ----------
    poissd : PoissonDiscSphere

    Returns
    -------
    PoissonDiscSphere
    """

    newpoissd = PoissonDiscSphere(poissd.r,
                                  width_bds=poissd.width,
                                  height_bds=poissd.height,
                                  coarse_grid=poissd.coarseGrid,
                                  k_coarse=15)
    newpoissd.samples = poissd.samples

    for i, s in enumerate(poissd.samples):
        newpoissd.samplesByGrid[poissd.assign_grid_point(s)].append(i)

    return newpoissd

def mod_angle(angle, radians=True):
    """
    Modulus into (-pi,pi) or (-180,180).
    
    Parameters
    ----------
    angle : ndarray
    """
    
    if radians:
        return np.mod(angle+np.pi,2*np.pi)-np.pi
    return np.mod(angle+180,2*180)-180

def ortho_plane(v):
    """Return a plane defined by two vectors orthogonal to the given vector using random
    vector and Gram-Schmidt.
    
    Parameters
    ----------
    v : ndarray
    
    Returns
    -------
    ndarray
    ndarray
    """
    
    assert v.size==3
    
    # Get a first orthogonal vector
    r1 = np.random.rand(3)
    r1 -= v*r1.dot(v)
    r1 /= np.sqrt(r1.dot(r1))
    
    # Get second othorgonal vector
    r2 = np.cross(v,r1)
    
    return r1, r2

def convex_hull(xy, recursive=False, concatenate_first=False):
    """Identify convex hull of points in 2 dimensions. I think this is the same as
    Quickhull.
    
    Recursive version. Number of points to consider typically goes like sqrt(n), so
    this can handle a good number, but this could be made faster and to handle larger
    systems by making it sequential.

    This has been tested visually on a number of random examples for the armed_conflict
    project.
    
    Parameters
    ----------
    xy : ndarray
        List of coordinates.
    concatenate_first : bool, False
        If True, will append first coordinate again at end of returned list for a closed
        path.
        
    Returns
    -------
    list
        Indices of rows in xy that correspond to the convex hull. It is traversed in a
        clockwise direction.
    
    Example
    -------
    >>> xy = np.random.normal(size=(100,2))
    >>> hull = convex_hull(xy)
    >>> fig, ax = plt.subplots()
    >>> for i in range(10):
    >>>     ax.text(xy[i,0], xy[i,1], i)
    >>> ax.plot(*xy.T,'o')
    >>> ax.plot(*xy[4],'o')
    >>> ax.plot(*xy[9],'o')
    >>> ax.plot(xy[hull][:,0], xy[hull][:,1], 'k-')
    """
    
    if len(xy)<=3:
        return np.arange(len(xy), dtype=int)
    assert xy.shape[1]==2, "This only works for 2D."
    assert len(np.unique(xy,axis=0))==len(xy), "No duplicate entries allowed."
    
    # going around clockwise, get the extrema along each axis
    endptsix = [xy[:,0].argmin(), xy[:,1].argmax(),
                xy[:,0].argmax(), xy[:,1].argmin()]
    # remove duplicates
    if endptsix[0]==endptsix[1]:
        endptsix.pop(0)
    elif endptsix[1]==endptsix[2]:
        endptsix.pop(1)
    elif endptsix[2]==endptsix[3]:
        endptsix.pop(2)
    elif endptsix[3]==endptsix[0]:
        endptsix.pop(0)
    
    if recursive:
        pairsToConsider = [(endptsix[i], endptsix[(i+1)%len(endptsix)])
                           for i in range(len(endptsix))]
        
        # for each pair, assembly a list of points to check by using a cutting region determined
        # by the line passing through that pair of points
        pointsToCheck = []
        for i,j in pairsToConsider:
            ix = np.delete(range(len(xy)), endptsix)
            pointsToCheck.append( ix[_boundaries_diag_cut_out(xy[ix], xy[i], xy[j])] )
        
        # whittle 
        hull = []
        for ix, checkxy in zip(pairsToConsider, pointsToCheck):
            subhull = []
            _check_between_pair(xy, ix[0], ix[1], checkxy, subhull)
            hull.append(subhull)

        # extract loop
        hull = np.concatenate(hull).ravel()
        # monkey patch because some elements appear twice
        hull = np.append(hull[::2], hull[-1])
        _, ix = np.unique(hull, return_index=True)
        hull = hull[ix[np.argsort(ix)]]
        if concatenate_first:
            hull = np.concatenate((hull, [hull[0]]))
        return hull
    
    # sequential Graham algorithm
    # center all the points about a centroid defined as between min/max pairs of x and y axes
    midxy = (xy.max(0)+xy.min(0))/2
    xy = xy-midxy
    sortix = _sort_by_phi(xy)[::-1]
    xysorted = xy[sortix]
    # start with point with leftmost point
    startix = xysorted[:,0].argmin()
    sortix = np.roll(sortix, -startix)
    xysorted = np.roll(xysorted, -startix, axis=0)
    hull = _check_between_triplet(xysorted)

    return sortix[hull]

@njit(cache=True)
def ind_to_sub(n, ix):
    """Convert index from flattened upper triangular matrix to pair subindex.

    Parameters
    ----------
    n : int
        Dimension size of square array.
    ix : int
        Index to convert.

    Returns
    -------
    subix : tuple
        (i,j)
    """

    k = 0
    for i in range(n-1):
        for j in range(i+1,n):
            if k==ix:
                return (i,j)
            k += 1

def plot_unit_sphere(ax, radius=.98):
    """Plot transparent unit sphere.
    
    Parameters
    ----------
    ax : mpl.Axes
    radius : float, .98

    Returns
    -------
    None
    """

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v)) * radius
    y = np.outer(np.sin(u), np.sin(v)) * radius
    z = np.outer(np.ones(np.size(u)), np.cos(v)) * radius

    # Plot the surface
    ax.plot_surface(x, y, z, alpha=.2, color='k')

def rand(n=1, degree=True):
    """Randomly sample points from the surface of a sphere.

    Parameters
    ----------
    n : int, 1
    degree : bool, True

    Returns
    -------
    randlon : float
    randlat : float
    """

    if degree:
        randlat = arccos(2*np.random.rand(n)-1)/pi*180-90
        randlon = np.random.uniform(-180,180,size=n)
        return randlon, randlat
    randlat = arccos(2*np.random.rand(n)-1)-pi/2
    randlon = np.random.uniform(-pi,pi,size=n)
    return randlon, randlat

def haversine(x, y, r=1):
    """
    Parameters
    ----------
    x,y : tuple
        (phi, theta) azimuthal angle first
    radius : float, 1

    Returns
    -------
    dist : float
    """
    
    dphi = y[0] - x[0]
    # assuming that convention of theta in [0,pi] holds, this shift of coordinate system
    # will vastly enhance accuracy of calculation
    th1 = x[1] - pi/2
    th2 = y[1] - pi/2

    num = np.sqrt((cos(th2) * sin(dphi))**2 + (cos(th1)*sin(th2) -
                  sin(th1) * cos(th2) * cos(dphi))**2)
    den = sin(th1) * sin(th2) + cos(th1) * cos(th2) * cos(dphi)
    return r * arctan2(num, den)

@njit
def jithaversine(x, y):
    """
    Parameters
    ----------
    x,y : tuple
        (phi, theta)

    Returns
    -------
    dist : float
    """

    dphi = y[0] - x[0]
    # assuming that convention of theta in [0,pi] holds, this shift of coordinate system
    # will vastly enhance accuracy of calculation
    th1 = x[1] - pi/2
    th2 = y[1] - pi/2

    num = np.sqrt((cos(th2) * sin(dphi))**2 + (cos(th1)*sin(th2) -
                  sin(th1) * cos(th2) * cos(dphi))**2)
    den = sin(th1) * sin(th2) + cos(th1) * cos(th2) * cos(dphi)
    return arctan2(num, den)

def latlon2angle(*args):
    """
    Parameters
    ----------
    lonlat as one or lon,lat
    """

    if len(args)==2:
        lat, lon = args
        return lat/180*pi, lon/180*pi
    return args[0]/180*pi

def vincenty(point1, point2, a, f, MAX_ITERATIONS=200, CONVERGENCE_THRESHOLD=1e-12):
    """Vincenty's formula (inverse method) to calculate the distance between two points on
    the surface of a spheroid

    Parameters
    ----------
    point1 : twople
        (xy-angle, polar angle). These should be given in radians.
    point2 : twople
        (xy-angle, polar angle)
    a : float
        Equatorial radius.
    f : float
        eccentricity, semi-minor polar axis b=(1-f)*a
    """

    # short-circuit coincident points
    if point1[0] == point2[0] and point1[1] == point2[1]:
        return 0.0
    b=(1-f)*a

    U1 = math.atan((1 - f) * math.tan(point1[0]))
    U2 = math.atan((1 - f) * math.tan(point2[0]))
    L = point2[1] - point1[1]
    Lambda = L

    sinU1 = math.sin(U1)
    cosU1 = math.cos(U1)
    sinU2 = math.sin(U2)
    cosU2 = math.cos(U2)

    for iteration in range(MAX_ITERATIONS):
        sinLambda = math.sin(Lambda)
        cosLambda = math.cos(Lambda)
        sinSigma = math.sqrt((cosU2 * sinLambda) ** 2 +
                             (cosU1 * sinU2 - sinU1 * cosU2 * cosLambda) ** 2)
        if sinSigma == 0:
            return 0.0  # coincident points
        cosSigma = sinU1 * sinU2 + cosU1 * cosU2 * cosLambda
        sigma = math.atan2(sinSigma, cosSigma)
        sinAlpha = cosU1 * cosU2 * sinLambda / sinSigma
        cosSqAlpha = 1 - sinAlpha ** 2
        try:
            cos2SigmaM = cosSigma - 2 * sinU1 * sinU2 / cosSqAlpha
        except ZeroDivisionError:
            cos2SigmaM = 0
        C = f / 16 * cosSqAlpha * (4 + f * (4 - 3 * cosSqAlpha))
        LambdaPrev = Lambda
        Lambda = L + (1 - C) * f * sinAlpha * (sigma + C * sinSigma *
                                               (cos2SigmaM + C * cosSigma *
                                                (-1 + 2 * cos2SigmaM ** 2)))
        if abs(Lambda - LambdaPrev) < CONVERGENCE_THRESHOLD:
            break  # successful convergence
    else:
        return None  # failure to converge

    uSq = cosSqAlpha * (a ** 2 - b ** 2) / (b ** 2)
    A = 1 + uSq / 16384 * (4096 + uSq * (-768 + uSq * (320 - 175 * uSq)))
    B = uSq / 1024 * (256 + uSq * (-128 + uSq * (74 - 47 * uSq)))
    deltaSigma = B * sinSigma * (cos2SigmaM + B / 4 * (cosSigma *
                 (-1 + 2 * cos2SigmaM ** 2) - B / 6 * cos2SigmaM *
                 (-3 + 4 * sinSigma ** 2) * (-3 + 4 * cos2SigmaM ** 2)))
    s = b * A * (sigma - deltaSigma)

    return round(s, 6)

def max_geodist_pair(phitheta, force_slow=False, return_dist=False):
    """Find approximately most distant pair of points on surface of the sphere. First,
    collapse points onto 2D plane that is orthogonal to vector to center of mass of
    points. Then, assume that the convex hull contains the most distant pair of points.
    This is different from max_dist_pair2D() because we use the haversine distance for
    this last step.

    Parameters
    ----------
    xy : ndarray
        (x,y) coordinations
    force_slow : bool, False
        Use slow calculation computing entire matrix of pairwise distances.
    return_dist : bool, False

    Returns
    -------
    tuple
        Indices of two max separated points.
    """
    
    from .utils import convex_hull, ind_to_sub, ortho_plane

    if type(phitheta) is list:
        phitheta = np.vstack(phitheta)
    
    # it is faster to do every pairwise computation when the size of the is small
    if force_slow or len(phitheta)<500:
        return _max_dist_pair(phitheta, return_dist)

    xyz = np.zeros((len(phitheta), 3))
    xyz[:,0] = np.sin(phitheta[:,1]) * np.cos(phitheta[:,0])
    xyz[:,1] = np.sin(phitheta[:,1]) * np.sin(phitheta[:,0])
    xyz[:,2] = np.cos(phitheta[:,1])

    # collapse points down to plane orthogonal to center of mass
    mxyz = xyz.mean(0)
    mxyz /= np.linalg.norm(mxyz)
    v1, v2 = ortho_plane(mxyz)

    xy = np.vstack((xyz.dot(v1), xyz.dot(v2))).T
    
    hull = convex_hull(xy, recursive=True)
    dist = pdist(phitheta[hull], jithaversine)
    mxix = ind_to_sub(hull.size, dist.argmax())
    if return_dist:
        return (hull[mxix[0]], hull[mxix[1]]), dist.max()
    return hull[mxix[0]], hull[mxix[1]]
          
def _max_dist_pair(phitheta, return_dist):
    """Slow way of finding maximally distant pair by checking every pair.
    """
    
    assert len(phitheta)>1
    dmat = pdist(phitheta, jithaversine)
    dmaxix = dmat.argmax()
    majix = ind_to_sub(len(phitheta), dmaxix)
    if return_dist:
        return majix, dmat[dmaxix]
    return majix

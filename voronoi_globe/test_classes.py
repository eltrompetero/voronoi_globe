# ====================================================================================== #
# Testing suite for globe.py
# Author: Eddie Lee, edlee@csh.ac.at
# ====================================================================================== #
from .classes import *
from numpy import pi
import numpy as np



def test_VoronoiCell():
    np.random.seed(0)
    
    for i in range(5):
        # generate random points
        pts = [SphereCoordinate(*np.random.normal(size=3)) for i in range(50)]
        # by placing center at center of mass, we can avoid absurdly large cells that
        # occur in open regions and boundaries of clusters
        center = SphereCoordinate(np.vstack([pt.vec for pt in pts]).mean(0))

        cell = VoronoiCell(center)
        assert np.isclose(np.cross(cell.x, cell.y), center.vec).all()

        newVertexIx = cell.initialize_with_tri(pts)

def test_VoronoiCell_refined():
    """A more precise and difficult test of Voronoi cell creation.
    """

    center = np.array([ 0.48759464, -0.34545269])
    neighbors = np.array([[ 0.49056691, -0.34870281],
                          [ 0.48752733, -0.34848874],
                          [ 0.4841127 , -0.34634591],
                          [ 0.48311632, -0.34977197],
                          [ 0.48576446, -0.35489646],
                          [ 0.4906576 , -0.35120316],
                          [ 0.48581855, -0.35224661],
                          [ 0.48869649, -0.3531848 ],
                          [ 0.49037744, -0.34517818],
                          [ 0.49152938, -0.34250627],
                          [ 0.48887247, -0.34138358],
                          [ 0.48459742, -0.34239052],
                          [ 0.48609738, -0.3401912 ],
                          [ 0.4804698 , -0.34860666],
                          [ 0.48084128, -0.34351105],
                          [ 0.47943089, -0.34615071],
                          [ 0.47772318, -0.34902833],
                          [ 0.49789873, -0.34555398],
                          [ 0.49394745, -0.34402275],
                          [ 0.49431092, -0.34805863],
                          [ 0.48243824, -0.35560724],
                          [ 0.4804153 , -0.35758931],
                          [ 0.48308334, -0.35274429],
                          [ 0.49783819, -0.35349077],
                          [ 0.49378988, -0.35125133],
                          [ 0.49164769, -0.35551661],
                          [ 0.49647939, -0.35082186],
                          [ 0.49431035, -0.3554826 ],
                          [ 0.47902027, -0.35256556],
                          [ 0.47441156, -0.35089235],
                          [ 0.47362387, -0.35395346],
                          [ 0.47642148, -0.35302967],
                          [ 0.48856687, -0.33331962],
                          [ 0.48839527, -0.33636544],
                          [ 0.48609393, -0.33752007],
                          [ 0.47816204, -0.33835023],
                          [ 0.48256183, -0.34018605],
                          [ 0.48253348, -0.33706058],
                          [ 0.47837019, -0.34170196],
                          [ 0.47577654, -0.33953754],
                          [ 0.49497841, -0.34002303],
                          [ 0.49292039, -0.33580577],
                          [ 0.49049387, -0.33837046],
                          [ 0.48860656, -0.36391687],
                          [ 0.49306303, -0.36041884],
                          [ 0.48857066, -0.36014334],
                          [ 0.49082097, -0.35798846],
                          [ 0.48795341, -0.35725574],
                          [ 0.47550248, -0.34563929],
                          [ 0.47597734, -0.34317242],
                          [ 0.47430852, -0.34838188],
                          [ 0.47135694, -0.343806  ],
                          [ 0.47173304, -0.34716923],
                          [ 0.48328937, -0.36486577],
                          [ 0.48526361, -0.36147505],
                          [ 0.48602475, -0.36457737],
                          [ 0.48307135, -0.35862503],
                          [ 0.48026792, -0.36101802],
                          [ 0.4765167 , -0.35649775],
                          [ 0.4776168 , -0.35916183],
                          [ 0.48451067, -0.33509927],
                          [ 0.48464161, -0.33258111],
                          [ 0.48046638, -0.33523696],
                          [ 0.48173872, -0.33102942],
                          [ 0.47978138, -0.3327397 ]])

    center = SphereCoordinate(center[0], center[1]+pi/2)
    neighbors = [SphereCoordinate(s[0], s[1]+pi/2) for s in neighbors]
    
    precision = 1e-7
    cell = VoronoiCell(center, rng=np.random.RandomState(0), precision=precision)
    triIx = cell.initialize_with_tri(neighbors)

def test_ortho_great_circle():
    np.random.seed(0)
    
    for i in range(10):
        # random points confined to the positive octant
        v = SphereCoordinate(np.random.rand(3))
        w0 = SphereCoordinate(np.random.rand(3))
        G = GreatCircle.ortho(v, w0)
        f = G.ring(as_angle=True)
        d0 = v.geo_dist(w0)
        assert np.isclose(v.geo_dist(f(.01)), v.geo_dist(f(-.01))), "Not symmetric."
        assert (d0 < v.geo_dist(f(.001))) and (d0 < v.geo_dist(f(-.001))), "Not min distance."
        assert v.dot(G.w)>0

def test_haversine():
    np.random.RandomState(0)
    # simple checks for distances between the same point
    assert haversine([0,0], [0,0])==0
    assert np.isclose(haversine([0,0], [2*pi,0]), 0)
    
    # make sure that random rotations don't change distances
    x = SphereCoordinate(1,0,0)
    y = SphereCoordinate(0,1,0)
    z = SphereCoordinate(0,0,1)
    assert np.isclose([x.geo_dist(z), z.geo_dist(x), y.geo_dist(z)], pi/2).all()
    
    for i in range(3):
        # some random rotation axis
        rotvec = np.random.normal(size=3)
        d = np.random.uniform(0, 2*pi)

        x = x.rotate(rotvec, d)
        y = y.rotate(rotvec, d)
        z = z.rotate(rotvec, d)

        # these should all be the same
        assert np.isclose([x.geo_dist(z), z.geo_dist(x), y.geo_dist(z)], pi/2).all()

def test_quaternion():
    # Check that rotations combine approriately
    theta = pi/4
    r=Quaternion(cos(theta/2),0,0,sin(theta/2))
    r2=r.hprod(r)
    assert np.isclose(r2.real,cos(pi/4)) and np.isclose(r2.vec[-1],sin(pi/4)),r2
    r3=r2.hprod(r)
    assert np.isclose(r3.real,cos(3*pi/8)) and np.isclose(r3.vec[-1],sin(3*pi/8)),r3
    r4=r3.hprod(r)
    assert np.isclose(r4.real,cos(pi/2)) and np.isclose(r4.vec[-1],sin(pi/2)),r4
    
    # compare with rotation matrix
    theta=pi/2
    p=Quaternion(0,1.,0,0)
    q=Quaternion(cos(theta/2),0,sin(theta/2),0)
    assert np.isclose( q.hprod(p.hprod(q.inv())).vec, q.rotmat().dot(p.vec) ).all()

    # check that inverse rotation on rotation return original vector
    theta=pi/2
    p=Quaternion(0,1.,0,0)
    q=Quaternion(cos(theta/2),0,sin(theta/2),0)
    assert np.isclose( p.vec, p.rotate(q).rotate(q.inv()).vec ).all()

def test_SphereCoordinate():
    from numpy import pi
    np.random.seed(0)
    rotvec=np.random.rand(3)*2-1 
    rotvec/=np.linalg.norm(rotvec) 

    # Check that a full rotation returns you to the same point
    coord=SphereCoordinate(0,1)
    newcoord=coord.rotate(rotvec,2*pi)
    assert np.isclose([0,1], (newcoord.phi,newcoord.theta)).all(),(newcoord.phi,newcoord.theta)
    coord=SphereCoordinate(pi/7,2*pi/3)
    newcoord=coord.rotate(rotvec,2*pi)
    assert np.isclose([pi/7,2*pi/3], (newcoord.phi,newcoord.theta)).all(), (newcoord.phi,newcoord.theta)

    # Check that a full rotation (broken into two parts) returns you to the same point
    coord=SphereCoordinate(0,1)
    newcoord=coord.rotate(rotvec,np.pi).rotate(rotvec,np.pi)
    assert np.isclose([0,1], (newcoord.phi,newcoord.theta)).all(), (newcoord.phi,newcoord.theta)
    coord = SphereCoordinate(pi/7, 2*pi/3)
    newcoord = coord.rotate(rotvec, np.pi).rotate(rotvec, np.pi)
    assert np.isclose([pi/7,2*pi/3], (newcoord.phi,newcoord.theta)).all(), (newcoord.phi,newcoord.theta)

    # Check that a rotation and its inverse return you to same point
    coord=SphereCoordinate(0,1)
    newcoord=coord.rotate(rotvec,np.pi).rotate(rotvec,-np.pi)
    assert np.isclose([0,1], (newcoord.phi,newcoord.theta)).all(), (newcoord.phi,newcoord.theta)
    
    coord=SphereCoordinate(0, pi/2)
    newcoord=coord.rotate(rotvec, 3*pi/2).rotate(rotvec, pi/2)
    assert np.isclose((coord.phi,coord.theta), (newcoord.phi,newcoord.theta), atol=1e-10).all(), (newcoord.phi,newcoord.theta)

def test_jitSphereCoordinate():
    from numpy import pi
    np.random.seed(0)
    rotvec=np.random.rand(3)*2-1 
    rotvec/=np.linalg.norm(rotvec) 

    # Check that a full rotation returns you to the same point
    coord=jitSphereCoordinate(0,1)
    newcoord=coord.rotate(rotvec,2*pi)
    assert np.isclose([0,1], (newcoord.phi,newcoord.theta)).all(),(newcoord.phi,newcoord.theta)
    coord=jitSphereCoordinate(pi/7,2*pi/3)
    newcoord=coord.rotate(rotvec,2*pi)
    assert np.isclose([pi/7,2*pi/3], (newcoord.phi,newcoord.theta)).all(), (newcoord.phi,newcoord.theta)

    # Check that a full rotation (broken into two parts) returns you to the same point
    coord=jitSphereCoordinate(0,1)
    newcoord=coord.rotate(rotvec,np.pi).rotate(rotvec,np.pi)
    assert np.isclose([0,1], (newcoord.phi,newcoord.theta)).all(), (newcoord.phi,newcoord.theta)
    coord=jitSphereCoordinate(pi/7,2*pi/3)
    newcoord=coord.rotate(rotvec,np.pi).rotate(rotvec,np.pi)
    assert np.isclose([pi/7,2*pi/3], (newcoord.phi,newcoord.theta)).all(), (newcoord.phi,newcoord.theta)

    # Check that a rotation and its inverse return you to same point
    coord=jitSphereCoordinate(0,1)
    newcoord=coord.rotate(rotvec,np.pi).rotate(rotvec,-np.pi)
    assert np.isclose([0,1], (newcoord.phi,newcoord.theta)).all(), (newcoord.phi,newcoord.theta)
    
    coord=jitSphereCoordinate(0, pi/2)
    newcoord=coord.rotate(rotvec, 3*pi/2).rotate(rotvec, pi/2)
    assert np.isclose((coord.phi,coord.theta), (newcoord.phi,newcoord.theta), atol=1e-10).all(), (newcoord.phi,newcoord.theta)

def test_PoissonDiscSphere(use_coarse_grid=True):
    from scipy.spatial.distance import pdist
    
    # Generate coarse grid
    # These include the discontinuity in phi in the interval.
    if use_coarse_grid:
        poissdCoarse = PoissonDiscSphere(pi/50*3,
                                         fast_sample_size=5,
                                         width_bds=(2*pi-.25,.25),
                                         height_bds=(0,.5),
                                         rng=np.random.RandomState(0))
        poissdCoarse.sample()
        cg = poissdCoarse.samples
    else:
        cg = None

    poissd = PoissonDiscSphere(pi/50,
                               fast_sample_size=5,
                               coarse_grid=cg,
                               width_bds=(2*pi-.25,.25),
                               height_bds=(0,.5),
                               rng=np.random.RandomState(1))
    poissd.sample()
    
    assert poissd.within_limits(np.array([0, .2]))
    assert not poissd.within_limits(np.array([.5, .2]))
    assert poissd.within_limits(np.array([[0, .2]]))
    assert not poissd.within_limits(np.array([[.5, .2]]))
    print("Test passed: Boundaries checking works.")

    pt = np.array([.2,.3])
    nearestix = poissd.closest_neighbor(pt)
    d = poissd.dist(pt, poissd.samples)
    assert nearestix==np.argmin(d)
    print("Test passed: Closest neighbor is the closest one in the entire sample.")

    samples = poissd.samples.copy()
    poissd.expand(1)
    assert np.isclose(samples, poissd.samples).all()
    print("Test passed: nothing changes when factor=1.")

    samples = poissd.samples.copy()
    d = [pdist(poissd.samples, jithaversine)]
    poissd.expand(.5)
    d.append(pdist(poissd.samples, jithaversine))
    poissd.expand(2)
    d.append(pdist(poissd.samples, jithaversine))
    # there will be some variation because of the way that we're calculating center of mass in 3D
    assert np.isclose(samples, poissd.samples, atol=1e-6, rtol=1).all()
    print("Test passed: points remain roughly unchanged when contracted and expanded by reciprocal factors.")
    
    assert (np.isclose(d[0]/2, d[1], atol=1e-6, rtol=1).all() and
            np.isclose(d[0], d[2], atol=1e-6, rtol=1).all())
    print("Test passed: pairwise distances remain unchanged.")

    # allow samples to grow to entire region of globe to test whether cutoffs work properly or not
    poissd.expand(10, force=True, truncate_to_bounds=False)
    assert ( (0<poissd.samples[:,0])&((2*pi)>poissd.samples[:,0]) ).all()
    assert ( ((-pi/2)<poissd.samples[:,1])&((pi/2)>poissd.samples[:,1]) ).all()
    assert ( (0<poissd.coarseGrid[:,0])&((2*pi)>poissd.coarseGrid[:,0]) ).all()
    assert ( ((-pi/2)<poissd.coarseGrid[:,1])&((pi/2)>poissd.coarseGrid[:,1]) ).all()

if __name__=='__main__':
    test_jitSphereCoordinate()

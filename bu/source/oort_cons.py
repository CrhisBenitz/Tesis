import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import MT as mt
from multiprocessing import Pool
import contextlib

from scipy.optimize import minimize
import emcee as mc
import sys
sys.path.append("/fs/nas14/other0/cbenitez/Pkgs/")
import corner

from operator import gt,lt,le,ge,eq,ne

def vb_model(c,theta):
    A,B,C,K = theta
    #return -c[0]*np.sin(c[2])*(K+C*np.cos(2*c[1])+A*np.sin(2*c[1]))+v_LSR[0]*np.sin(c[2])*np.cos(c[1])+v_LSR[1]*np.sin(c[2])*np.sin(c[1])
    #return -c[0]*np.sin(c[2])*np.cos(c[2])*(K+C*np.cos(2*c[1])+A*np.sin(2*c[1]))+mvz1(c[0]*np.sin(c[2]))*np.cos(c[2])
    return -c[0]*np.sin(c[2])*np.cos(c[2])*(K+C*np.cos(2*c[1])+A*np.sin(2*c[1]))

def vell_model(c,theta):
    A,B,C,K = theta
    #return c[0]*np.cos(c[2])*(B+A*np.cos(2*c[1])-C*np.sin(2*c[1]))-v_LSR[0]*np.sin(c[1])-v_LSR[1]*np.cos(c[1])
    return c[0]*np.cos(c[2])*(B+A*np.cos(2*c[1])-C*np.sin(2*c[1]))

def vr_model(c,theta):
    A,B,C,K = theta
    #return c[0]*np.cos(c[2])*(B+A*np.cos(2*c[1])-C*np.sin(2*c[1]))-v_LSR[0]*np.sin(c[1])-v_LSR[1]*np.cos(c[1])
    #return c[0]*np.cos(c[2])*np.cos(c[2])*(K+C*np.cos(2*c[1])+A*np.sin(2*c[1]))+mvz1(c[0]*np.sin(c[2]))*np.sin(c[2])
    return c[0]*np.cos(c[2])*np.cos(c[2])*(K+C*np.cos(2*c[1])+A*np.sin(2*c[1]))

def vb_model_ext(c,theta):
    A,B,C,K,D,d,E,e,k = theta
    #return -c[0]*np.sin(c[2])*(K+C*np.cos(2*c[1])+A*np.sin(2*c[1]))+v_LSR[0]*np.sin(c[2])*np.cos(c[1])+v_LSR[1]*np.sin(c[2])*np.sin(c[1])
    #return -c[0]*np.sin(c[2])*np.cos(c[2])*(K+C*np.cos(2*c[1])+A*np.sin(2*c[1]))+mvz1(c[0]*np.sin(c[2]))*np.cos(c[2])
    return -.5*c[0]*(K*np.sin(2*c[2])+C*np.cos(2*c[1])*np.sin(2*c[2])+A*np.sin(2*c[1])*np.sin(2*c[2])\
    +2*D*np.cos(2*c[2])*np.sin(c[1])+2*E*np.cos(2*c[2])*np.cos(c[1])-k*np.sin(2*c[2])\
    -2*d*np.sin(c[1])-2*e*np.cos(c[1]))

def vell_model_ext(c,theta):
    A,B,C,K,D,d,E,e,k = theta
    #return c[0]*np.cos(c[2])*(B+A*np.cos(2*c[1])-C*np.sin(2*c[1]))-v_LSR[0]*np.sin(c[1])-v_LSR[1]*np.cos(c[1])
    return c[0]*(B*np.cos(c[2])+A*np.cos(2*c[1])*np.cos(c[2])-C*np.sin(2*c[1])*np.cos(c[2])\
    -(D+d)*np.sin(c[2])*np.cos(c[1])+(E+e)*np.sin(c[2])*np.sin(c[1]))

def vr_model_ext(c,theta):
    A,B,C,K,D,d,E,e,k = theta
    #return c[0]*np.cos(c[2])*(B+A*np.cos(2*c[1])-C*np.sin(2*c[1]))-v_LSR[0]*np.sin(c[1])-v_LSR[1]*np.cos(c[1])
    #return c[0]*np.cos(c[2])*np.cos(c[2])*(K+C*np.cos(2*c[1])+A*np.sin(2*c[1]))+mvz1(c[0]*np.sin(c[2]))*np.sin(c[2])
    return c[0]*(K*np.cos(c[2])*np.cos(c[2])+C*np.cos(2*c[1])*np.cos(c[2])*np.cos(c[2])+A*np.sin(2*c[1])*np.cos(c[2])*np.cos(c[2])\
    -2*D*np.sin(c[2])*np.cos(c[2])*np.sin(c[1])-2*E*np.sin(c[2])*np.cos(c[2])*np.cos(c[1])+k*np.sin(c[2])*np.sin(c[2]))

def cart2galactic(coord,coord_sun):
    zh = np.array([0.,0.,1.])
    Xh_new = -coord_sun/np.linalg.norm(coord_sun)
    Yh_new = np.cross(zh,Xh_new)
    dr = coord-coord_sun
    drn = np.linalg.norm(dr)
    ell = np.arctan2(Yh_new.dot(dr),Xh_new.dot(dr))
    b = np.arcsin((coord[2]-coord_sun[2])/drn)
    return np.array([drn,ell,b])

def cgalac2ccart(coord_galac,coord_sun):
    coord_sun = np.array(coord_sun)
    coord_galac = np.array(coord_galac)
    X = coord_sun[0] - coord_galac[0]*np.cos(coord_galac[1])*np.cos(coord_galac[2])
    Y = coord_sun[1] - coord_galac[0]*np.sin(coord_galac[1])*np.cos(coord_galac[2])
    Z = coord_sun[2] + coord_galac[0]*np.sin(coord_galac[2])
    return np.array([X, Y, Z])

def vcart2vgalactic(coord,coord_sun,vel,vel_sun):
    zh = np.array([0.,0.,1.])
    omega = -np.linalg.norm(vel_sun)/np.linalg.norm(coord_sun)*zh
    dr = (coord-coord_sun)
    v = vel-vel_sun-np.cross(omega,dr)

    v_r = v.dot(dr)/np.linalg.norm(dr)
    v_ell = v.dot(np.array([-dr[1],dr[0],0]))/np.linalg.norm([-dr[1],dr[0],0])
    v_b = v.dot( np.array([-dr[0]*dr[2],dr[1]*dr[2],dr[0]**2+dr[1]**2]) )/np.linalg.norm(dr)/np.linalg.norm([dr[0],dr[1],0])
    return np.array([v_r, v_ell, v_b])

def vgalac2vcart(coord_galac,coord_sun,vel_galac,vel_sun):
    zh = np.array([0.,0.,1.])
    omega = -np.linalg.norm(vel_sun)/np.linalg.norm(coord_sun)*zh
    coord = cgalac2ccart(coord_galac,coord_sun)
    theta_sun = np.arctan2(coord_sun[1],coord_sun[0])
    dr = (coord-coord_sun)
    aux = vel_sun+np.cross(omega,dr)

    v_x = vel_galac[0]*np.cos(coord_galac[2])*np.cos(coord_galac[1])\
        - vel_galac[1]*np.sin(coord_galac[1])\
        - vel_galac[2]*np.sin(coord_galac[2])*np.cos(coord_galac[1])
    v_y = vel_galac[0]*np.cos(coord_galac[2])*np.sin(coord_galac[1])\
        + vel_galac[1]*np.cos(coord_galac[1])\
        - vel_galac[2]*np.sin(coord_galac[2])*np.sin(coord_galac[1])
    v_z = vel_galac[0]*np.sin(coord_galac[2])\
        + vel_galac[2]*np.cos(coord_galac[2])

    v_X = -v_x*np.cos(theta_sun)+v_y*np.sin(theta_sun)
    v_Y = -v_x*np.sin(theta_sun)-v_y*np.cos(theta_sun)
    v_Z = v_z

    return np.array([v_X, v_Y, v_Z])+aux

def vcart2vgalacticOC(coord,coord_sun,vel,vel_sun):

    dr = (coord-coord_sun)
    v = vel-vel_sun

    v_r = v.dot(dr)/np.linalg.norm(dr)
    v_ell = v.dot(np.array([-dr[1],dr[0],0]))/np.linalg.norm([-dr[1],dr[0],0])
    v_b = v.dot( np.array([-dr[0]*dr[2],dr[1]*dr[2],dr[0]**2+dr[1]**2]) )/np.linalg.norm(dr)/np.linalg.norm([dr[0],dr[1],0])
    return np.array([v_r, v_ell, v_b])

def vgalacOC2vcart(coord_galac,coord_sun,vel_galac,vel_sun):

    theta_sun = np.arctan2(coord_sun[1],coord_sun[0])

    v_x = vel_galac[0]*np.cos(coord_galac[2])*np.cos(coord_galac[1])\
        - vel_galac[1]*np.sin(coord_galac[1])\
        - vel_galac[2]*np.sin(coord_galac[2])*np.cos(coord_galac[1])
    v_y = vel_galac[0]*np.cos(coord_galac[2])*np.sin(coord_galac[1])\
        + vel_galac[1]*np.cos(coord_galac[1])\
        - vel_galac[2]*np.sin(coord_galac[2])*np.sin(coord_galac[1])
    v_z = vel_galac[0]*np.sin(coord_galac[2])\
        + vel_galac[2]*np.cos(coord_galac[2])

    v_X = -v_x*np.cos(theta_sun)+v_y*np.sin(theta_sun)
    v_Y = -v_x*np.sin(theta_sun)-v_y*np.cos(theta_sun)
    v_Z = v_z

    return np.array([v_X, v_Y, v_Z])+vel_sun

def Rot_mat(alpha, axis):
    axis = np.array(axis)/np.linalg.norm(axis)
    R = np.array([[np.cos(alpha)+axis[0]**2*(1-np.cos(alpha)), axis[0]*axis[1]*(1-np.cos(alpha))-axis[2]*np.sin(alpha), axis[0]*axis[2]*(1-np.cos(alpha))+axis[1]*np.sin(alpha)],\
          [axis[0]*axis[1]*(1-np.cos(alpha))+axis[2]*np.sin(alpha), np.cos(alpha)+axis[1]**2*(1-np.cos(alpha)), axis[1]*axis[2]*(1-np.cos(alpha))-axis[0]*np.sin(alpha)],\
          [axis[0]*axis[2]*(1-np.cos(alpha))-axis[1]*np.sin(alpha), axis[1]*axis[2]*(1-np.cos(alpha))+axis[0]*np.sin(alpha), np.cos(alpha)+axis[2]**2*(1-np.cos(alpha))] \
         ])
    return R

def m1_coord(N,r): #Modelo 1: circunferencia a z=0 con velocidad circular. Genero \ell aleaotorios

    ll = np.random.rand(N)*2*np.pi-np.pi
    xx = r*np.cos(ll)
    yy = r*np.sin(ll)
    zz = np.zeros(N)
    return np.array([xx, yy, zz])


def m3_coord(N,r,i,vnod): #Modelo 3: circunferencia con una inclinacion i respecto al plano xy y direccion de los nodos dada por el vector vnod.

    ll = np.random.rand(N)*2*np.pi-np.pi
    xx = r*np.cos(ll)
    yy = r*np.sin(ll)
    zz = np.zeros(N)
    xyz = np.array([xx, yy, zz]).T
    Mrot = Rot_mat(i,vnod)
    xyz = np.array([Mrot.dot(c) for c in xyz]).T
    return xyz

def m3b_coord(N,rmin,rmax,i,vnod): #Modelo 3: circunferencia con una inclinacion i respecto al plano xy y direccion de los nodos dada por el vector vnod.

    ll = np.random.rand(N)*2*np.pi-np.pi
    rr = np.random.rand(N)*(rmax-rmin)+rmin
    xx = rr*np.cos(ll)
    yy = rr*np.sin(ll)
    zz = np.zeros(N)
    xyz = np.array([xx, yy, zz]).T
    Mrot = Rot_mat(i,vnod)
    xyz = np.array([Mrot.dot(c) for c in xyz]).T
    return xyz

def mvz1(cz, vz_max = .2):
    hz = .5
    if abs(cz) > hz:
        return -vz_max*cz/abs(cz)
    return -vz_max*cz/hz

def m1_vels(coords,vz_dist = mvz1):
    kpc2km = mt.C.kpc.express(mt.C.km)
    #return np.array([np.append(mt.vel_circ(c,mt.force_T)[:2]*kpc2km,vz_dist(c[2])) for c in coords.T]).T
    #return np.array([np.append(mt.vel_circ(c,mt.force_T)[:2],vz_dist(c[2])/kpc2km) for c in coords.T]).T
    return np.array([mt.vel_circ(c,mt.force_T) for c in coords.T]).T

def log_likelihood(theta, coords, vels):
    #A,B,C,K = theta
    vreq = [vr_model(c,theta) for c in coords]
    velleq = [vell_model(c,theta) for c in coords]
    vbeq = [vb_model(c,theta) for c in coords]
    vvr = np.array([v[0] for v in vels])
    vvell = np.array([v[1] for v in vels])
    vvb = np.array([v[2] for v in vels])

    #return -0.5 * np.sum(np.sqrt((vvr - vreq)** 2) + np.sqrt((vvell - velleq)** 2) + np.sqrt((vvb - vbeq)** 2) )
    return -0.5 * np.sum(np.sqrt((vvr - vreq)** 2) + np.sqrt((vvell - velleq)** 2) )

def log_prior(theta):
    A,B,C,K = theta
    #if -20.0 < A < 20 and -20 < B < 20.0 and -20.0 < C < 20.0 and -20.<K<20.:
    #    return 0.0
    #return -np.inf
    return 0.0

def log_probability(theta, coords, vels):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, coords, vels)

def log_likelihood_global(theta):
    #A,B,C,K = theta
    vreq = [vr_model(c,theta) for c in coords]
    velleq = [vell_model(c,theta) for c in coords]
    vbeq = [vb_model(c,theta) for c in coords]
    vvr = np.array([v[0] for v in vels])
    vvell = np.array([v[1] for v in vels])
    vvb = np.array([v[2] for v in vels])
    return -.5*np.sum(np.sqrt( (vvr - vreq)** 2 + (vvell - velleq)** 2 ))

def log_probability_global(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_global(theta)

def log_likelihood_ext(theta, coords, vels):
    #A,B,C,K = theta
    vreq = [vr_model_ext(c,theta) for c in coords]
    velleq = [vell_model_ext(c,theta) for c in coords]
    vbeq = [vb_model_ext(c,theta) for c in coords]
    vvr = np.array([v[0] for v in vels])
    vvell = np.array([v[1] for v in vels])
    vvb = np.array([v[2] for v in vels])

    #return -0.5 * np.sum(np.sqrt((vvr - vreq)** 2) + np.sqrt((vvell - velleq)** 2) + np.sqrt((vvb - vbeq)** 2) )
    return -.5*np.sum(np.sqrt( (vvr - vreq)** 2 + (vvell - velleq)** 2 + (vvb - vbeq)** 2))

def log_prior_ext(theta):
    A,B,C,K,D,d,E,e,k = theta
    #if -20.0 < A < 20 and -20 < B < 20.0 and -20.0 < C < 20.0 and -20.<K<20.:
    #    return 0.0
    #return -np.inf
    return 0.0

def log_probability_ext(theta, coords, vels):
    lp = log_prior_ext(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_ext(theta, coords, vels)

def log_likelihood_ext_global(theta):
    #A,B,C,K = theta
    vreq = [vr_model_ext(c,theta) for c in coords]
    velleq = [vell_model_ext(c,theta) for c in coords]
    vbeq = [vb_model_ext(c,theta) for c in coords]
    vvr = np.array([v[0] for v in vels])
    vvell = np.array([v[1] for v in vels])
    vvb = np.array([v[2] for v in vels])
    return -.5*np.sum(np.sqrt( (vvr - vreq)** 2 + (vvell - velleq)** 2 + (vvb - vbeq)** 2))

def log_probability_ext_global(theta):
    lp = log_prior_ext(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_ext_global(theta)


def rut_fit(r=.5,r_LSR=[8.,0.,0.],Npart=500, Nsteps=2000, Nwalkers=500,vexp=0.,vrot=0., inc=0., vec_nod=[0,2,0]):
    r_LSR = np.array(r_LSR)

    if inc!=0:
        m3 = ((m3_coord(Npart,1.,inc,vec_nod).T)+r_LSR).T
    else:
        m3 = ((m1_coord(Npart,r).T)+r_LSR).T

    vv = m1_vels(m3,mvz1).T
    cc = m3.T

    v_LSR = mt.vel_circ(r_LSR,mt.force_T,)*mt.C.kpc.express(mt.C.km)

    vvkm = vv*mt.C.kpc.express(mt.C.km)

    vv_galac = np.array([vcart2vgalacticOC(cc[i],r_LSR,vvkm[i],v_LSR) for i in range(len(vvkm))])
    cc_galac = np.array([cart2galactic(c,r_LSR) for c in cc])

    if vexp!=0:
        vv_galac = [v+np.array([v_exp,0.,0.]) for v in vv_galac]

    if vrot!=0:
        vv_galac = [v+np.array([0.,v_rot,0.]) for v in vv_galac]

    np.random.seed(42)
    nll = lambda *args: -log_likelihood(*args)
    initial = np.array([0.,0.,0.,0.])
    print("minimizing")
    soln = minimize(nll, initial,method = 'Nelder-Mead', args=(cc_galac, vv_galac))
    #soln = np.array([0.,0.,0.,0.])

    #pos = soln.x + 1e-4 * np.random.randn(Nwalkers, 4)
    pos = soln.x + 10 * np.random.randn(Nwalkers, 4)
    nwalkers, ndim = pos.shape

    sampler = mc.EnsembleSampler(Nwalkers, ndim, log_probability, args=(cc_galac, vv_galac))
    sampler.run_mcmc(pos, Nsteps,progress=True);


    flat_samples = sampler.get_chain(discard=0, flat=True)

    mcmc = [np.percentile(flat_samples[:, i], [16, 50, 84]) for i in range(ndim)]
    qq = [np.diff(mcmc[i]) for i in range(ndim)]
    theta_f = [par[1] for par in mcmc]

    print "[A, B, C, D]:",theta_f

    return cc, vv, cc_galac, vv_galac, soln, mcmc, qq, theta_f, sampler


def rut_fit_global(r=.5,r_LSR=[8.,0.,0.],Npart=500, Nsteps=2000, Nwalkers=500,vexp=0.,vrot=0., inc=0., vec_nod=[0,2,0]):
    r_LSR = np.array(r_LSR)

    if inc!=0:
        m3 = ((m3_coord(Npart,1.,inc,vec_nod).T)+r_LSR).T
    else:
        m3 = ((m1_coord(Npart,r).T)+r_LSR).T

    vv = m1_vels(m3,mvz1).T
    cc = m3.T

    v_LSR = mt.vel_circ(r_LSR,mt.force_T,)*mt.C.kpc.express(mt.C.km)

    vvkm = vv*mt.C.kpc.express(mt.C.km)

    vv_galac = np.array([vcart2vgalacticOC(cc[i],r_LSR,vvkm[i],v_LSR) for i in range(len(vvkm))])
    cc_galac = np.array([cart2galactic(c,r_LSR) for c in cc])

    if vexp!=0:
        vv_galac = [v+np.array([vexp,0.,0.]) for v in vv_galac]

    if vrot!=0:
        vv_galac = [v+np.array([0.,vrot,0.]) for v in vv_galac]

    global coords, vels
    coords = cc_galac
    vels = vv_galac

    random_seed = 42
    np.random.seed(random_seed)
    #nll = lambda *args: -log_likelihood(*args)
    #initial = np.array([0.,0.,0.,0.])
    #print("minimizing")
    #soln = minimize(nll, initial,method = 'Nelder-Mead', args=(cc_galac, vv_galac))
    soln = np.array([0.,0.,0.,0.])

    #pos = soln.x + 1e-4 * np.random.randn(Nwalkers, 4)
    pos = soln + 10 * (np.random.rand(Nwalkers, 4)*2.-1.)
    nwalkers, ndim = pos.shape

    with contextlib.closing(Pool()) as pool:
        sampler = mc.EnsembleSampler(Nwalkers, ndim, log_probability_global, pool=pool)
        sampler.run_mcmc(pos, Nsteps,progress=True);


    flat_samples = sampler.get_chain(discard=0, flat=True)

    mcmc = [np.percentile(flat_samples[:, i], [16, 50, 84]) for i in range(ndim)]
    qq = [np.diff(mcmc[i]) for i in range(ndim)]
    theta_f = [par[1] for par in mcmc]

    print "[A, B, C, D]:",theta_f

    info_dic = {"r":r,"r_LSR":r_LSR,"Npart":Npart, "Nsteps":Nsteps, "Nwalkers":Nwalkers,"vexp":vexp,"vrot":vrot, "inc":inc, "vec_nod":vec_nod,"random_seed":42}

    return cc, vv, cc_galac, vv_galac, soln, mcmc, qq, theta_f, sampler, info_dic


def sim_fit_global(DF,r_LSR=[8.,0.,0.], Nsteps=2000, Nwalkers=500):

    r_LSR = np.array(r_LSR)
    v_LSR = mt.vel_circ(r_LSR,mt.force_T,)*mt.C.kpc.express(mt.C.km)

    cc = np.array(DF[['x','y','z']])
    vv = np.array(DF[['vx','vy','vz']])
    vvkm = vv*mt.C.kpc.express(mt.C.km)

    vv_galac = np.array([vcart2vgalacticOC(cc[i],r_LSR,vvkm[i],v_LSR) for i in range(len(vvkm))])
    cc_galac = np.array([cart2galactic(c,r_LSR) for c in cc])

    global coords, vels
    coords = cc_galac
    vels = vv_galac

    random_seed = 42
    np.random.seed(random_seed)
    #nll = lambda *args: -log_likelihood(*args)
    #initial = np.array([0.,0.,0.,0.])
    #print("minimizing")
    #soln = minimize(nll, initial,method = 'Nelder-Mead', args=(cc_galac, vv_galac))
    soln = np.array([0.,0.,0.,0.])

    #pos = soln.x + 1e-4 * np.random.randn(Nwalkers, 4)
    pos = soln + 10 * (np.random.rand(Nwalkers, 4)*2.-1.)
    nwalkers, ndim = pos.shape

    with contextlib.closing(Pool()) as pool:
        sampler = mc.EnsembleSampler(Nwalkers, ndim, log_probability_global, pool=pool)
        sampler.run_mcmc(pos, Nsteps,progress=True);


    flat_samples = sampler.get_chain(discard=0, flat=True)

    mcmc = [np.percentile(flat_samples[:, i], [16, 50, 84]) for i in range(ndim)]
    qq = [np.diff(mcmc[i]) for i in range(ndim)]
    theta_f = [par[1] for par in mcmc]

    print "[A, B, C, D]:",theta_f

    info_dic = {"r_LSR":r_LSR,"Npart":len(cc), "Nsteps":Nsteps, "Nwalkers":Nwalkers,"random_seed":random_seed}

    return cc, vv, cc_galac, vv_galac, soln, mcmc, qq, theta_f, sampler, info_dic

def plot_particles(cc, saveplot=False):
        fig = plt.figure(figsize=(16,7))
        ax = fig.add_subplot(1, 2, 1)
        ax.scatter(cc.T[0],\
                   cc.T[1],\
                   s=4, alpha=.5)
        ax.set_xlabel(r'$x \mathrm{[kpc]}$',fontsize=10)
        ax.set_ylabel(r'$y \mathrm{[kpc]}$',fontsize=10)
        #ax.set_zlabel(r'$z \mathrm{[kpc]}$',fontsize=10)
        ax.set_aspect('equal')
        ax.set_xlim(6,10)
        #ax.set_xlim(-2,2)
        ax.set_ylim(-2,2)
        #ax.set_zlim(-2,2)

        ax = fig.add_subplot(1, 2, 2)
        ax.scatter(cc.T[0],\
                   cc.T[2],\
                   s=4, alpha=.5)
        ax.set_xlabel(r'$x \mathrm{[kpc]}$',fontsize=10)
        ax.set_ylabel(r'$y \mathrm{[kpc]}$',fontsize=10)
        #ax.set_zlabel(r'$z \mathrm{[kpc]}$',fontsize=10)
        ax.set_aspect('equal')
        ax.set_xlim(6,10)
        #ax.set_xlim(-2,2)
        ax.set_ylim(-2,2)
        #ax.set_zlim(-2,2)
        #plt.show()
        return

def plot_walks(sampler,discard_first=0):
        fig, axes = plt.subplots(4, figsize=(10, 7), sharex=True)
        samples = sampler.get_chain(discard=discard_first)
        labels = ["A", "B", "C",'K']
        for i in range(4):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("Pasos");
        return

def plot_corner(sampler,discard_one_of=4):
        flat_samples = sampler.get_chain(discard=sampler.chain.shape[1]/discard_one_of, thin=15, flat=True)
        if sampler.ndim==4:
            param_names = ["A","B","C","K"]
        else:
            param_names = ["A","B","C","K","D","D'","E","E'","K'"]

        fig = corner.corner(
            flat_samples,  labels=param_names
        );
        return

def plot_fit(fitObj,figax=None):
    ll = np.linspace(-np.pi,np.pi,500)

    #fig.subplots_adjust(hspace=0)

    theta_f = np.array(fitObj.theta)
    theta_fp = theta_f + np.array(fitObj.std)
    theta_fm = theta_f - np.array(fitObj.std)
    cc_galac,vv_galac, dd = fitObj.cc_galac, fitObj.vv_galac, fitObj.dd

    if len(theta_f)==4:

        if figax==None:
            fig, axs = plt.subplots(1, 2, sharex=False,figsize=(16,4))
        else:
            fig, axs = figax

        tth = [ [theta_fm[0],theta_fm[1],theta_fm[2],theta_fm[3]],\
                [theta_fm[0],theta_fm[1],theta_fp[2],theta_fm[3]],\
                [theta_fp[0],theta_fm[1],theta_fm[2],theta_fm[3]],\
                [theta_fp[0],theta_fm[1],theta_fp[2],theta_fm[3]],\
                [theta_fp[0],theta_fp[1],theta_fp[2],theta_fp[3]],\
                [theta_fp[0],theta_fp[1],theta_fm[2],theta_fp[3]],\
                [theta_fm[0],theta_fp[1],theta_fp[2],theta_fp[3]],\
                [theta_fm[0],theta_fp[1],theta_fm[2],theta_fp[3]]]

        axs[0].plot(ll,[vr_model(c,theta_f) for c in [np.array([1.,l,0.]) for l in ll]],ls='--',lw=1,c="k",alpha=.8)
        axs[0].fill_between(ll,\
                            [min(vr_model(c,th) for th in tth) for c in [np.array([1.,l,0.]) for l in ll]],\
                            [max(vr_model(c,th) for th in tth) for c in [np.array([1.,l,0.]) for l in ll]],\
                            alpha=.5,color="gray")
        axs[0].scatter([c[1] for c in cc_galac],[vv_galac[i][0]/cc_galac[i][0]/np.cos(cc_galac[i][2])/np.cos(cc_galac[i][2]) for i in range(len(vv_galac))],\
                        s=4,c='red',alpha=.5)

        plt.yticks(fontsize=22)

        #axs[0].plot(ll,[vell_model(c,[15.1,-11.8,-9.2,-2.5])/1./np.cos(0.17) for c in [np.array([1.,l,0.17]) for l in ll]])
        #axs[0].plot(ll,[vell_model(c,[15.1,-11.8,-9.2,-2.5])/1./np.cos(.35) for c in [np.array([1.,l,.35]) for l in ll]])
        axs[1].plot(ll,[vell_model(c,theta_f) for c in [np.array([1.,l,0.]) for l in ll]],ls='--',lw=1,c="k",alpha=.8)
        axs[1].fill_between(ll,\
                            [min(vell_model(c,th) for th in tth) for c in [np.array([1.,l,0.]) for l in ll]],\
                            [max(vell_model(c,th) for th in tth) for c in [np.array([1.,l,0.]) for l in ll]],\
                            alpha=.5,color="gray")
        axs[1].scatter([c[1] for c in cc_galac],[vv_galac[i][1]/cc_galac[i][0]/np.cos(cc_galac[i][2]) for i in range(len(vv_galac))],\
                        s=4,c='red',alpha=.5)

        plt.yticks(fontsize=22)

        #axs[2].plot(ll,[vb_model(c,theta_f) for c in [np.array([1.,l,.0]) for l in ll]])
        #axs[2].scatter([c[1] for c in cc_galac],[vv_galac[i][2]/cc_galac[i][0]/np.sin(cc_galac[i][2])/np.cos(cc_galac[i][2]) for i in range(len(vv_galac))],s=4)


        #axs[0].set_yticks(np.arange(-0.9, 1.0, 0.4))
        #axs[0].set_ylim(-1, 1)

        axs[0].locator_params(nbins=7)
        axs[1].locator_params(nbins=7)
        axs[0].set_ylabel(r'$\frac{v_{r}}{r\cos^2{b}}$',fontsize=24)
        axs[1].set_ylabel(r'$\frac{v_{\ell}}{r\cos{b}}$',fontsize=24)
        axs[0].set_xlabel(r'$\ell$',fontsize=24)
        axs[1].set_xlabel(r'$\ell$',fontsize=24)

        #axs[0].invert_xaxis()
        #axs[1].invert_xaxis()
        #axs[2].set_ylabel(r'$\frac{v_{b}}{r\cos{b}\sin{b}}$',fontsize=28)

        axs[0].set_xticks([l for l in np.linspace(np.pi,-np.pi,9)])
        axs[0].set_xticklabels(labels=[str(int(l*180/np.pi)) for l in np.linspace(np.pi,-np.pi,9)])
        axs[1].set_xticks([l for l in np.linspace(np.pi,-np.pi,9)])
        axs[1].set_xticklabels(labels=[str(int(l*180/np.pi)) for l in np.linspace(np.pi,-np.pi,9)])

        axs[0].xaxis.set_tick_params(labelsize=16)
        axs[1].xaxis.set_tick_params(labelsize=16)
        axs[0].yaxis.set_tick_params(labelsize=16)
        axs[1].yaxis.set_tick_params(labelsize=16)
        plt.tight_layout(w_pad=2.)

        return fig,axs

    else:
        if figax==None:
            fig, axs = plt.subplots(3, 1, sharex=False,figsize=(8,12))
        else:
            fig, axs = figax

        #axs[0].plot(cc_galac[:,1],[vr_model_ext(c,theta_f) for c in cc_galac],ls='--',lw=1,c="k",alpha=.8)
        axs[0].scatter(cc_galac[:,1],[vr_model_ext(c,theta_f) for c in cc_galac],s=map(int,dd*10),c='blue',alpha=.4,label="data")
        axs[0].scatter(cc_galac[:,1],vv_galac[:,0],s=5,c='red',alpha=1,label="fitted")
        plt.yticks(fontsize=22)

        #axs[1].plot(cc_galac[:,1],[vell_model_ext(c,theta_f) for c in cc_galac],ls='--',lw=1,c="k",alpha=.8)
        axs[1].scatter(cc_galac[:,1],[vell_model_ext(c,theta_f) for c in cc_galac],s=map(int,dd*10),c='blue',alpha=.4,label="data")
        axs[1].scatter(cc_galac[:,1],vv_galac[:,1],s=5,c='red',alpha=1,label="fitted")

        plt.yticks(fontsize=22)

        #axs[2].plot(cc_galac[:,1],[vb_model_ext(c,theta_f) for c in cc_galac],ls='--',lw=1,c="k",alpha=.8)
        axs[2].scatter(cc_galac[:,1],[vb_model_ext(c,theta_f) for c in cc_galac],s=map(int,dd*10),c='blue',alpha=.4,label="data")
        axs[2].scatter(cc_galac[:,1],vv_galac[:,2],s=5,c='red',alpha=1,label="fitted",marker="+")

        plt.yticks(fontsize=22)

        #axs[2].plot(ll,[vb_model(c,theta_f) for c in [np.array([1.,l,.0]) for l in ll]])
        #axs[2].scatter([c[1] for c in cc_galac],[vv_galac[i][2]/cc_galac[i][0]/np.sin(cc_galac[i][2])/np.cos(cc_galac[i][2]) for i in range(len(vv_galac))],s=4)


        #axs[0].set_yticks(np.arange(-0.9, 1.0, 0.4))
        #axs[0].set_ylim(-1, 1)

        axs[0].locator_params(nbins=7)
        axs[1].locator_params(nbins=7)
        axs[2].locator_params(nbins=7)
        axs[0].set_ylabel(r'$v_r$',fontsize=24)
        axs[1].set_ylabel(r'$v_{\ell}$',fontsize=24)
        axs[2].set_ylabel(r'$v_{b}$',fontsize=24)
        axs[0].set_xlabel(r'$\ell$  [deg]',fontsize=24)
        axs[1].set_xlabel(r'$\ell$  [deg]',fontsize=24)
        axs[2].set_xlabel(r'$\ell$  [deg]',fontsize=24)

        #axs[0].invert_xaxis()
        #axs[1].invert_xaxis()
        #axs[2].set_ylabel(r'$\frac{v_{b}}{r\cos{b}\sin{b}}$',fontsize=28)

        axs[0].set_xticks([l for l in np.linspace(np.pi,-np.pi,9)])
        axs[0].set_xticklabels(labels=[str(int(l*180/np.pi)) for l in np.linspace(np.pi,-np.pi,9)])
        axs[1].set_xticks([l for l in np.linspace(np.pi,-np.pi,9)])
        axs[1].set_xticklabels(labels=[str(int(l*180/np.pi)) for l in np.linspace(np.pi,-np.pi,9)])
        axs[2].set_xticks([l for l in np.linspace(np.pi,-np.pi,9)])
        axs[2].set_xticklabels(labels=[str(int(l*180/np.pi)) for l in np.linspace(np.pi,-np.pi,9)])

        axs[0].xaxis.set_tick_params(labelsize=16)
        axs[1].xaxis.set_tick_params(labelsize=16)
        axs[0].yaxis.set_tick_params(labelsize=16)
        axs[1].yaxis.set_tick_params(labelsize=16)
        plt.tight_layout(w_pad=2.)

        return fig,axs

def calc_error(cc_galac,vv_galac,theta_f):

    ee_ell = [abs(vell_model(cc_galac[i],theta_f)-vv_galac[i][1]) for i in range(len(cc_galac))]
    ee_r = [abs(vr_model(cc_galac[i],theta_f)-vv_galac[i][0]) for i in range(len(cc_galac))]

    return np.array([ee_r,ee_ell])

def calc_error_rel(cc_galac,vv_galac,theta_f):

    ee_ell = [abs((vell_model(cc_galac[i],theta_f)-vv_galac[i][1])/vv_galac[i][1]) for i in range(len(cc_galac))]
    ee_r   = [abs((vr_model(cc_galac[i],theta_f)-vv_galac[i][0])/vv_galac[i][0]) for i in range(len(cc_galac))]

    return np.array([ee_r,ee_ell])

def plot_error(rrs,ccc,figax=None):

    eee = [calc_error(ccc[i][2],ccc[i][3],ccc[i][7]) for i in range(len(ccc))]

    if figax==None:
        fig, axs = plt.subplots(1, 2, sharex=False,figsize=(16,8))
    else:
        fig, axs = figax

    axs[0].errorbar(rrs,[np.mean(eee[i][0]) for i in range(len(rrs))],yerr = [np.std(eee[i][0]) for i in range(len(rrs))],\
                    marker='v',ecolor="r",capthick=2,mfc='r',ls='-',c='k',ms=10,alpha=.5,label=r"error$_{v_r}$")
    axs[0].errorbar(rrs,[np.mean(eee[i][1]) for i in range(len(rrs))],yerr = [np.std(eee[i][1]) for i in range(len(rrs))],\
                marker='^',ecolor="b",capthick=2,mfc='b',ls='--',c='k',ms=10,alpha=.5,label=r"error$_{v_{\ell}}$")
    axs[0].scatter(rrs,[np.mean(eee[i][0]) for i in range(len(rrs))],s=10,c='r',alpha=1,marker="v")
    axs[0].scatter(rrs,[np.mean(eee[i][1]) for i in range(len(rrs))],s=10,c='b',alpha=1,marker="^")


    axs[1].errorbar(rrs,[np.mean(eee[i][0]) for i in range(len(rrs))],yerr = [np.std(eee[i][0]) for i in range(len(rrs))],\
                    marker='v',ecolor="r",capthick=2,mfc='r',ls='-',c='k',ms=10,alpha=.5,label=r"error$_{v_r}$")
    axs[1].errorbar(rrs,[np.mean(eee[i][1]) for i in range(len(rrs))],yerr = [np.std(eee[i][1]) for i in range(len(rrs))],\
                marker='^',ecolor="b",capthick=2,mfc='b',ls='--',c='k',ms=10,alpha=.5,label=r"error$_{v_{\ell}}$")
    axs[1].scatter(rrs,[np.mean(eee[i][0]) for i in range(len(rrs))],s=10,c='r',alpha=1,marker="v")
    axs[1].scatter(rrs,[np.mean(eee[i][1]) for i in range(len(rrs))],s=10,c='b',alpha=1,marker="^")


    axs[0].set_ylabel(r'$<v-v_{\mathrm{modelo}}>$',fontsize=24)
    axs[1].set_ylabel(r'$<v-v_{\mathrm{modelo}}>$',fontsize=24)
    axs[0].set_xlabel(r'$r \ \ [\mathrm{kpc}]$',fontsize=24)
    axs[1].set_xlabel(r'$r \ \ [\mathrm{kpc}]$',fontsize=24)


    axs[0].set_xticks(rrs[::2])
    axs[0].set_xticklabels(labels=[str(r) for r in rrs[::2]],fontsize=22)
    axs[1].set_xticks(rrs[::2])
    axs[1].set_xticklabels(labels=[str(r) for r in rrs[::2]],fontsize=22)

    axs[0].xaxis.set_tick_params(labelsize=16)
    axs[1].xaxis.set_tick_params(labelsize=16)
    axs[0].yaxis.set_tick_params(labelsize=16)
    axs[1].yaxis.set_tick_params(labelsize=16)

    axs[1].set_xscale("log")
    axs[1].set_yscale("log")

    plt.tight_layout(w_pad=2.)
    axs[0].legend(loc="best")
    axs[1].legend(loc="best")

    return fig,axs

def plot_error_rel(rrs,ccc,figax=None):

    eee = [calc_error_rel(ccc[i][2],ccc[i][3],ccc[i][7]) for i in range(len(ccc))]

    if figax==None:
        fig, axs = plt.subplots(1, 2, sharex=False,figsize=(16,8))
    else:
        fig, axs = figax

    axs[0].errorbar(rrs,[np.mean(eee[i][0]) for i in range(len(rrs))],yerr = [np.std(eee[i][0]) for i in range(len(rrs))],\
                    marker='v',ecolor="r",capthick=2,mfc='r',ls='-',c='k',ms=10,alpha=.5,label=r"error$_{v_r}$")
    axs[0].errorbar(rrs,[np.mean(eee[i][1]) for i in range(len(rrs))],yerr = [np.std(eee[i][1]) for i in range(len(rrs))],\
                marker='^',ecolor="b",capthick=2,mfc='b',ls='--',c='k',ms=10,alpha=.5,label=r"error$_{v_{\ell}}$")
    axs[0].scatter(rrs,[np.mean(eee[i][0]) for i in range(len(rrs))],s=10,c='r',alpha=1,marker="v")
    axs[0].scatter(rrs,[np.mean(eee[i][1]) for i in range(len(rrs))],s=10,c='b',alpha=1,marker="^")


    axs[1].errorbar(rrs,[np.mean(eee[i][0]) for i in range(len(rrs))],yerr = [np.std(eee[i][0]) for i in range(len(rrs))],\
                    marker='v',ecolor="r",capthick=2,mfc='r',ls='-',c='k',ms=10,alpha=.5,label=r"error$_{v_r}$")
    axs[1].errorbar(rrs,[np.mean(eee[i][1]) for i in range(len(rrs))],yerr = [np.std(eee[i][1]) for i in range(len(rrs))],\
                marker='^',ecolor="b",capthick=2,mfc='b',ls='--',c='k',ms=10,alpha=.5,label=r"error$_{v_{\ell}}$")
    axs[1].scatter(rrs,[np.mean(eee[i][0]) for i in range(len(rrs))],s=10,c='r',alpha=1,marker="v")
    axs[1].scatter(rrs,[np.mean(eee[i][1]) for i in range(len(rrs))],s=10,c='b',alpha=1,marker="^")


    axs[0].set_ylabel(r'$<v-v_{\mathrm{modelo}}>$',fontsize=24)
    axs[1].set_ylabel(r'$<v-v_{\mathrm{modelo}}>$',fontsize=24)
    axs[0].set_xlabel(r'$r \ \ [\mathrm{kpc}]$',fontsize=24)
    axs[1].set_xlabel(r'$r \ \ [\mathrm{kpc}]$',fontsize=24)


    axs[0].set_xticks(rrs[::2])
    axs[0].set_xticklabels(labels=[str(r) for r in rrs[::2]],fontsize=22)
    axs[1].set_xticks(rrs[::2])
    axs[1].set_xticklabels(labels=[str(r) for r in rrs[::2]],fontsize=22)

    axs[0].xaxis.set_tick_params(labelsize=16)
    axs[1].xaxis.set_tick_params(labelsize=16)
    axs[0].yaxis.set_tick_params(labelsize=16)
    axs[1].yaxis.set_tick_params(labelsize=16)

    axs[1].set_xscale("log")
    axs[1].set_yscale("log")

    plt.tight_layout(w_pad=2.)
    axs[0].legend(loc="best")
    axs[1].legend(loc="best")

    return fig,axs

def run_ruts():

    run_dir1 = "caso1/"
    run_dir2 = "caso2/"
    run_dir3 = "caso3/"

#    for i in range(1,11):
#        run = rut_fit_global(r=i/10.,r_LSR=[8.,0.,0.],Npart=400, Nsteps=2000, Nwalkers=100,vexp=0.,vrot=0., inc=0., vec_nod=[0,2,0])
#        np.save(mt.tardir+run_dir1+"rut_caso1"+str(i),run)
#        print("Run %g for case 1 saved \n"%i)

    for i in range(10):
        run = rut_fit_global(r=.2,r_LSR=[8.,0.,0.],Npart=400, Nsteps=2000, Nwalkers=100,vexp=-4.5+i,vrot=0., inc=0., vec_nod=[0,2,0])
        np.save(mt.tardir+run_dir2+"rut_caso2"+str(i),run)
        print("Run %g for case 2 saved \n"%i)

    for i in range(10):
        run = rut_fit_global(r=.2,r_LSR=[8.,0.,0.],Npart=400, Nsteps=2000, Nwalkers=100,vexp=0,vrot=-4.5+i, inc=0., vec_nod=[0,2,0])
        np.save(mt.tardir+run_dir3+"rut_caso3"+str(i),run)
        print("Run %g for case 3 saved \n"%i)

    return None


def sim_ruts(DF,DF_LSR,filters,rut_dir,marbles_file):

    for filter in filters:

        r_sun = np.array((DF_LSR[['x','y','z']][filter['sun_filter']]))[0]

        run = sim_fit_global(DF[filter['evaluation']],r_LSR=r_sun, Nsteps=2000, Nwalkers=100)
        np.save(mt.tardir+rut_dir+marbles_file+"_"+filter['name'],run)
        print("Run for filter %s saved \n"%filter['name'])

    return None

def fit_OC(xxvv,xv_LSR, Nsteps=2000, Nwalkers=50,discard_N_over=4):

    r_LSR = np.array(xv_LSR[:3:])
    v_LSR = np.array(xv_LSR[3::])*mt.C.kpc.express(mt.C.km)

    cc = np.array([xv[:3:] for xv in xxvv])
    vv = np.array([xv[3::] for xv in xxvv])
    vvkm = vv*mt.C.kpc.express(mt.C.km)

    vv_galac = np.array([vcart2vgalacticOC(cc[i],r_LSR,vvkm[i],v_LSR) for i in range(len(vvkm))])
    cc_galac = np.array([cart2galactic(c,r_LSR) for c in cc])

    global coords, vels
    coords = cc_galac
    vels = vv_galac

    random_seed = 42
    np.random.seed(random_seed)
    #nll = lambda *args: -log_likelihood(*args)
    #initial = np.array([0.,0.,0.,0.])
    #print("minimizing")
    #soln = minimize(nll, initial,method = 'Nelder-Mead', args=(cc_galac, vv_galac))
    soln = np.array([0.,0.,0.,0.])

    #pos = soln.x + 1e-4 * np.random.randn(Nwalkers, 4)
    pos = soln + 10 * (np.random.rand(Nwalkers, 4)*2.-1.)
    nwalkers, ndim = pos.shape

    with contextlib.closing(Pool()) as pool:
        sampler = mc.EnsembleSampler(Nwalkers, ndim, log_probability_global, pool=pool)
        sampler.run_mcmc(pos, Nsteps,progress=True);


    flat_samples = sampler.get_chain(discard=Nsteps/discard_N_over, flat=True)

    #mcmc = [np.percentile(flat_samples[:, i], [16, 50, 84]) for i in range(ndim)]
    #qq = [np.diff(mcmc[i]) for i in range(ndim)]
    #theta_f = [par[1] for par in mcmc]
    stds = [np.sqrt(sampler.acor[i]/(Nsteps-Nsteps/discard_N_over)/Nwalkers*np.var(flat_samples[:, i])) for i in range(ndim)]
    theta_f = [np.mean(flat_samples[:, i]) for i in range(ndim)]

    print "[A, B, C, D]:",theta_f

    info_dic = {"r_LSR":r_LSR,"Npart":len(cc), "Nsteps":Nsteps, "Nwalkers":Nwalkers,"random_seed":random_seed}

    #return cc, vv, cc_galac, vv_galac, soln, mcmc, qq, theta_f, sampler, info_dic
    return cc, vv, cc_galac, vv_galac, soln, theta_f, stds, sampler, info_dic


def fit_OC4class(xxvv,xv_LSR, Nsteps=2000, Nwalkers=50):

    r_LSR = np.array(xv_LSR[:3:])
    v_LSR = np.array(xv_LSR[3::])*mt.C.kpc.express(mt.C.km)

    cc = np.array([xv[:3:] for xv in xxvv])
    vv = np.array([xv[3::] for xv in xxvv])
    vvkm = vv*mt.C.kpc.express(mt.C.km)

    vv_galac = np.array([vcart2vgalacticOC(cc[i],r_LSR,vvkm[i],v_LSR) for i in range(len(vvkm))])
    cc_galac = np.array([cart2galactic(c,r_LSR) for c in cc])

    global coords, vels
    coords = cc_galac
    vels = vv_galac

    random_seed = 42
    np.random.seed(random_seed)
    #nll = lambda *args: -log_likelihood(*args)
    #initial = np.array([0.,0.,0.,0.])
    #print("minimizing")
    #soln = minimize(nll, initial,method = 'Nelder-Mead', args=(cc_galac, vv_galac))
    soln = np.array([0.,0.,0.,0.])

    #pos = soln.x + 1e-4 * np.random.randn(Nwalkers, 4)
    pos = soln + 10 * (np.random.rand(Nwalkers, 4)*2.-1.)
    nwalkers, ndim = pos.shape

    with contextlib.closing(Pool()) as pool:
        sampler = mc.EnsembleSampler(Nwalkers, ndim, log_probability_global, pool=pool)
        sampler.run_mcmc(pos, Nsteps,progress=True);

    tau = sampler.get_autocorr_time(tol=0)

    return cc, vv, cc_galac, vv_galac, sampler


class MCMCfit:
    def __init__(self,xxvv,xv_LSR, Nsteps=4500, Nwalkers=32):

        self.cc, self.vv, self.cc_galac, self.vv_galac, self.sampler = fit_OC4class(xxvv,xv_LSR, Nsteps, Nwalkers)
        self.Nwalkers = Nwalkers
        acort_set = False
        self.flag_conv = True
        isnotlooped = True
        while (not acort_set and Nsteps<15000) or isnotlooped:
            try:
                self.acort = self.sampler.get_autocorr_time()
                acort_set = True
            except Exception as e:
                s = str(e)
                Nstepsnew = int(50*max([float(tp) for tp in s[s.find("[")+1:s.find("]")].split()]))
                if NNstepsnew==Nsteps:
                    isnotlooped=False
                else:
                    Nsteps = Nstepsnew
                    self.cc, self.vv, self.cc_galac, self.vv_galac, self.sampler = fit_OC4class(xxvv,xv_LSR, Nsteps, Nwalkers)
        if not acort_set:
            self.acort = self.sampler.get_autocorr_time(tol=0)
        self.Nsteps = Nsteps
        burn_in = int(2*max(self.acort))
        self.burn_in = burn_in
        self.acort_std = self.sampler.get_autocorr_time(tol=0,discard=burn_in)
        if max(self.acort_std)*50 > Nsteps-burn_in:
            self.flag_conv = False
        flat_samples = self.sampler.get_chain(discard=burn_in, flat=True)
        self.std = [np.sqrt(self.acort_std[i]/(Nsteps-burn_in)/Nwalkers*np.var(flat_samples[:, i])) for i in range(4)]
        self.theta = [np.mean(flat_samples[:, i]) for i in range(4)]

        print "[A, B, C, D]:",self.theta


def fit_OC4class_ext(xxvv,xv_LSR, Nsteps=2000, Nwalkers=50):

    r_LSR = np.array(xv_LSR[:3:])
    v_LSR = np.array(xv_LSR[3::])*mt.C.kpc.express(mt.C.km)

    cc = np.array([xv[:3:] for xv in xxvv])
    vv = np.array([xv[3::] for xv in xxvv])
    vvkm = vv*mt.C.kpc.express(mt.C.km)

    vv_galac = np.array([vcart2vgalacticOC(cc[i],r_LSR,vvkm[i],v_LSR) for i in range(len(vvkm))])
    cc_galac = np.array([cart2galactic(c,r_LSR) for c in cc])

    global coords, vels
    coords = cc_galac
    vels = vv_galac

    random_seed = 42
    np.random.seed(random_seed)
    #nll = lambda *args: -log_likelihood(*args)
    #initial = np.array([0.,0.,0.,0.])
    #print("minimizing")
    #soln = minimize(nll, initial,method = 'Nelder-Mead', args=(cc_galac, vv_galac))
    soln = np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.])

    #pos = soln.x + 1e-4 * np.random.randn(Nwalkers, 4)
    pos = soln + 10 * (np.random.rand(Nwalkers, 9)*2.-1.)
    nwalkers, ndim = pos.shape

    with contextlib.closing(Pool()) as pool:
        sampler = mc.EnsembleSampler(Nwalkers, ndim, log_probability_ext_global, pool=pool)
        sampler.run_mcmc(pos, Nsteps,progress=True);

    tau = sampler.get_autocorr_time(tol=0)

    return cc, vv, cc_galac, vv_galac, sampler


class MCMCfit_ext:
    def __init__(self,xxvv,xv_LSR, Nsteps=4500, Nwalkers=32):

        self.cc, self.vv, self.cc_galac, self.vv_galac, self.sampler = fit_OC4class_ext(xxvv,xv_LSR, Nsteps, Nwalkers)
        self.Nwalkers = Nwalkers
        acort_set = False
        self.flag_conv = True
        while not acort_set and Nsteps<15000:
            try:
                self.acort = self.sampler.get_autocorr_time()
                acort_set = True
            except Exception as e:
                s = str(e)
                Nsteps = int(50*max([float(tp) for tp in s[s.find("[")+1:s.find("]")].split()]))
                self.cc, self.vv, self.cc_galac, self.vv_galac, self.sampler = fit_OC4class_ext(xxvv,xv_LSR, Nsteps, Nwalkers)
        if not acort_set:
            self.acort = self.sampler.get_autocorr_time(tol=0)
        self.Nsteps = Nsteps
        burn_in = int(2*max(self.acort))
        self.burn_in = burn_in
        self.acort_std = self.sampler.get_autocorr_time(tol=0,discard=burn_in)
        if max(self.acort_std)*50 > Nsteps-burn_in:
            self.flag_conv = False
        flat_samples = self.sampler.get_chain(discard=burn_in, flat=True)
        self.std = [np.sqrt(self.acort_std[i]/(Nsteps-burn_in)/Nwalkers*np.var(flat_samples[:, i])) for i in range(9)]
        self.theta = [np.mean(flat_samples[:, i]) for i in range(9)]

        print "[A,B,C,K,D,D',E,E',K']:",self.theta

def fit_OC4class(xxvv,xv_LSR, Nsteps=2000, Nwalkers=50):

    r_LSR = np.array(xv_LSR[:3:])
    v_LSR = np.array(xv_LSR[3::])*mt.C.kpc.express(mt.C.km)

    cc = np.array([xv[:3:] for xv in xxvv])
    vv = np.array([xv[3::] for xv in xxvv])
    vvkm = vv*mt.C.kpc.express(mt.C.km)

    vv_galac = np.array([vcart2vgalacticOC(cc[i],r_LSR,vvkm[i],v_LSR) for i in range(len(vvkm))])
    cc_galac = np.array([cart2galactic(c,r_LSR) for c in cc])

    global coords, vels
    coords = cc_galac
    vels = vv_galac

    random_seed = 42
    np.random.seed(random_seed)
    #nll = lambda *args: -log_likelihood(*args)
    #initial = np.array([0.,0.,0.,0.])
    #print("minimizing")
    #soln = minimize(nll, initial,method = 'Nelder-Mead', args=(cc_galac, vv_galac))
    soln = np.array([0.,0.,0.,0.])

    #pos = soln.x + 1e-4 * np.random.randn(Nwalkers, 4)
    pos = soln + 10 * (np.random.rand(Nwalkers, 4)*2.-1.)
    nwalkers, ndim = pos.shape

    with contextlib.closing(Pool()) as pool:
        sampler = mc.EnsembleSampler(Nwalkers, ndim, log_probability_global, pool=pool)
        sampler.run_mcmc(pos, Nsteps,progress=True);

    tau = sampler.get_autocorr_time(tol=0)

    return cc, vv, cc_galac, vv_galac, sampler


class MCMCfit:
    def __init__(self,xxvv,xv_LSR, Nsteps=4500, Nwalkers=32):

        self.cc, self.vv, self.cc_galac, self.vv_galac, self.sampler = fit_OC4class(xxvv,xv_LSR, Nsteps, Nwalkers)
        self.Nwalkers = Nwalkers
        acort_set = False
        self.flag_conv = True
        isnotlooped = True
        while (not acort_set and Nsteps<15000) or isnotlooped:
            try:
                self.acort = self.sampler.get_autocorr_time()
                acort_set = True
                isnotlooped = False
            except Exception as e:
                s = str(e)
                Nstepsnew = int(50*max([float(tp) for tp in s[s.find("[")+1:s.find("]")].split()]))
                if abs(Nstepsnew-Nsteps)/Nsteps<.05:
                    Nstepsnew = Nstepsnew*1.5
                if Nstepsnew==Nsteps:
                    isnotlooped=False
                else:
                    Nsteps = Nstepsnew
                    self.cc, self.vv, self.cc_galac, self.vv_galac, self.sampler = fit_OC4class(xxvv,xv_LSR, Nsteps, Nwalkers)
        if not acort_set:
            self.acort = self.sampler.get_autocorr_time(tol=0)
        self.Nsteps = Nsteps
        burn_in = int(2*max(self.acort))
        self.burn_in = burn_in
        self.acort_std = self.sampler.get_autocorr_time(tol=0,discard=burn_in)
        if max(self.acort_std)*50 > Nsteps-burn_in:
            self.flag_conv = False
        flat_samples = self.sampler.get_chain(discard=burn_in, flat=True)
        self.std = [np.sqrt(self.acort_std[i]/(Nsteps-burn_in)/Nwalkers*np.var(flat_samples[:, i])) for i in range(4)]
        self.theta = [np.mean(flat_samples[:, i]) for i in range(4)]

        print "[A, B, C, D]:",self.theta









############# MCMC pesado #############


def log_likelihood_extd_global(theta):
    #A,B,C,K = theta
    vreq = [vr_model_ext(c,theta) for c in coords]
    velleq = [vell_model_ext(c,theta) for c in coords]
    vbeq = [vb_model_ext(c,theta) for c in coords]
    vvr = np.array([v[0] for v in vels])
    vvell = np.array([v[1] for v in vels])
    vvb = np.array([v[2] for v in vels])
    ww = dens/sum(dens)
    return -.5*np.sum(ww*np.sqrt( (vvr - vreq)** 2 + (vvell - velleq)** 2 + (vvb - vbeq)** 2))
    #return -.5*np.sum(ww*(np.sqrt((vvr - vreq)** 2) + np.sqrt((vvell - velleq)** 2) + np.sqrt((vvb - vbeq)** 2)))

def log_probability_extd_global(theta):
    lp = log_prior_ext(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_extd_global(theta)

def fit_OC4class_extd(xxvvdd,xv_LSR, Nsteps=2000, Nwalkers=50):

    r_LSR = np.array(xv_LSR[:3:])
    v_LSR = np.array(xv_LSR[3::])*mt.C.kpc.express(mt.C.km)

    cc = np.array([xv[:3:] for xv in xxvvdd])
    vv = np.array([xv[3:6] for xv in xxvvdd])
    dd = np.array([xv[-1] for xv in xxvvdd])
    vvkm = vv*mt.C.kpc.express(mt.C.km)

    vv_galac = np.array([vcart2vgalacticOC(cc[i],r_LSR,vvkm[i],v_LSR) for i in range(len(vvkm))])
    cc_galac = np.array([cart2galactic(c,r_LSR) for c in cc])

    global coords, vels, dens
    coords = cc_galac
    vels = vv_galac
    dens = dd

    random_seed = 42
    np.random.seed(random_seed)
    #nll = lambda *args: -log_likelihood(*args)
    #initial = np.array([0.,0.,0.,0.])
    #print("minimizing")
    #soln = minimize(nll, initial,method = 'Nelder-Mead', args=(cc_galac, vv_galac))
    soln = np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.])

    #pos = soln.x + 1e-4 * np.random.randn(Nwalkers, 4)
    pos = soln + 10 * (np.random.rand(Nwalkers, 9)*2.-1.)
    nwalkers, ndim = pos.shape

    with contextlib.closing(Pool()) as pool:
        sampler = mc.EnsembleSampler(Nwalkers, ndim, log_probability_extd_global, pool=pool)
        sampler.run_mcmc(pos, Nsteps,progress=True);

    tau = sampler.get_autocorr_time(tol=0)

    return dd, cc, vv, cc_galac, vv_galac, sampler


class MCMCfit_extd:
    def __init__(self,xxvvdd,xv_LSR, Nsteps=4500, Nwalkers=32):

        self.dd, self.cc, self.vv, self.cc_galac, self.vv_galac, self.sampler = fit_OC4class_extd(xxvvdd,xv_LSR, Nsteps, Nwalkers)
        self.Nwalkers = Nwalkers
        acort_set = False
        self.flag_conv = True

        while not acort_set and Nsteps<15000:
            try:
                self.acort = self.sampler.get_autocorr_time()
                acort_set = True
            except Exception as e:
                s = str(e)
                if (50.*max([float(tp) for tp in s[s.find("[")+1:s.find("]")].split()]) - Nsteps)/Nsteps<.1:
                    break
                else:
                    Nsteps = int(50*max([float(tp) for tp in s[s.find("[")+1:s.find("]")].split()]))
                    self.dd, self.cc, self.vv, self.cc_galac, self.vv_galac, self.sampler = fit_OC4class_extd(xxvvdd,xv_LSR, Nsteps, Nwalkers)

        if not acort_set:
            self.acort = self.sampler.get_autocorr_time(tol=0)
        self.Nsteps = Nsteps
        burn_in = int(2*max(self.acort))
        self.burn_in = burn_in
        self.acort_std = self.sampler.get_autocorr_time(tol=0,discard=burn_in)
        if max(self.acort_std)*50 > Nsteps-burn_in:
            self.flag_conv = False
        flat_samples = self.sampler.get_chain(discard=burn_in, flat=True)
        self.std = [np.sqrt(self.acort_std[i]/(Nsteps-burn_in)/Nwalkers*np.var(flat_samples[:, i])) for i in range(9)]
        self.theta = [np.mean(flat_samples[:, i]) for i in range(9)]

        print "[A,B,C,K,D,D',E,E',K']:",self.theta

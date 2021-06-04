import sys
import os
import glob as gb
sys.path.append('/home/cbenitez/')
import MTmp as mt
import matplotlib.pyplot as plt
import numpy as np
import oort_cons as oc

h=1.
def RK4(xxvv,time_step=h):

    k_1 = np.array(map(lambda xv: np.append(xv[3::],mt.force_T(xv[0],xv[1],xv[2])),xxvv))
    xxvv_1 = xxvv+k_1*h/2

    k_2 = np.array(map(lambda xv: np.append(xv[3::],mt.force_T(xv[0],xv[1],xv[2])),xxvv_1))
    xxvv_2 = xxvv+k_2*h/2

    k_3 = np.array(map(lambda xv: np.append(xv[3::],mt.force_T(xv[0],xv[1],xv[2])),xxvv_2))
    xxvv_3 = xxvv+k_3*h

    k_4 = np.array(map(lambda xv: np.append(xv[3::],mt.force_T(xv[0],xv[1],xv[2])),xxvv_3))

    xxvv = xxvv + h/6.*(k_1+2.*k_2+2.*k_3+k_4)

    return xxvv

R0 = 8.0
rc = .2

def varOort(vv_adi,N=120):

    LSRs =  [np.append(np.array([R0,0.,0.]),mt.vel_circ(np.array([R0,0.,0.]),mt.force_T))]
    NT = 100
    global h
    h = 0.0199009793340327*2.65860467535e+15/NT
    on= 10
    for i in range(on):
        rv_LSRn = LSRs[-1]
        with mt.contextlib.closing(mt.Pool()) as pool:
                    for nt in range(NT):
                        rv_LSRn = RK4([rv_LSRn])[0]
        LSRs.append(rv_LSRn)

    rrr = []

    for v_adi in vv_adi:
        print v_adi, " completed..."
        ccgalac = np.array([[1. for i in range(N)],np.linspace(0.,1.-1./N,N),np.zeros(N)]).T*np.array([rc,2*np.pi,0.]) #espaciadas uniforme
        cc = [oc.cgalac2ccart(c,LSRs[0][:3:]) for c in ccgalac]
        vv_base = [mt.vel_circ(c,mt.force_T) for i in range(N)]
        vvgalac = [oc.vcart2vgalactic(cc[i],LSRs[0][:3:],vv_base[i],LSRs[0][3::])+v_adi*mt.C.km.express(mt.C.kpc) for i in range(N)]
        vv = [oc.vgalac2vcart(ccgalac[i],LSRs[0][:3:],vvgalac[i],LSRs[0][3::]) for i in range(N)]

        xxvv = np.array([np.append(cc[i],vv[i]) for i in range(N)])
        res = [xxvv]
        for i in range(on):
            xxvv_n = res[-1]
            with mt.contextlib.closing(mt.Pool()) as pool:
                        for nt in range(NT):
                            xxvv_n = list(pool.map(RK4,[xxvv_n])[0])
            res.append(xxvv_n)
        rrr.append(res)
    return LSRs,rrr


vv_exp = [-15,-10,-5,-1,0,1,5,10,15]
vv_rot = [-15,-10,-5,-1,0,1,5,10,15]
vv_adi = [np.array([v_exp,v_rot,0.]) for v_exp in vv_exp for v_rot in vv_rot]

evols = varOort(vv_adi)
np.save('/fs/nas14/other0/cbenitez/evolsOC/evols_exp_central',evols)
np.save('/fs/nas14/other0/cbenitez/evolsOC/evols_exp_central_vels_adi',vv_adi)

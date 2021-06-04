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
A,B = 13.55037486116183, -13.938877070529985
oo = [A,B,0.,0.]

def varOort(rr,N=120):

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

    rrrOC = []
    rrrAS = []

    for rc in rr:
        print rc, " completed..."
        ccgalac = np.array([[1. for i in range(N)],np.linspace(0.,1.-1./N,N),np.zeros(N)]).T*np.array([rc,2*np.pi,0.]) #espaciadas uniforme
        cc = [oc.cgalac2ccart(c,LSRs[0][:3:]) for c in ccgalac]
        vvOCr = np.array([oc.vr_model(c,np.array(oo))*mt.C.km.express(mt.C.kpc) for c in ccgalac])
        vvOCl = np.array([oc.vell_model(c,np.array(oo))*mt.C.km.express(mt.C.kpc) for c in ccgalac])
        vvOCb = np.array([oc.vb_model(c,np.array(oo))*mt.C.km.express(mt.C.kpc) for c in ccgalac])
        vvOCgalac = np.array([[vvOCr[i],vvOCl[i],vvOCb[i]] for i in range(N)])
        vvOC = [oc.vgalacOC2vcart(ccgalac[i],LSRs[0][:3:],vvOCgalac[i],LSRs[0][3::]) for i in range(N)]
        vvAS = [mt.vel_circ(c,mt.force_T) for c in cc]

        xxvvOC = np.array([np.append(cc[i],vvOC[i]) for i in range(N)])
        xxvvAS = np.array([np.append(cc[i],vvAS[i]) for i in range(N)])
        resOC = [xxvvOC]
        resAS = [xxvvAS]
        for i in range(on):
            xxvvOC_n = resOC[-1]
            xxvvAS_n = resAS[-1]
            with mt.contextlib.closing(mt.Pool()) as pool:
                        for nt in range(NT):
                            xxvvOC_n = list(pool.map(RK4,[xxvvOC_n])[0])
                            xxvvAS_n = list(pool.map(RK4,[xxvvAS_n])[0])
            resOC.append(xxvvOC_n)
            resAS.append(xxvvAS_n)
        rrrOC.append(resOC)
        rrrAS.append(resAS)
    return LSRs,rrrOC,rrrAS

rr_checked = np.array([.1,.2,.3,.4,.5,1.])
#rr_checked = np.array([.1])
evols_r = varOort(rr_checked)
np.save('/fs/nas14/other0/cbenitez/evolsOC/evols_r_axisim_rp1p2p3p4p5p10',evols_r)

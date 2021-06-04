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
rc = .4
ocas = np.array([13.55037486116183, -13.938877070529985,0.,0.])

def varOort(vvcc,N=120,):

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

    for vve in vvcc:
        print vve, " completed..."
        ccgalac = np.array([[1. for i in range(N)],np.linspace(0.,1.-1./N,N),np.zeros(N)]).T*np.array([rc,2*np.pi,0.]) #espaciadas uniforme
        cc = [oc.cgalac2ccart(c,LSRs[0][:3:]) for c in ccgalac]
        vvr = np.array([(oc.vr_model(c,ocas)+vve[0])*mt.C.km.express(mt.C.kpc) for c in ccgalac])
        vvl = np.array([(oc.vell_model(c,ocas+vve[1]))*mt.C.km.express(mt.C.kpc) for c in ccgalac])
        vvb = np.array([oc.vb_model(c,ocas)*mt.C.km.express(mt.C.kpc) for c in ccgalac])
        vvgalac = np.array([[vvr[i],vvl[i],vvb[i]] for i in range(N)])
        vvOC = [oc.vgalacOC2vcart(ccgalac[i],LSRs[0][:3:],vvgalac[i],LSRs[0][3::]) for i in range(N)]

        xxvvOC = np.array([np.append(cc[i],vvOC[i]) for i in range(N)])
        res = [xxvvOC]
        for i in range(on):
            xxvvOC_n = res[-1]
            with mt.contextlib.closing(mt.Pool()) as pool:
                        for nt in range(NT):
                            xxvvOC_n = list(pool.map(RK4,[xxvvOC_n])[0])
            res.append(xxvvOC_n)
        rrr.append(res)
    return LSRs,rrr


EE = [np.round(3*i,decimals=1) for i in np.linspace(-2,2,9)]
RR = [np.round(3*i,decimals=1) for i in np.linspace(-2,2,9)]
er_checked = [[einE,rinR] for einE in EE for rinR in RR]


evols_r = varOort(er_checked)
np.save('/fs/nas14/other0/cbenitez/evolsOC/evols_era_go',evols_r)
np.save('/fs/nas14/other0/cbenitez/evolsOC/evols_era_cons_go',er_checked)

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

def varOort(oocc,N=120,):

    LSRs =  [np.append(np.array([R0,0.,0.]),mt.vel_circ(np.array([R0,0.,0.]),mt.force_T))]
    tt_go = [0]
    NT = 100
    global h
    h = 0.0199009793340327*2.65860467535e+15/NT
    on= 10
    for i in range(on):
        rv_LSRn = LSRs[-1]
        with mt.contextlib.closing(mt.Pool()) as pool:
                    for nt in range(NT):
                        rv_LSRn = RK4([rv_LSRn])[0]
                        tt_go.append(tt_go[-1]+h)
        LSRs.append(rv_LSRn)

    rrrOC = []

    for oo in oocc:
        #print oo, " completed..."
        ccgalac = np.array([[1. for i in range(N)],np.linspace(0.,1.-1./N,N),np.zeros(N)]).T*np.array([rc,2*np.pi,0.]) #espaciadas uniforme
        cc = [oc.cgalac2ccart(c,LSRs[0][:3:]) for c in ccgalac]
        vvOCr = np.array([oc.vr_model(c,np.array(oo))*mt.C.km.express(mt.C.kpc) for c in ccgalac])
        vvOCl = np.array([oc.vell_model(c,np.array(oo))*mt.C.km.express(mt.C.kpc) for c in ccgalac])
        vvOCb = np.array([oc.vb_model(c,np.array(oo))*mt.C.km.express(mt.C.kpc) for c in ccgalac])
        vvOCgalac = np.array([[vvOCr[i],vvOCl[i],vvOCb[i]] for i in range(N)])
        vvOC = [oc.vgalac2vcart(ccgalac[i],LSRs[0][:3:],vvOCgalac[i],LSRs[0][3::]) for i in range(N)]

        xxvvOC = np.array([np.append(cc[i],vvOC[i]) for i in range(N)])
        resOC = [xxvvOC]
        for i in range(on):
            xxvvOC_n = resOC[-1]
            with mt.contextlib.closing(mt.Pool()) as pool:
                        for nt in range(NT):
                            xxvvOC_n = list(pool.map(RK4,[xxvvOC_n])[0])
            resOC.append(xxvvOC_n)
        rrrOC.append(resOC)
    return LSRs,rrrOC,tt_go


A,B = -13.55037486116183, 13.938877070529985
AA = [A*i for i in np.linspace(-1.5,1.5,7)]
BB = [B*i for i in np.linspace(-1.5,1.5,7)]
CC = np.linspace(-11.25,11.25,7)
KK = np.linspace(-11.25,11.25,7)
oocc_checked = []
for i in range(6):
    if i == 0:
        oocc_checked.extend([[ainA,binB,0.,0.] for ainA in AA for binB in BB])
    if i == 1:
        oocc_checked.extend([[ainA,B,cinC,0.] for ainA in AA for cinC in CC])
    if i == 2:
        oocc_checked.extend([[ainA,B,0.,kinK] for ainA in AA for kinK in KK])
    if i == 3:
        oocc_checked.extend([[A,binB,cinC,0.] for binB in BB for cinC in CC])
    if i == 4:
        oocc_checked.extend([[A,binB,0.,kinK] for binB in BB for kinK in KK])
    if i == 5:
        oocc_checked.extend([[A,B,cinC,kinK] for cinC in CC for kinK in KK])

evols_r = varOort(oocc_checked)
np.save('evols_oc_var2_go',evols_r)
np.save('evols_oc_var2_cons_go',oocc_checked)

import sys
import os
import glob as gb
sys.path.append('/home/cbenitez/')
import MTmp as mt
import matplotlib.pyplot as plt
import numpy as np
import oort_cons as oc

varOC_dir = "/fs/nas14/other0/cbenitez/evolsOC/"

exp3 = [np.load(varOC_dir+"evols_oc_var2_cons_go.npy"),np.load(varOC_dir+"evols_oc_var2_go.npy")]

oocc,lsrs,parts = exp3[0],exp3[1][0],exp3[1][1]

for ioc in range(81+8,81*2):
    occ = oocc[ioc]
    oocct = [oc.MCMCfit(parts[ioc][ic],lsrs[ic],Nsteps=5000) for ic in range(11)]
    np.save('/fs/nas14/other0/cbenitez/evolsOC/MCMCruns/oocct_AvsC_'+\
            str(np.round(occ[0],decimals=2))+"x"+\
            str(np.round(occ[2],decimals=2)),oocct)

for ioc in range(81*2,81*3):
    occ = oocc[ioc]
    oocct = [oc.MCMCfit(parts[ioc][ic],lsrs[ic],Nsteps=5000) for ic in range(11)]
    np.save('/fs/nas14/other0/cbenitez/evolsOC/MCMCruns/oocct_AvsK_'+\
            str(np.round(occ[0],decimals=2))+"x"+\
            str(np.round(occ[3],decimals=2)),oocct)

for ioc in range(81*3,81*4):
    occ = oocc[ioc]
    oocct = [oc.MCMCfit(parts[ioc][ic],lsrs[ic],Nsteps=5000) for ic in range(11)]
    np.save('/fs/nas14/other0/cbenitez/evolsOC/MCMCruns/oocct_BvsC_'+\
            str(np.round(occ[1],decimals=2))+"x"+\
            str(np.round(occ[2],decimals=2)),oocct)

for ioc in range(81*4,81*5):
    occ = oocc[ioc]
    oocct = [oc.MCMCfit(parts[ioc][ic],lsrs[ic],Nsteps=5000) for ic in range(11)]
    np.save('/fs/nas14/other0/cbenitez/evolsOC/MCMCruns/oocct_BvsK_'+\
            str(np.round(occ[1],decimals=2))+"x"+\
            str(np.round(occ[3],decimals=2)),oocct)

for ioc in range(81*5,81*6):
    occ = oocc[ioc]
    oocct = [oc.MCMCfit(parts[ioc][ic],lsrs[ic],Nsteps=5000) for ic in range(11)]
    np.save('/fs/nas14/other0/cbenitez/evolsOC/MCMCruns/oocct_CvsK_'+\
            str(np.round(occ[2],decimals=2))+"x"+\
            str(np.round(occ[3],decimals=2)),oocct)

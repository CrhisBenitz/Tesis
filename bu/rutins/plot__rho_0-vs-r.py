
import sys

import matplotlib.pyplot as plt
import numpy as np
import MT as mt

import jtplot
jtplot.style(grid=False)

where2save = str(sys.argv[1])


def rho_0(r):
    if r<=3:
        return np.exp(8/8.5)*np.exp(-3/8.5)
    return np.exp(8/8.5)*np.exp(-r/8.5)

def g_z(r,z):
    return mt.force_T(r,0,z)[-1]

def P_integrated(r,z):
    s = np.sign(z)
    z = abs(z)
    zz = np.linspace(0,z,int(np.floor(z/.001)))
    return s*sum(g_z(r,s*zz[i])*(zz[i]-zz[i-1]) for i in range(1,len(zz)))

def rho(r,z):
    return rho_0(r)*np.exp(P_integrated(r,z)/cs2)

rr = np.linspace(.0000,15,1000)

plt.ion()
plt.figure(figsize=(12,8))
plt.plot(rr,[rho(0,z) for r in rr],c='k',ls='--',label='$r=3$ kpc')
plt.plot(rr,[rho(.1,z) for r in rr],c='k',label='$r=8$ kpc')
plt.plot(rr,[rho(.3,z) for r in rr],c='k',ls=':',label='$r=10$ kpc')
plt.xlabel(r'$z$   [kpc]',fontsize=28)
plt.ylabel(r'$\rho(r,z)$   [cm$^{-3}$]',fontsize=28)
plt.legend(loc='best',fontsize=22)
plt.locator_params(nbins=6)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.ylim(1e-10,10)
plt.yscale('log')
plt.show()

if raw_input("Save image? (will be saved to %s)"%where2save) in ['yes','y',"Yes","YES"]:
  plt.savefig(where2save+".pdf", format='pdf')


import sys

import matplotlib.pyplot as plt
import numpy as np
import MT as mt

import jtplot
jtplot.style(grid=False)

where2save = str(sys.argv[1])

rr = np.linspace(0.000001,100,5000)

ff1 = [np.linalg.norm(mt.force_1(r,0,0)) for r in rr]
ff2 = [np.linalg.norm(mt.force_2(r,0,0)) for r in rr]
ff3 = [np.linalg.norm(mt.force_3(r,0,0)) for r in rr]

vv1 = np.array([np.linalg.norm(mt.vel_circ([r,0.,0.],mt.force_1)) for r in rr])
vv2 = np.array([np.linalg.norm(mt.vel_circ([r,0.,0.],mt.force_2)) for r in rr])
vv3 = np.array([np.linalg.norm(mt.vel_circ([r,0.,0.],mt.force_3)) for r in rr])
vv = np.array([np.linalg.norm(mt.vel_circ([r,0.,0.],mt.force_T)) for r in rr])

plt.ion()
plt.figure(figsize=(12,8))
plt.plot(rr,vv1*mt.C.kpc.express(mt.C.km),lw=2,label='Masa central')
plt.plot(rr,vv2*mt.C.kpc.express(mt.C.km),ls='--',lw=2,label='Disco')
plt.plot(rr,vv3*mt.C.kpc.express(mt.C.km),ls=':',lw=2,label='Halo')
plt.xlabel(r'$r$   [kpc]',fontsize=28)
plt.ylabel(r'$v_{circ}$   [kms$^{-1}$]',fontsize=28)
plt.legend(loc='best',fontsize=22)
plt.locator_params(nbins=6)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.show()

if raw_input("Save image? (will be saved to %s)"%where2save) in ['yes','y',"Yes","YES"]:
  plt.savefig(where2save+".pdf", format='pdf')

import MTmp as mt
import numpy as np
from shutil import copy as shcp
from datetime import datetime

#mt.basedir = ""
path_file = "/fs/nas14/other0/cbenitez/partsSim/"
name_file = "mp_marbles_"+datetime.now().strftime("%d%m%Y-%H%M%S")

this_file = mt.os.path.basename(__file__)
shcp(this_file,path_file+name_file+"_script.py")

def dens_colum(dd,zlim_index,z_central = None):
    if z_central ==None:
        z_central = dd.shape[0]/2
    zz = range(z_central-zlim_index,z_central+zlim_index+1)
    return sum(dd[z,:,:] for z in zz)/len(zz)

def cloud_center_at(op):
    yx = np.where(dens_colum(op.dd,zlim_index=80)==np.max(dens_colum(op.dd,zlim_index=80)))
    return np.array([np.mean(yx[0]),np.mean(yx[1])])

def best_sun_position(op):
    center_at = cloud_center_at(op)
    center_at = [int(c) for c in center_at]
    return np.sqrt(op.xx[center_at[1]]**2+op.yy[center_at[0]]**2),np.arctan2(op.yy[center_at[0]],op.xx[center_at[1]])

op_0 = 9
op_f = 16
op_c = 9
limits=(0.6,.85,0.375,0.625,0.375,0.625)

#nop_0 = mt.pymses.RamsesOutput("%s" % mt.basedir, op_0, verbose=False)
#nop_c = mt.pymses.RamsesOutput("%s" % mt.basedir, op_c, verbose=False)

#sun_position = best_sun_position(mt.rop(op=op_0,limits=limits))
#v_LSR = mt.vel_circ([8.,0.,0.],mt.force_T)

#time2colision = nop_c.info['time']*nop_c.info['unit_time'].express(mt.C.s) - nop_0.info['time']*nop_0.info['unit_time'].express(mt.C.s)
#theta_0 = sun_position[1] + np.linalg.norm(v_LSR)/8.*time2colision

NT = 200
#r_LSR = np.array([np.cos(theta_0),np.sin(theta_0),0.0])*8.
r_LSR = np.array([6.927001304913123, 0.8250001554140809,0.0])
v_LSR = mt.vel_circ(r_LSR,mt.force_T)
rv_LSR = np.append(r_LSR,v_LSR)
rho_lim = 2.5
box_dims = (2.,2.,.5)

print("Empezando")

mt.evolution(op_0,op_f,limits, NT, rv_LSR, rho_lim, box_dims, path_file, name_file)

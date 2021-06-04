import sys
import os
import glob as gb

sys.path.append('/home/cbenitez/pymses/pymses_4.0.0')
sys.path.append("/fs/nas14/other0/cbenitez/Pkgs/J/jupyterthemes-0.19.6/jupyterthemes")

import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from matplotlib.colors import LogNorm

from matplotlib import rc #tamano figuras
import matplotlib as mpl

plot_size = (20,20)
mpl.rcParams['figure.figsize'] = plot_size

import matplotlib.patches as patches

from mpl_toolkits.mplot3d import Axes3D

import jtplot

jtplot.style(grid = False)

from pandas import *

from operator import gt,lt,le,ge,eq,ne

import pymses
from pymses.analysis import sample_points
from pymses.analysis.visualization import *
from pymses import RamsesOutput
from pymses.utils import constants as C

import warnings
warnings.simplefilter("ignore")


#basedir = "/fs/nas14/other0/gilgo/gould_belt/patch/vertical/"
#basedir = "/fs/calzonzin/Part1/gilgo/gould_belt/patch/vertical/"
basedir = "/fs/nas14/other0/gilgo/gould_belt/patch_geom_amr/vertical/"
tardir = "/fs/nas14/other0/cbenitez/Tesis/B/"

def savefig(name):
    plt.savefig(tardir+name+'.pdf', format='pdf')

def plot(xx,yy,fig=None,figsize= (8,8),title='',label='',xlab='',ylab='',zlab='',xlims=None,ylims=None,color='k'):

    plt.ion()

    if fig!=None:
        fig=plt.figure(figsize = figsize)

    plt.plot(xx,yy,label=label,c=color)
    plt.title(title,fontsize=16)
    plt.xlabel(xlab,fontsize=15)
    plt.ylabel(ylab,fontsize=15)
    if xlims!=None:
        plt.xlim(xlims[0],xlims[1])
    if ylims!=None:
        plt.ylim(ylims[0],ylims[1])
    plt.title('')
    plt.tight_layout()

def scatter(xx,yy,fig=None,figsize = (8,8),title='',label='',xlab='',ylab='',zlab='',xlims=None,ylims=None,color='k',size=2):

    plt.ion()

    if fig!=None:
        fig=plt.figure(figsize = figsize)

    plt.scatter(xx,yy,label=label,c=color,s=size)
    plt.title(title,fontsize=16)
    plt.xlabel(xlab,fontsize=15)
    plt.ylabel(ylab,fontsize=15)
    if xlims!=None:
        plt.xlim(xlims[0],xlims[1])
    if ylims!=None:
        plt.ylim(ylims[0],ylims[1])
    plt.title
    plt.tight_layout()

def imshow(zz,fig=None,figsize = (5,5),xlab='',ylab='',zlab='',title='',label='',xlims=None,ylims=None,zlims=(None,None),colorscale=None,extentto=None,shrink=.75,colorbar = False,logscale=False):

    plt.ion()

    if fig==None:
        fig=plt.figure(figsize = figsize)

    if logscale:

            plt.imshow(zz,extent=extentto,cmap=colorscale,label=label,vmin=zlims[0], vmax=zlims[1],origin='lower',norm=LogNorm())

    else:

        plt.imshow(zz,extent=extentto,cmap=colorscale,label=label,vmin=zlims[0], vmax=zlims[1],origin='lower')

    plt.title(title,fontsize=16)
    plt.xlabel(xlab,fontsize=15)
    plt.ylabel(ylab,fontsize=15)

    if colorbar == True:

        cbar = plt.colorbar(shrink = shrink)
        cbar.set_label(zlab, rotation=90,fontsize=15)

    if xlims!=None:
        plt.xlim(xlims[0],xlims[1])
    if ylims!=None:
        plt.ylim(ylims[0],ylims[1])
    plt.title('')
    plt.tight_layout()

class rop:

    def __init__(self,limits=None,op=None):


        if op ==None:
            self.op = input("Number of the RAMSES output: \n")
        else:
            self.op = op

        output = pymses.RamsesOutput("%s" % basedir, self.op, verbose=False)
        self.N_max = 2**(output.info['levelmax'])

        if limits == None:

            self.limits = input("Limits of the normalized box to compute (ax,bx,ay,by,az,bz):\n")
            self.nx = int(self.N_max*(self.limits[1]-self.limits[0]))
            self.ny = int(self.N_max*(self.limits[3]-self.limits[2]))
            self.nz = int(self.N_max*(self.limits[5]-self.limits[4]))

        else:

            self.nx = int(self.N_max*(limits[1]-limits[0]))
            self.ny = int(self.N_max*(limits[3]-limits[2]))
            self.nz = int(self.N_max*(limits[5]-limits[4]))
            self.limits = limits

        self.deltas = 1./self.N_max/2

        z, y, x = np.mgrid[self.limits[4]+self.deltas:self.limits[5]-self.deltas:self.nz*1j, self.limits[2]+self.deltas:self.limits[3]-self.deltas:self.ny*1j, self.limits[0]+self.deltas:self.limits[1]-self.deltas:self.nx*1j] #Grids en donde se copiaran los datos
        npoints = np.prod(x.shape) #Dar la estructura correcta
        x1 = np.reshape(x, npoints)
        y1 = np.reshape(y, npoints)
        z1 = np.reshape(z, npoints)

        pxyz_op = np.array([x1, y1, z1]).T #arreglo con los datos de l output de RAMSES

        source = output.amr_source(["rho", "vel", "P"])
        dset = pymses.analysis.sample_points(source, pxyz_op, use_C_code=True) # Sample Hydro Fields

        print("\n Succesfully read the output file %g. \n"%self.op)

        self.vel_unit = output.info["unit_velocity"].express(C.km/C.s)
        self.dist_unit = output.info["unit_length"].express(C.kpc)
        self.dens_unit = output.info["unit_density"].express(C.H_cc)
        self.time = (output.info["unit_time"]*output.info["time"]).express(C.s)
        next_output = pymses.RamsesOutput("%s" % basedir, self.op+1, verbose=False)
        self.deltat_forward = (next_output.info["unit_time"]*next_output.info["time"]).express(C.s) - self.time
        self.info = output.info

        self.dd = np.reshape(dset["rho"], (self.nz,self.ny,self.nx)) * self.dens_unit
        self.vv_x = np.reshape(dset["vel"][:,0], (self.nz,self.ny,self.nx)) * self.vel_unit
        self.vv_y = np.reshape(dset["vel"][:,1], (self.nz,self.ny,self.nx)) * self.vel_unit
        self.vv_z = np.reshape(dset["vel"][:,2], (self.nz,self.ny,self.nx)) * self.vel_unit

        self.xx = x[0,0,:] * self.dist_unit - self.dist_unit/2
        self.yy = y[0,:,0] * self.dist_unit - self.dist_unit/2
        self.zz = z[:,0,0] * self.dist_unit - self.dist_unit/2

        self.x_max = max(self.xx)
        self.y_max = max(self.yy)
        self.z_max = max(self.zz)
        self.x_min = min(self.xx)
        self.y_min = min(self.yy)
        self.z_min = min(self.zz)


#        self.dx = 2.*self.x_max/(self.N_max-1)
#        self.dy = 2.*self.y_max/(self.N_max-1)
#        self.dz = 2.*self.z_max/(self.N_max-1)

#        self.r_max = max(self.x_max,self.y_max) - 2*max(self.dx,self.dy,self.dz)

#####################################################################
#####################################################################
# Funciones
#####################################################################
#####################################################################


#####################################################################
# Funciones del potencial (Allen-Santillan)
#####################################################################


def potential_1(x,y,z):
    M_1 = 606.0
    b_1 = .3873
    phi_1 = - M_1/np.sqrt(x**2 + y**2 + z**2 + b_1**2)
    return phi_1

def potential_2(x,y,z):
    M_2 = 3690.0
    b_2 = .25
    a_2 = 5.3178
    phi_2 = - M_2/np.sqrt(x**2 + y**2 + (a_2 + np.sqrt(z**2 + b_2**2) )**2)
    return phi_2

def potential_3(x,y,z):
    M_3 = 4615.0
    a_3 = 12.0
    phi_3 = - (M_3*(np.sqrt(x**2+y**2+z**2)/a_3)**2.02/(1.+(np.sqrt(x**2+y**2+z**2)/a_3)**1.02))/np.sqrt(x**2+y**2+z**2) -(M_3/1.02/a_3)*(-1.02/(1.+(100./a_3)**1.02) + np.log(1.+(100./a_3)**1.02) - (-1.02/(1.+(np.sqrt(x**2+y**2+z**2)/a_3)**1.02) + np.log(1.+(np.sqrt(x**2+y**2+z**2)/a_3)**1.02)))
    return phi_3

def potential_T(x,y,z):
    return potential_1(x,y,z)+potential_2(x,y,z)+potential_3(x,y,z)

def force_1(x,y,z):
    M_1 = 606.0
    b_1 = .3873
    F_1 = -M_1/np.sqrt(x**2 + y**2 + z**2 + b_1**2)**3*np.array([x,y,z])
    return F_1*(3.2408e-17)**2*100

def force_2(x,y,z):
    M_2 = 3690.0
    b_2 = .25
    a_2 = 5.3178
    F_2 = -M_2/np.sqrt(x**2 + y**2 + (a_2 + np.sqrt(z**2 + b_2**2) )**2)**3*np.array([x,y,z*(a_2 + np.sqrt(z**2+b_2**2))/np.sqrt(z**2+b_2**2)])
    return F_2*(3.2408e-17)**2*100

def force_3(x,y,z):
    M_3 = 4615.0
    a_3 = 12.0
    F_3 = -M_3*(2.02*np.sqrt(x**2+y**2+z**2)*(np.sqrt(x**2+y**2+z**2)/a_3)**1.02 + np.sqrt(x**2+y**2+z**2)*(np.sqrt(x**2+y**2+z**2)/a_3)**2.04 - 1.02*a_3*(np.sqrt(x**2+y**2+z**2)/a_3)**2.02)/((x**2+y**2+z**2)*a_3*(2.*(np.sqrt(x**2+y**2+z**2)/a_3)**1.02 + (np.sqrt(x**2+y**2+z**2)/a_3)**2.04 + 1 ))*np.array([x,y,z])/np.sqrt(x**2+y**2+z**2)
    return F_3*(3.2408e-17)**2*100

def force_T(x,y,z):
    return force_1(x,y,z) + force_2(x,y,z) + force_3(x,y,z)


#####################################################################
# Runge-Kutta 4
#####################################################################

def RK4(xx,vv,h,fuerza = force_T):

    vv = vv

    kx_1=vv
    xx_1=xx+kx_1*h/2

    kv_1 = fuerza(xx[0],xx[1],xx[2])
    vv_1 = vv+kv_1*h/2

    kx_2 = vv+kv_1*h/2
    xx_2 = xx+kx_2*h/2

    kv_2 = fuerza(xx_1[0],xx_1[1],xx_1[2])
    vv_2 = vv+kv_2*h/2

    kx_3 = vv+kv_2*h/2
    xx_3 = xx+kx_3*h

    kv_3 = fuerza(xx_2[0],xx_2[1],xx_2[2])
    vv_3 = vv+kv_3*h

    kx_4 = vv+kv_3*h
    kv_4 = fuerza(xx_3[0],xx_3[1],xx_3[2])

    xx=xx+h/6.*(kx_1+2.*kx_2+2.*kx_3+kx_4)
    vv=vv+h/6.*(kv_1+2.*kv_2+2.*kv_3+kv_4)

    return np.array(xx),np.array(vv)


def evolution(op_0,op_f,limits, NT, r_LSR, v_LSR, rho_lim, box_dims, path_file, name_file):

    name_file_LSR = name_file+'LSR'

    while (os.path.exists(path_file+name_file) or os.path.exists(path_file+name_file_LSR)):
        if not os.path.exists(path_file+name_file):
            auxt = name_file_LSR + ' in ' + path_file
        elif not os.path.exists(path_file+name_file_LSR):
            auxt = name_file + ' in ' + path_file
        else:
            auxt = name_file + ' and ' + name_file_LSR + ' in ' + path_file

        if 'yes' == raw_input("Overwrite existent %s file: (yes or no)\n"%auxt):
            break
        if os.path.exists(path_file+name_file):
            name_file = raw_input("Name of the .dat file to store the marbles data: \n")
        if os.path.exists(path_file+name_file_LSR):
            name_file_LSR = raw_input("Name of the .dat file to store LSR data: \n")

    sf = rop(limits,op_0)
    deltatT = sf.deltat_forward
    h = deltatT/NT
    r_LSR_f, v_LSR_f = r_LSR, v_LSR

    xx_RK4 = []
    vv_RK4 = []
    dd_RK4 = []
    creation_times = []
    creation_time = 0
    ids = []

    dens_th = rho_lim
    box_w = box_dims

    where = np.array([np.where(abs(r_LSR_f[0]-sf.xx)<box_w[0])[0],np.where(abs(r_LSR_f[1]-sf.yy)<box_w[1])[0],np.where(abs(r_LSR_f[2]-sf.zz)<box_w[2])[0]])

    dd_box = np.array([[[sf.dd[k,j,i] for i in where[0]] for j in where[1]] for k in where[2]])
    zz_box = sf.zz[where[2]]
    yy_box = sf.yy[where[1]]
    xx_box = sf.xx[where[0]]

    vvx_box = np.array([[[sf.vv_x[k,j,i] for i in where[0]] for j in where[1]] for k in where[2]])*C.km.express(C.kpc)
    vvy_box = np.array([[[sf.vv_y[k,j,i] for i in where[0]] for j in where[1]] for k in where[2]])*C.km.express(C.kpc)
    vvz_box = np.array([[[sf.vv_z[k,j,i] for i in where[0]] for j in where[1]] for k in where[2]])*C.km.express(C.kpc)

    where_picos = np.where(dd_box>dens_th)

    dd_picos = [dd_box[where_picos[0][i],where_picos[1][i],where_picos[2][i]] for i in range(len(where_picos[0]))]
    zz_picos = zz_box[where_picos[0]]
    yy_picos = yy_box[where_picos[1]]
    xx_picos = xx_box[where_picos[2]]

    vvx_picos = [vvx_box[where_picos[0][i],where_picos[1][i],where_picos[2][i]] for i in range(len(where_picos[0]))]
    vvy_picos = [vvy_box[where_picos[0][i],where_picos[1][i],where_picos[2][i]] for i in range(len(where_picos[0]))]
    vvz_picos = [vvz_box[where_picos[0][i],where_picos[1][i],where_picos[2][i]] for i in range(len(where_picos[0]))]

    xx_RK4.extend([np.array([xx_picos[i],yy_picos[i],zz_picos[i]]) for i in range(len(where_picos[0]))])
    vv_RK4.extend([np.array([vvx_picos[i],vvy_picos[i],vvz_picos[i]]) for i in range(len(where_picos[0]))])

    #dd_RK4.extend(dd_RK4)
    dd_RK4.extend(dd_picos)


    creation_times.extend([creation_time for i in range(len(where_picos[0]))])
    ids = range(0,len(where_picos[0])+1)


    with open(path_file+name_file_LSR,"a") as cfile:
            cfile.write(
                        str(r_LSR_f[0])+' '+str(r_LSR_f[1])+' '+str(r_LSR_f[2])+' '+ \
                        str(v_LSR_f[0])+' '+str(v_LSR_f[1])+' '+str(v_LSR_f[2])+' '+ \
                        str(creation_time)+' '+str(op_0)+'\n')


    for i in range(len(xx_RK4)):

        with open(path_file+name_file,"a") as cfile:
            cfile.write(
                        str(xx_RK4[i][0])+' '+str(xx_RK4[i][1])+' '+str(xx_RK4[i][2])+' '+ \
                        str(vv_RK4[i][0])+' '+str(vv_RK4[i][1])+' '+str(vv_RK4[i][2])+' '+ \
                        str(dd_RK4[i])+' '+str(creation_times[i])+' '+str(op_0)+' '+str(ids[i+1])+'\n')

    creation_time = creation_time+deltatT
    op_now = op_0+1

    while op_now<=op_f:

        for t in range(NT):

            r_LSR_f, v_LSR_f = RK4(r_LSR_f, v_LSR_f,h)

        sf = rop(limits,op_now)
        deltatT = sf.deltat_forward
        h = deltatT/NT

        where = np.array([np.where(abs(r_LSR_f[0]-sf.xx)<box_w[0])[0],np.where(abs(r_LSR_f[1]-sf.yy)<box_w[1])[0],np.where(abs(r_LSR_f[2]-sf.zz)<box_w[2])[0]])

        dd_box = np.array([[[sf.dd[k,j,i] for i in where[0]] for j in where[1]] for k in where[2]])
        zz_box = sf.zz[where[2]]
        yy_box = sf.yy[where[1]]
        xx_box = sf.xx[where[0]]

        vvx_box = np.array([[[sf.vv_x[k,j,i] for i in where[0]] for j in where[1]] for k in where[2]])*C.km.express(C.kpc)
        vvy_box = np.array([[[sf.vv_y[k,j,i] for i in where[0]] for j in where[1]] for k in where[2]])*C.km.express(C.kpc)
        vvz_box = np.array([[[sf.vv_z[k,j,i] for i in where[0]] for j in where[1]] for k in where[2]])*C.km.express(C.kpc)

        where_picos = np.where(dd_box>dens_th)

        dd_picos = [dd_box[where_picos[0][i],where_picos[1][i],where_picos[2][i]] for i in range(len(where_picos[0]))]
        zz_picos = zz_box[where_picos[0]]
        yy_picos = yy_box[where_picos[1]]
        xx_picos = xx_box[where_picos[2]]

        vvx_picos = [vvx_box[where_picos[0][i],where_picos[1][i],where_picos[2][i]] for i in range(len(where_picos[0]))]
        vvy_picos = [vvy_box[where_picos[0][i],where_picos[1][i],where_picos[2][i]] for i in range(len(where_picos[0]))]
        vvz_picos = [vvz_box[where_picos[0][i],where_picos[1][i],where_picos[2][i]] for i in range(len(where_picos[0]))]

        for t in range(NT):
            xx_RK4, vv_RK4 = [np.array(RK4(xx_RK4[i], vv_RK4[i],h)[0]) for i in range(len(xx_RK4))],[np.array(RK4(xx_RK4[i], vv_RK4[i],h)[1]) for i in range(len(xx_RK4))]

        xx_RK4.extend([np.array([xx_picos[i],yy_picos[i],zz_picos[i]]) for i in range(len(where_picos[0]))])
        vv_RK4.extend([np.array([vvx_picos[i],vvy_picos[i],vvz_picos[i]]) for i in range(len(where_picos[0]))])

        #dd_RK4.extend(dd_RK4)
        dd_RK4.extend(dd_picos)


        creation_times.extend([creation_time for i in range(len(where_picos[0]))])
        ids.extend(range(ids[-1]+1,ids[-1]+1+len(where_picos[0])+1))


        with open(path_file+name_file_LSR,"a") as cfile:
            cfile.write(
                        str(r_LSR_f[0])+' '+str(r_LSR_f[1])+' '+str(r_LSR_f[2])+' '+ \
                        str(v_LSR_f[0])+' '+str(v_LSR_f[1])+' '+str(v_LSR_f[2])+' '+ \
                        str(creation_time)+' '+str(op_now)+'\n')

        with open(path_file+name_file,"a") as cfile:
            for i in range(len(xx_RK4)):
                cfile.write(
                            str(xx_RK4[i][0])+' '+str(xx_RK4[i][1])+' '+str(xx_RK4[i][2])+' '+ \
                            str(vv_RK4[i][0])+' '+str(vv_RK4[i][1])+' '+str(vv_RK4[i][2])+' '+ \
                            str(dd_RK4[i])+' '+str(creation_times[i])+' '+str(op_now)+' '+str(ids[i+1])+'\n')

        op_now = op_now+1
        creation_time = creation_time+deltatT

    with open(path_file+'info_'+name_file,"a") as cfile:
        cfile.write("op_0 = "+str(op_0)+" \n" + \
                    "op_f = "+str(op_f) +" \n" +\
                    "limits = ("+str(limits[0])+","+str(limits[1])+","+str(limits[2])+","+str(limits[3])+","+str(limits[4])+","+str(limits[5])+") \n" + \
                    "NT = "+str(NT)+" \n" +\
                    "r_LSR = ("+str(r_LSR[0])+","+str(r_LSR[1])+","+str(r_LSR[2])+") \n"+\
                    "v_LSR = ("+str(v_LSR[0])+","+str(v_LSR[1])+","+str(v_LSR[2])+") \n"+ \
                    "rho_lim = "+str(rho_lim)+" \n"+ \
                    "box_dims = ("+str(box_dims[0])+","+str(box_dims[1])+","+str(box_dims[2])+") \n"+ \
                    "path_file = "+path_file+" \n"+ \
                    "name_file = "+name_file
                    )




def read_dat(path_file, name_file,name_file_LSR=None):

    if name_file_LSR==None:
        name_file_LSR = name_file+'LSR'

    canicas=read_table(path_file+name_file,sep="\s+",header=None)
    canicas.columns = ["x","y","z","vx","vy","vz","dens","creation_time","output","id"]
    datos_LSR=read_table(path_file+name_file_LSR,sep="\s+",header=None)
    datos_LSR.columns = ["x","y","z","vx","vy","vz","creation_time","output"]

    return canicas, datos_LSR

#####################################################################
# Mich
#####################################################################

def vel_circ(position,force,L=np.array([0.,0.,-1.])):
    position = np.array(position)
    v_uni = np.cross(L,position)/np.linalg.norm(position)
    v_mag = np.sqrt(abs(np.dot(force(position[0],position[1],position[2]),position)))

    return v_mag*v_uni


def get_names(startswith,dir = tardir):
    return [name.rsplit('/',1)[-1] for name in gb.glob(dir+startswith)]

def pdfilter(DF,keys_array,bool_funct,values,DF_mask=None):
    if DF_mask==None:
        DF_mask=DF

    bool_array = bool_funct[0](DF_mask[keys_array[0]],values[0])
    for i in range(1,len(keys_array)):
        bool_array = bool_array * bool_funct[i](DF_mask[keys_array[i]],values[i])

    return DF[bool_array]

def MR_andy(xx_t,vv_t):

    x_cm = 1/sum(mm)*(sum([mm[i]*np.array([xx_t[0][i],xx_t[1][i],xx_t[2][i]]) for i in range(len(mm))]))
    v_cm = 1/sum(mm)*(sum([mm[i]*np.array([vv_t[0][i],vv_t[1][i],vv_t[2][i]]) for i in range(len(mm))]))

    prod_cruz = [np.cross([xx_t[0][i],xx_t[1][i],xx_t[2][i]]-x_cm, [vv_t[0][i],vv_t[1][i],vv_t[2][i]]-v_cm) for i in range(len(mm))]

    MR = [np.median([prod_cruz[i][0] for i in range(len(mm))]),np.mean([prod_cruz[i][1] for i in range(len(mm))]),np.mean([prod_cruz[i][2] for i in range(len(mm))])]

    return np.array(MR)

def ME_andy(xx_t,vv_t):

    x_cm = 1/sum(mm)*(sum([mm[i]*np.array([xx_t[0][i],xx_t[1][i],xx_t[2][i]]) for i in range(len(mm))]))
    v_cm = 1/sum(mm)*(sum([mm[i]*np.array([vv_t[0][i],vv_t[1][i],vv_t[2][i]]) for i in range(len(mm))]))

    prod_punto = [np.dot([xx_t[0][i],xx_t[1][i],xx_t[2][i]]-x_cm, [vv_t[0][i],vv_t[1][i],vv_t[2][i]]-v_cm) for i in range(len(mm))]

    ME = np.median(prod_punto)
    return np.array(ME)

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


def plot_ocxoc(exp,oc1,oc2,tc_index,savefig=False,filename="ocxoc.pdf"):

    oc_dic = {"AB":0,"AC":1,"AK":2,"BC":3,"BK":4,"CK":5}
    oc_dic1 = {"A":0,"B":1,"C":2,"K":3}
    oci_dic = {"AB":[6,6],"AC":[6,4],"AK":[6,4],"BC":[6,4],"BK":[6,4],"CK":[4,4]}
    oc_index = oc_dic[oc1+oc2]
    auxtt2 = np.linspace(-np.pi,np.pi,200)
    evols_var2_cons,evols_var2 =exp
    varAB = [[evols_var2[1][81*oc_index+i*9+j] for i in range(9)] for j in range(9)]
    varAB_cons = [[evols_var2_cons[81*oc_index+i*9+j] for i in range(9)] for j in range(9)]
    Dt = 0.0199009793340327*2.65860467535e+15/60./60./24./365./1e6
    rc=.4


    fig, axes = plt.subplots(9, 9,figsize=(18,18))
    for i in range(len(axes)):
        for j in range(len(axes)):
            xxvv,xv_LSR = varAB[j][i][tc_index],evols_var2[0][tc_index]
            r_LSR = np.array(xv_LSR[:3:])
            v_LSR = np.array(xv_LSR[3::])
            cc = np.array([xv[:3:] for xv in xxvv])
            vv = np.array([xv[3::] for xv in xxvv])
            vvkm = vv
            axes[j][i].grid(True)
            vv_galac = np.array([oc.vcart2vgalactic(cc[k],r_LSR,vvkm[k],v_LSR)*mt.C.kpc.express(mt.C.km) for k in range(len(vvkm))])
            cc_galac = np.array([oc.cart2galactic(c,r_LSR) for c in cc])
            axes[j][i].scatter(cc_galac[:,0]*np.cos(cc_galac[:,1]),cc_galac[:,0]*np.sin(cc_galac[:,1]),alpha=.5,s=8)
            axes[j][i].plot([0,10],[0,0],ls="--",c='k',alpha=.5)
            axes[j][i].plot(rc*np.cos(auxtt2),rc*np.sin(auxtt2),ls="--",c='k',alpha=.5)
            axes[j][i].scatter(cc_galac[0:-1:15,0]*np.cos(cc_galac[0:-1:15,1]),\
                               cc_galac[0:-1:15,0]*np.sin(cc_galac[0:-1:15,1]),alpha=1,s=30,c="r")
            axes[j][i].scatter(cc_galac[0,0]*np.cos(cc_galac[0,1]),cc_galac[0,0]*np.sin(cc_galac[0,1]),alpha=1,s=60,c="r")
            axes[j][i].quiver(cc_galac[0:-1:15,0]*np.cos(cc_galac[0:-1:15,1]),\
                              cc_galac[0:-1:15,0]*np.sin(cc_galac[0:-1:15,1]),\
                              vv_galac[0:-1:15,0]*np.cos(cc_galac[0:-1:15,1])-vv_galac[0:-1:15,1]*np.sin(cc_galac[0:-1:15,1]),\
                              vv_galac[0:-1:15,0]*np.sin(cc_galac[0:-1:15,1])+vv_galac[0:-1:15,1]*np.cos(cc_galac[0:-1:15,1]),\
                              angles='xy', units='width', scale=50,color="k",alpha=1)
            axes[j][i].set_xlim(-2*rc,2*rc)
            axes[j][i].set_ylim(-2*rc,2*rc)
            #axes[j][i].get_yaxis().set_visible(False)
            #axes[j][i].get_xaxis().set_visible(False)
            axes[j][i].xaxis.set_ticklabels([])
            axes[j][i].yaxis.set_ticklabels([])
            axes[j][i].minorticks_on()
            axes[j][i].set_axisbelow(True)

    x2red,y2red = oci_dic[oc1+oc2]
    axes[y2red][x2red].spines["bottom"].set_color("red")
    axes[y2red][x2red].spines["left"].set_color("red")
    axes[y2red][x2red].spines["top"].set_color("red")
    axes[y2red][x2red].spines["right"].set_color("red")
    fig.tight_layout()
    for i in range(9):
        axes[8][i].set_xlabel("%g"%varAB_cons[8][i][oc_dic1[oc1]])
    for j in range(9):
        axes[j][0].set_ylabel("%g"%varAB_cons[j][0][oc_dic1[oc2]])
    axes[8][4].set_title(oc1,y=-.4,fontsize=16)
    axes[4][0].set_title(oc2,x=-.4,y=.5,fontsize=16)
    #axes[0][4].set_title("t = %g Myr"%(Dt*tc_index),fontsize=18)
    #fig.show()
    if savefig:
        fig.savefig(filename,bbox_inches='tight')
        plt.close(fig);
    return fig

for t in range(11):
    plot_ocxoc(exp=exp3,oc1="A",oc2="B",tc_index=t,savefig=True,filename=varOC_dir+"gra_AvsB_quiv_"+str(t)+".pdf")
    plot_ocxoc(exp=exp3,oc1="A",oc2="C",tc_index=t,savefig=True,filename=varOC_dir+"gra_AvsC_quiv_"+str(t)+".pdf")
    plot_ocxoc(exp=exp3,oc1="A",oc2="K",tc_index=t,savefig=True,filename=varOC_dir+"gra_AvsK_quiv_"+str(t)+".pdf")
    plot_ocxoc(exp=exp3,oc1="B",oc2="C",tc_index=t,savefig=True,filename=varOC_dir+"gra_BvsC_quiv_"+str(t)+".pdf")
    plot_ocxoc(exp=exp3,oc1="B",oc2="K",tc_index=t,savefig=True,filename=varOC_dir+"gra_BvsK_quiv_"+str(t)+".pdf")
    plot_ocxoc(exp=exp3,oc1="C",oc2="K",tc_index=t,savefig=True,filename=varOC_dir+"gra_CvsK_quiv_"+str(t)+".pdf")

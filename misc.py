import numpy as np
import numpy.random as npr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
import matplotlib.colors as mcol
import scipy.stats as sts
from sklearn.neighbors import KernelDensity
from mpl_toolkits.axes_grid1 import make_axes_locatable

def hess_arr(x_val,y_val,xy_ext,nbins):
    x_val = (x_val - xy_ext[0])/(xy_ext[1]-xy_ext[0])
    y_val = (y_val - xy_ext[2])/(xy_ext[3]-xy_ext[2])
    dz=1.0/nbins
    
    
    xx,yy=np.meshgrid(np.arange(dz,1+dz,dz),np.arange(dz,1+2*dz,dz))
    xy_sample = np.vstack([yy[:-1].ravel(), xx[:-1].ravel()]).T
    xy_train  = np.vstack([y_val,x_val]).T

    kde_skl = KernelDensity(kernel='gaussian', bandwidth=0.08)
    kde_skl.fit(xy_train)
    hss_est = np.exp(np.reshape(kde_skl.score_samples(xy_sample),(nbins,nbins)))
    #hss_est = hss_est/((xy_ext[3] - xy_ext[2])*(xy_ext[1] - xy_ext[0]))
    hss_est = hss_est/np.sum(hss_est)
    #hss_est = hss_est/np.amax(hss_est)
    #print(dz*dz*np.sum(hss_est))

    return hss_est

def hist_2d_marg_axes(Xlabel,Ylabel,XYext):
        left, width = 0.175, 0.65
        bottom, height = 0.1, 0.65
        spacing = 0.0
    
        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom + height + spacing, width, 0.25]
        rect_histy = [left + width + spacing, bottom, 0.175, height]
        
        fig = plt.figure(figsize=(10, 10))
        
        ax_scatter = plt.axes(rect_scatter)
        ax_scatter.tick_params(direction='in', top=True, right=True)
        ax_histx = plt.axes(rect_histx)
        ax_histx.tick_params(direction='in', labelbottom=False)
        ax_histy = plt.axes(rect_histy)
        ax_histy.tick_params(direction='in', labelleft=False)
        
        ax_scatter.set_xlabel(Xlabel,fontsize=42)
        ax_scatter.set_ylabel(Ylabel,fontsize=42)
        ax_scatter.set_xlim((XYext[0],XYext[1]))
        ax_scatter.set_ylim((XYext[2],XYext[3]))

        return fig

def marg_hist_1d(AXIS,W,Wext,LAYER,ORIEN='horizontal'):
        Wbins1 = np.linspace(Wext[0],Wext[1],40)
        kde_1D = KernelDensity(kernel='gaussian', bandwidth=0.1*(Wext[1]-Wext[0])).fit(W.values[:, np.newaxis])
        Wkde_1D = np.exp(kde_1D.score_samples(Wbins1[:, np.newaxis]))
        if ORIEN=='horizontal': 
            AXIS.plot(Wbins1,Wkde_1D/np.sum(Wkde_1D),color='black',lw=3,alpha=max(0.4,LAYER))
            AXIS.tick_params(axis='y', labelleft=False)
        elif ORIEN=='vertical': 
            AXIS.plot(Wkde_1D/np.sum(Wkde_1D),Wbins1,color='black',lw=3,alpha=max(0.4,LAYER))
            AXIS.tick_params(axis='x', labelbottom=False)

    
def cmap_hists(FIG,X,Y,LAYER,XYext,Xlabel,Ylabel,CMAP,Z=None,Zext=None,Zlabel=None):
        Xval = (X - XYext[0])/(XYext[1]-XYext[0])
        Yval = (Y - XYext[2])/(XYext[3]-XYext[2])
        UV = np.linspace(0,1,20)

        ax = FIG.axes
    
        #ax_scatter.scatter(X,Y,alpha=0.6,s=40,c=Z,cmap=mcm.coolwarm,vmin=Zbins[0],vmax=Zbins[-1])
        ret = sts.binned_statistic_2d(Xval, Yval, Z, statistic='median', bins=[UV,UV])
        im = ax[0].imshow(np.transpose(ret.statistic),cmap=CMAP,aspect='auto',extent =XYext,origin='lower',interpolation='gaussian',vmin=Zext[0],vmax=Zext[1],alpha=max(0.2,LAYER))

        marg_hist_1d(ax[1],X,XYext[:2],LAYER)
        marg_hist_1d(ax[2],Y,XYext[2:],LAYER,ORIEN='vertical')

        if LAYER>0:
            H = hess_arr(X, Y, XYext,40)
            CS = ax[0].contour(H,lw=6,extent = XYext,origin='lower',colors='black')
            ax[0].clabel(CS, fontsize=14)

            divider = make_axes_locatable(ax[2])
            cax = divider.append_axes('right', size='10%', pad=0.0)
            cbar = FIG.colorbar(im, cax=cax, orientation='vertical')
            cbar.set_label(Zlabel, size=28) 

def scatter_hists(X,Y,Xbins,Ybins,Xlabel,Ylabel,Z=None,Zbins=None):
        fig = hist_2d_marg_axes(Xlabel,Ylabel,XYext)
        ax = fig.axes
        
        ax_scatter.scatter(X,Y,alpha=0.6,s=40,c=Z,cmap=mcm.coolwarm,vmin=Zbins[0],vmax=Zbins[-1])
        
        marg_hist_1d(ax[1],X,XYext[:2])
        marg_hist_1d(ax[2],Y,XYext[2:],ORIEN='vertical')
        ax_scatter.set_rasterized(True)
        
        
        

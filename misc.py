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

def cmap_hists(X,Y,XYext,Xlabel,Ylabel,CMAP,Z=None,Zext=None,Zlabel=None):
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

        Xval = (X - XYext[0])/(XYext[1]-XYext[0])
        Yval = (Y - XYext[2])/(XYext[3]-XYext[2])
        UV = np.linspace(0,1,20)

        #ax_scatter.scatter(X,Y,alpha=0.6,s=40,c=Z,cmap=mcm.coolwarm,vmin=Zbins[0],vmax=Zbins[-1])
        ret = sts.binned_statistic_2d(Xval, Yval, Z, statistic='median', bins=[UV,UV])
        im = ax_scatter.imshow(np.transpose(ret.statistic),cmap=CMAP,aspect='auto',extent = XYext,origin='lower',interpolation='gaussian',vmin=Zext[0],vmax=Zext[1])

        H = hess_arr(X, Y, XYext,40)
        CS = ax_scatter.contour(H,lw=6,extent = XYext,origin='lower',colors='black')
        ax_scatter.clabel(CS, fontsize=14)

        # use the previously defined function
        Xbins1,Ybins1 = np.linspace(XYext[0],XYext[1],40),np.linspace(XYext[2],XYext[3],40)
        kde_1D = KernelDensity(kernel='gaussian', bandwidth=0.1*(XYext[3]-XYext[2])).fit(Y.values[:, np.newaxis])
        Ykde_1D = np.exp(kde_1D.score_samples(Ybins1[:, np.newaxis]))
        ax_histy.plot(Ykde_1D/np.sum(Ykde_1D),Ybins1,color='black',lw=3)

        kde_1D = KernelDensity(kernel='gaussian', bandwidth=0.1*(XYext[1]-XYext[0])).fit(X.values[:, np.newaxis])
        Xkde_1D = np.exp(kde_1D.score_samples(Xbins1[:, np.newaxis]))
        ax_histx.plot(Xbins1,Xkde_1D/np.sum(Xkde_1D),color='black',lw=3)
        #ax_histy.hist(Y,bins=Ybins,color='black',histtype='step',lw=3,orientation='horizontal')
        #ax_histx.hist(X,bins=Xbins,color='black',lw=3,histtype='step')

        ax_histy.tick_params(axis='x', labelbottom=False)
        ax_histx.tick_params(axis='y', labelleft=False)

        divider = make_axes_locatable(ax_histy)
        cax = divider.append_axes('right', size='10%', pad=0.0)
        cbar = fig.colorbar(im, cax=cax, orientation='vertical')
        cbar.set_label(Zlabel, size=28) 



        #ax_scatter.set_rasterized(True)

def scatter_hists(X,Y,Xbins,Ybins,Xlabel,Ylabel,Z=None,Zbins=None):
        left, width = 0.175, 0.65
        bottom, height = 0.1, 0.65
        spacing = 0.0
    
        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom + height + spacing, width, 0.25]
        rect_histy = [left + width + spacing, bottom, 0.175, height]
        
        plt.figure(figsize=(8, 8))
        
        ax_scatter = plt.axes(rect_scatter)
        ax_scatter.tick_params(direction='in', top=True, right=True)
        ax_histx = plt.axes(rect_histx)
        ax_histx.tick_params(direction='in', labelbottom=False)
        ax_histy = plt.axes(rect_histy)
        ax_histy.tick_params(direction='in', labelleft=False)
        
        ax_scatter.set_xlabel(Xlabel,fontsize=28)
        ax_scatter.set_ylabel(Ylabel,fontsize=28)
        ax_scatter.set_xlim((Xbins[0],Xbins[-1]))
        ax_scatter.set_ylim((Ybins[0],Ybins[-1]))
        
        ax_scatter.scatter(X,Y,alpha=0.6,s=40,c=Z,cmap=mcm.coolwarm,vmin=Zbins[0],vmax=Zbins[-1])
        
        # use the previously defined function
        ax_histy.hist(Y,bins=Ybins,color='black',histtype='step',lw=3,density=True,orientation='horizontal')
        ax_histx.hist(X,bins=Xbins,color='black',lw=3,histtype='step',density=True)
        ax_scatter.set_rasterized(True)
        
        
        

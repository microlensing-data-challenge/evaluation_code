# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 15:55:48 2019

@author: rstreet
"""

from sys import argv
import numpy as np
import matplotlib.pyplot as plt

plt.rc('font', family='DejaVu Sans')
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)

def plot_lightcurves(fileW,fileZ,offZ):
    
    d1 = np.loadtxt(fileW)
    d2 = np.loadtxt(fileZ)
    
    fig = plt.figure(1,(10,10))

    dt = 2458000.0
    
    plt.errorbar(d1[:,0]-dt,d1[:,1],yerr=d1[:,2],fmt='.',ecolor='b',color='b',label='W149')
    
    if offZ != 0.0:
        zlabel = 'Z087 offset by '+str(offZ)+'mag'
    else:
        'Z087'
        
    plt.errorbar(d2[:,0]-dt,d2[:,1]+offZ,yerr=d2[:,2],fmt='.',ecolor='r',color='r',label=zlabel)
    
    plt.xlabel('HJD - '+str(dt),fontsize=18)
    plt.ylabel('Magnitude',fontsize=18)
    
    plt.legend(fontsize=18)
    plt.grid()
    
    [xmin,xmax,ymin,ymax] = plt.axis()
    plt.axis([xmin,xmax,ymax,ymin])
    
    plt.savefig('wfirst_lightcurve.png',bbox_inches='tight')
    
    plt.close(1)
    
if __name__ == '__main__':
    
    if len(argv) < 3:
        
        fileW = input('Please enter the path to the W149 lightcurve datafile: ')
        fileZ = input('Please enter the path to the Z087 lightcurve datafile: ')
        offZ = float(input('Please magnitude offset for the Z087 lightcurve: '))
        
    else:
        
        fileW = argv[1]
        fileZ = argv[2]
        offZ = float(argv[3])
    
    plot_lightcurves(fileW,fileZ,offZ)
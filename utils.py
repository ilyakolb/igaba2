# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 08:11:51 2020

@author: kolbi
    
"""

import pandas as pd
import numpy as np

def calcBg(streamFrame, percent):
    """
    calculate background of stream image (bottom (percent)% of image)

    inputs
    ======
    streamFrame: single frame from tif stack
    percent: bottom % to be considered background and subtracted from Ftrace
    """
    allVals = np.ndarray.flatten(streamFrame)
    allVals_sorted = np.sort(allVals)

    lenData = len(allVals)
    bgInd = int(np.round(lenData * percent / 100))
    if bgInd < 1: 
        bgInd = 1
    elif bgInd > lenData:
        bgInd = lenData

    bg = allVals_sorted[bgInd]
    return bg

def save_clipboard(filename = 'out.pkl'):
    '''
    save clipboard to pandas array
    
    INSTRUCTIONS
    ============
    
    if using with Fiji, save plot as Data >> Copy all data
    
    Run from command line:
        > conda activate base2
        > Z:
        > cd <destination folder>
        > save_clipboard(r'D:\imaging_local\20201124_iGABASnFR_imaging_IK\5AP\1_1.pkl')
    '''
    
    d = pd.read_clipboard()
    
    print(d.columns)
    if 'X' not in d.columns and 'Slice' not in d.columns and 'X0' not in d.columns:
        raise ValueError('X / X0 / Slice not in clipboard!')
    
    first_col = d.columns[0] # can be 'X' or 'X0'
    d.drop([first_col], axis=1)
    
    if filename == 'out.pkl':
        print('Saving to default out.pkl')
    
    print('Saving to: ' + filename)
    d.to_pickle(filename)
    print('Saved')

def calc_dff_trace(bg_trace, f_trace):
    # calculate df/f
    f0 = np.mean(f_trace[:100])
    return (f_trace-f0)/(f0-bg_trace)

def debleach(f_trace, t):
    '''
    simple linear regression
    f_trace: [1xN] F trace
    t: [1xN] entire time vector
    '''
    fit_idx = np.concatenate((np.arange(50), np.arange(-50,-1)))
    x_fit = t[fit_idx]
    y_fit = f_trace[fit_idx]
    p = np.polyfit(x_fit, y_fit, 1)
    return f_trace - np.polyval(p, t)
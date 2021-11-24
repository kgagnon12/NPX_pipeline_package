import numpy as np
import matplotlib.pyplot as plt
import _pickle as pkl
import pandas as pd
import os,sys,glob, h5py

from neuropixels import generalephys as ephys
from neuropixels.generalephys import placeAxesOnGrid, cleanAxes
from neuropixels import utils
from neuropixels import rf_analysis
from neuropixels import psth_and_raster as psth
from neuropixels import reliability_and_precision as rp
from neuropixels.continuous_traces import gaussian_filter1d
from scipy.signal import find_peaks_cwt
from scipy.stats import pearsonr, spearmanr, zscore
from itertools import combinations, chain

import seaborn as sns
sns.set_style('white')
%matplotlib inline
%load_ext autoreload
%autoreload 2
import warnings
warnings.filterwarnings('ignore')






# cross correlation normalized to geometric mean
def ccg(train1,train2,binrange,binsize,correction=False):
    diffs = []
    count=0
    if correction != False:
        if correction == 'jitter':
            train1 = jitter(train1,binsize)
            train2 = jitter(train2,binsize)
        else: 
            print('improper correction method specified')
            return np.nan
    if len(train1) > 1 and len(train2) > 1:
        for spiketime_train1 in train1:
            #for spiketime_train2 in train2:
            if train2[-1] > spiketime_train1 + binrange[0]: # if there are any spikes from train2 after the start of the window 
                start = np.where(train2 > spiketime_train1 + binrange[0])[0][0]

                if train2[-1] > spiketime_train1 + binrange[1]:#set the end of train2 to only relevant window around this spike
                    end = np.where(train2 > spiketime_train1 + binrange[1])[0][0]
                else:
                    end = len(train2)

                for spiketime_train2 in train2[start:end]:
                    diffs.extend([float(spiketime_train1) - float(spiketime_train2)])
                    count+=1
        diffs = np.array(diffs)*-1
        hist,edges = np.histogram(diffs,bins=int((binrange[1]-binrange[0])/binsize),range=binrange)
        return (hist / float(len(train1)))*100,edges
    else:
        print('input spiketrains not long enough: 1:'+str(len(train1))+' 2:'+str(len(train2)))
        return [0],[0,0]






def jitter(train,binsize):
    start = np.min(train) - (binsize/2.)
    end = np.max(train) + binsize
    bin_left_edges = np.arange(start,end,binsize)
    to = [train[np.where((train < bin_left_edges[i+1]) & (train > bin_left_edges[i]))[0]] - bin_left_edges[i]  for i in range(len(bin_left_edges)-1)]
    all_to = np.concatenate(to).ravel()
    jittered = []
    spike_indices = np.arange(all_to.shape[0]).tolist()
    np.random.seed()
    np.random.shuffle(spike_indices)
    for i,bin_ in tqdm.tqdm(enumerate(to)):
        spikes = []
        for spike in bin_:
            spikes.extend([all_to[spike_indices.pop(0)]])
        jittered.append(np.array(spikes) + bin_left_edges[i])
    jittered = np.concatenate(jittered).ravel()
    return jittered





def cross_corr(df,binsize,lag=0.5,mouseid):
    %%time
    df_pairs = pd.DataFrame(index=a,
                            columns=['mouse','ind1','ind2','cell1','cell2','type1','type2','cohort'])

    start_ = 0
    mouseid = mouseid
    numpairs=len(list(combinations(df.cell,2)))
        
    df_pairs['mouse'][start_:start_+numpairs]=mouseid
    df_pairs['cell1']=[c[0] for c in list(combinations(df.cell,2))]
    df_pairs['cell2']=[c[1] for c in list(combinations(df.cell,2))]
    df_pairs['ind1']=[c[0] for c in list(combinations(df.index,2))]
    df_pairs['ind2']=[c[1] for c in list(combinations(df.index,2))]
    df_pairs['type1']=[c[0] for c in list(combinations(df.waveform_class,2))]
    df_pairs['type2']=[c[1] for c in list(combinations(df.waveform_class,2))]
    df_pairs['ypos1']=[c[0] for c in list(combinations(df.ypos,2))]
    df_pairs['ypos2']=[c[1] for c in list(combinations(df.ypos,2))]
    df_pairs['cohort']=df.cohort.unique()[0]
        
    start_+=numpairs

    hists = []
    for i in df_pairs.index:
        train1 = np.array(df.times[df_pairs.ind1[i]])
        train2 = np.array(df.times[df_pairs.ind2[i]])
        hist = ccg(train1,train2,(-lag,lag),binsize)
        hists.append(hist)
    df_pairs['ccg'] = hists

    hists_ = []
    for i in df_pairs.index:
        train1 = np.array(df.jitter[df_pairs.ind1[i]])
        train2 = np.array(df.jitter[df_pairs.ind2[i]])
        hist = ccg(train1,train2,(-lag,lag),binsize)
        hists_.append(hist)
    df_pairs['jitter_ccg'] = hists

    for i in df_pairs.index:
        corrected = df_pairs.ccg[i] - df_pairs.jitter_ccg[i]
        corrected_ccgs.append(corrected)
    df_pairs['corrected_ccgs'] = corrected_ccgs

    return df_pairs




def find_pairs(df_pairs):

##############################################################################

#       if the maximum value between 'mid' is greater than 2.5 * STD of the pair
#       if no value outside 'mid' is greater than mx (i.e. the max value in mid is the peak)

#################################################################################
    pairs = []
    empties = []
    ccgs = []

    for i,pair in enumerate(df_pairs.corrected_ccg):
        mn = np.mean(pair[300:450])
        std = np.std(pair[300:450])
        #std = np.std(pair)
        vert_thresh = mn+(2.5*std) #theshold for height
        horiz_thresh = mn+(2.8*std) #if a point is greater than this number, discard
        mid = pair[490:510]
        mx = np.max(mid) * 0.60
        #if any(sides) > 1/4 of max(peak): then empty
        if any(mid>vert_thresh) and any(pair[440:490] > mx) == False and any(pair[510:650] > mx) == False:
                pairs.append('yes')
            
        else:
        pairs.append('no')

        df_pairs['result'] = pairs

    return df_pairs



def find_cell_copies(df_pairs):

##############################################################################

#       determines if cell pair is two unsplit units (not merged when sorted)
#       i.e. two identical units

#       if the units are within 80 microns of each other on the probe and has a 
#       max cross correlation greater than 15 standard deviations

#################################################################################

    confirmed = []

    for j,res in enumerate(df_pairs.result):
        if res == 'yes':
            unit1 = df_pairs.ind1[j]
            unit2 = df_pairs.ind2[j]
            depth_unit1 = df.ypos[unit1]
            depth_unit2 = df.ypos[unit2]
            pk = np.max(df_pairs.corrected_ccg[j][490:510])
            max_thresh = np.std(df_pairs.corrected_ccg[j])*15 + np.mean(df_pairs.corrected_ccg[j])
                
                # how much can a unit drift on the probe?
            if abs(depth_unit1-depth_unit2)<80 and pk>max_thresh:
                confirmed.append('copy')
            else:
                confirmed.append('yes')
        else:
            confirmed.append('None')
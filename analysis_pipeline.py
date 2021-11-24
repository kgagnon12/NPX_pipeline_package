import numpy as np
import matplotlib.pyplot as plt
import _pickle as pkl
import pandas as pd
import os,sys,glob, h5py

from neuropixels import generalephys as ephys
from neuropixels.generalephys import placeAxesOnGrid, cleanAxes
from neuropixels import utils_pipeline as utils
from neuropixels import rf_analysis
from neuropixels import psth_and_raster as psth
from neuropixels import reliability_and_precision as rp
from neuropixels.continuous_traces import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.stats import pearsonr, spearmanr, zscore
from itertools import combinations 

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import seaborn as sns
import warnings



###################################



def synchphys(software_start,frame_timestamps,df_reaches,samplerate = 30000):
#synch the ephys with the behavior to find reachmax timessoftware_start = 56549737/30000 #convert software start time in samples to seconds
    st = software_start/samplerate
    ts_sec = frame_timestamps/samplerate # convert frame timestamps to seconds
    ts_z = ts_sec - st #subtract software start from frames
    reachmax = np.array(df_reaches.reachMax)
    reachmax = reachmax[~np.isnan(reachmax)]
    reachmax = reachmax.astype('int')
    reach_times = ts_z[reachmax]
    return(reach_times)


def batch_synch_phys(df,df_start,df_reaches,df_timestamps,mouseid,inputs=False):
    sst = np.int(df_start[df_start.Mouse_number==mouseid].Proccessor_Start_time)
    ind = df_timestamps[df_timestamps.mouse==mouseid].index
    ts = np.array(df_timestamps[df_timestamps.mouse==mouseid].timestamps[ind])
    ts = np.array(ts[0])
    df_reaches = df_reaches[df_reaches.mouse==mouseid]
    print('mouse: '+str(mouseid))
    print('number of timestamps: '+str(np.shape(ts)))
#    if np.shape(ts)[0] > 1000000:
    bad_align = []
    if inputs==True:
        need_ts = input("remove odd indices? (yes or no) ")
        if need_ts == 'yes':
            print('removed odd indices')
            ts_lst = ts.tolist()
            del ts_lst[1::2] #remove odd indices
            ts = np.array(ts_lst)
        else:
            ts = ts
        try:
            reach_times = synchphys(sst,ts,df_reaches)
            df_reaches['rMax_t']= reach_times
            print('aligned')
        except:
            print('FRAME TIMESTAMPS DO NOT ALIGN WITH BEHAVIOR '+str(mouseid) + '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            pass
        print('plotting heatmap to confirm')
        trial_ave_heatmap(df,df_reaches,mouseid,binsize=0.020, start = -3.0,end = 3.0)
        print('heatmap complete')
        check_heatmap = input("is this correct? (yes or no) ")
        bad_align = []
        if check_heatmap == 'no':
            print('ALIGNING FAILED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! MOUSEID APPENDED TO BAD_ALIGN OUTPUT LIST')
            bad_align.append(mouseid)
        else:
            return(df_reaches)
    else:
        if np.shape(ts)[0] > 872236:
            print('removed odd indices')
            ts_lst = ts.tolist()
            del ts_lst[1::2] #remove odd indices
            ts = np.array(ts_lst)
        else:
            ts = ts
        try:
            reach_times = synchphys(sst,ts,df_reaches)
            df_reaches['rMax_t']= reach_times
            print('aligned')
            return(df_reaches)
        except:
            print('FRAME TIMESTAMPS DO NOT ALIGN WITH BEHAVIOR '+str(mouseid) + '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            bad_align.append(mouseid)
            pass



def reach_info(df,df_reaches,mouseid):
    print('computing reach epoch info for mouse ' +str(mouseid))
    df = df[df.mouse==mouseid].reset_index()
    df_reaches = df_reaches[df_reaches.mouse==mouseid].reset_index()
    df_align = pd.DataFrame(df,columns = ['times','depth','layer','group','mouse','cohort','cell','overall_rate','waveform_class','baseline_fr','reach_fr', 'mod up', 'mod down', 'peakfr_up', 'peakfr_down', 'max_epochfr', 'min_epochfr'])
    try: 
        reaches = epochfr(df,df_reaches,start = -0.5,end = 0.5,binsize=0.020)
        df_align['reach_fr'] = reaches
        baseline = epochfr_baseline(df,df_reaches,start = -1.0,end = -0.5,binsize=0.020) 
        df_align['baseline_fr'] = baseline
        ave_reach,normedbins,ave_reach_ = reachave_tensor(df,df_reaches)
        df_align['bin_ave'] = ave_reach
        df_align['norm_bin_ave'] = normedbins
        mod_up, mod_down, maxfr_ts, minfr_ts, maxfr, minfr = movement_mod(df,df_reaches,startb = -1.0,endb = -0.5,starte = -0.5,ende = 0.5,binsize=0.001)
        df_align['mod_up'] = mod_up
        df_align['mod_down'] = mod_down
        df_align['peakfr_up'] = maxfr_ts
        df_align['peakfr_down'] = minfr_ts
        df_align['max_epochfr'] = maxfr
        df_align['min_epochfr'] = minfr
    except:
        print(str(mouseid) + ' not aligned properly .. check heatmap and timestamps')
    return df_align

################# COMPLETE #####################

def trial_ave_heatmap(df,df_reaches,id_,binsize=0.020, start = -4.0,end = 2.0):
    #START IS TIME BEFORE REACHMAX
    #END IS TIME AFTER REACHMAX -- IF BEFORE REACHMAX THEN END MUST BE NEGATIVE
    #collapses mean fr along bins
    #initialize 
    df = df[df.mouse==id_].reset_index()
    df_reaches=df_reaches[df_reaches.mouse==id_].reset_index()
    print('creating heatmap for ' + str(id_) + '.....')

    try:
        mod_up, mod_down, maxfr_ts, minfr_ts = movement_mod_heatmap(df,df_reaches)
        df['mod_up'] = mod_up
        df['mod_down'] = mod_down
        df['peak_up'] = maxfr_ts
        df['peak_down'] = minfr_ts

        print('movement modulated units found')

        edges=np.arange(start,end,binsize)
        num_bins=edges.shape[0]-1 #number of bins 
        byreach=np.zeros((len(df_reaches.rMax_t),num_bins))
        ave_reach_=np.zeros((len(df),num_bins)) #for tensor purposes
        ave_reach = []
        normedbins = []

        for i,times in enumerate(df.times): #for each unit 
            t = np.array(times) #for reach unit create an array of that unit's spike times
            for j,tmax in enumerate(df_reaches.rMax_t): #compare that unit's spike times to each reach max
                rd = np.array(t[(t >= tmax+start) & (t <= tmax+end)]) #find if that unit spiked within designated timeframes around reachmax
                edges=np.arange(tmax+start,tmax+end,binsize) #designated bins around this iteration of reachmax
                hist=np.histogram(rd,edges)[0] #bin spikes into timeframe
                byreach[j,:] = hist
            meanbinfr = np.mean(byreach,axis=0)
            ave_reach.append(meanbinfr)
        
            ave_reach_[i,:] = meanbinfr #for tensor purposes (neural trajectories)
        
            normbins = (meanbinfr-min(meanbinfr))/max(meanbinfr) #per dailey
            normedbins.append(normbins)
        
        print('plotting heatmap')

        df['binz'] = normedbins
        df_s= df.sort_values(by=['peak_up'])
        df_bins_ = df_s.binz
        df_heatmap = list(filter(any, df_bins_))
        fig, ax = plt.subplots(figsize=(20, 10))
        fig = sns.heatmap(df_heatmap)
        plt.title(str(id_))
        plt.show()
        plt.close()
    except:
        print('movement modulation units not found mouse ' +str(id_))
        pass



################ EDITING ########################


def reach_psth(df,df_reaches,binsize=0.020,start=-3,end=8):

    edges=np.arange(start,end,binsize)
    num_bins=edges.shape[0] #number of bins 
    byreach=np.zeros((np.shape(df)[0],num_bins))
    alltrials = []
    outcomes = []
    units = []
    reach = []

    edges=np.arange(start,end,binsize)
    num_bins=edges.shape[0] #number of bins 
    byreach=np.zeros((np.shape(df_reaches)[0],num_bins))
    alltrials = []
    trajs = []
    evs = []
    outcomes = []
    units = []
    reach = []
    units_=[]
    reachs=[]
    hists=[]
    hist_rastor = []

    for i,times in enumerate(df.times): #compare that unit's spike times to each reach max
        t = np.array(times) #for reach unit create an array of that unit's spike times
        byreach=np.zeros((np.shape(df_reaches)[0],num_bins))
        for j,tmax in enumerate(df_reaches.rMax_t): #for each unit 
            a = tmax+start
            b = tmax+end
            unit = df.index[i]
            units.append(unit)
            rea = df_reaches.behaviors[j]
            reach.append(rea)
            try:
                rd = np.array(t[(t >= a) & (t <= b)]) #find if that unit spiked within designated timeframes around reachmax
                edges=np.arange(a,b,binsize) #designated bins around this iteration of reachmax
                num_bins=edges.shape[0]-1 #number of bins 
                hist=np.histogram(rd,edges)[0] #bin spikes into timeframe
                byreach[j,:] = hist
                byreach_df = hists.append(hist) # list
                byreach = center(byreach)
            except:
                []

        alltrials.append(byreach) # tensor rastor - every trial response for every neuron
        psth = np.sum(alltrials) # 
        reachs.append(reach) # which reach
        units_.append(units) # unit id
        hist_rastor.append(hists) #list rastor 

    return alltrials,psth,reachs,units_,hist_rastor











################ TO EDIT #########################
  
    
def epochfr(df,df_reaches,start,end,binsize=0.020):
    ## if epoch is BEFORE reachmax (such as baseline) then end variable must be input as negative as per code (i.e. end = -0.5)
    
    byreach=np.zeros((len(df),1))
    rd = []
    frs = []
    
    for i,times in enumerate(df.times): #for each unit 
        t = np.array(times) #for reach unit create an array of that unit's spike times
        for j,tmax in enumerate(df_reaches.rMax_t): #compare that unit's spike times to each reach max
            #rd = np.array(t[(t >= tmax-start) & (t <= tmax+end)]) #find if that unit spiked within designated timeframes around reachmax
            rd = np.array(t[(t >= tmax+start) & (t <= tmax+end)]) #find if that unit spiked within designated timeframes around reachmax
            #edges=np.arange(tmax-start,tmax+end,binsize) #designated bins around specific iteration of reachmax
            edges=np.arange(tmax+start,tmax+end,binsize)
            hist=np.histogram(rd,edges)[0] #bin spikes into timeframe
            fr = sum(hist)/abs(end-start) # in Hz 
            frs.extend([fr])
        meanfr = np.mean(frs)
        frs = []
        byreach[i,:] = meanfr

    return byreach

def epochfr_baseline(df,df_reaches,start,end,binsize=0.020):
    ## if epoch is BEFORE reachmax (such as baseline) then end variable must be input as negative as per code (i.e. end = -0.5)
    
    byreach=np.zeros((len(df),1))
    rd = []
    frs = []
    
    for i,times in enumerate(df.times): #for each unit 
        t = np.array(times) #for reach unit create an array of that unit's spike times
        for j,tmax in enumerate(df_reaches.rMax_t): #compare that unit's spike times to each reach max
            #rd = np.array(t[(t >= tmax-start) & (t <= tmax+end)]) #find if that unit spiked within designated timeframes around reachmax
            rd = np.array(t[(t >= tmax+start) & (t <= tmax+end)]) #find if that unit spiked within designated timeframes around reachmax
            #edges=np.arange(tmax-start,tmax+end,binsize) #designated bins around specific iteration of reachmax
            edges=np.arange(tmax+start,tmax+end,binsize)
            hist=np.histogram(rd,edges)[0] #bin spikes into timeframe
            fr = sum(hist)/abs(end-start) ######????????????????????????????????????????? 
            frs.extend([fr])
        meanfr = np.mean(frs)
        frs = []
        byreach[i,:] = meanfr
    return byreach


def reachave_bins(df,df_reaches,start,end,binsize=0.020): #DO NOT USE
    #collapses mean fr along bins and normalizes
    #start and end variables are negative if before reachmax (i.e. baseline)
    asdfasdfasdf
    #initialize 
    ave_reach = []
    normedbins = []
    edges=np.arange(start,end,binsize)
    num_bins=edges.shape[0]-1 #number of bins 
    ave_reach_tens=np.zeros((len(df),num_bins)) #for tensor

    for i,times in enumerate(df.times): #for each unit 
        t = np.array(times) #for reach unit create an array of that unit's spike times
        for j,tmax in enumerate(df_reaches.rMax_t): #compare that unit's spike times to each reach max
            rd = np.array(t[(t >= tmax+start) & (t <= tmax+end)]) #find if that unit spiked within designated timeframes around reachmax
            edges=np.arange(tmax+start,tmax+end,binsize) #designated bins around this iteration of reachmax
            num_bins=edges.shape[0]-1 #number of bins 
            byreach=np.zeros((len(df_reaches.rMax_t),num_bins)) #initialize or empty byreach
            hist=np.histogram(rd,edges)[0] #bin spikes into timeframe
            byreach[j,:] = hist
        meanbinfr = np.mean(byreach,axis=0)
        if sum(meanbinfr) > 0: 
            ave_reach.append(meanbinfr)
            ave_reach_tens[i,:] = meanbinfr #for tensor purposes (neural trajectories)
            normbins = (meanbinfr-min(meanbinfr))/max(meanbinfr) #per dailey
            normedbins.append(normbins)
    
    return ave_reach, normedbins, ave_reach_tens




def reachave_tensor(df,df_reaches,binsize=0.020, start = -4.0,end = 2.0):
    #START IS TIME BEFORE REACHMAX
    #END IS TIME AFTER REACHMAX -- IF BEFORE REACHMAX THEN END MUST BE NEGATIVE
    #collapses mean fr along bins
    #initialize 
    edges=np.arange(start,end,binsize)
    num_bins=edges.shape[0]-1 #number of bins 
    byreach=np.zeros((len(df_reaches.rMax_t),num_bins))
    ave_reach_=np.zeros((len(df),num_bins)) #for tensor purposes
    ave_reach = []
    normedbins = []

    for i,times in enumerate(df.times): #for each unit 
        t = np.array(times) #for reach unit create an array of that unit's spike times
        for j,tmax in enumerate(df_reaches.rMax_t): #compare that unit's spike times to each reach max
            rd = np.array(t[(t >= tmax+start) & (t <= tmax+end)]) #find if that unit spiked within designated timeframes around reachmax
            edges=np.arange(tmax+start,tmax+end,binsize) #designated bins around this iteration of reachmax
            hist=np.histogram(rd,edges)[0] #bin spikes into timeframe
            byreach[j,:] = hist
        meanbinfr = np.mean(byreach,axis=0)
        ave_reach.append(meanbinfr)
    
        ave_reach_[i,:] = meanbinfr #for tensor purposes (neural trajectories)
    
        normbins = (meanbinfr-min(meanbinfr))/max(meanbinfr) #per dailey
        normedbins.append(normbins)
    
        byreach=np.zeros((len(df_reaches.rMax_t),num_bins))
    
    #df_align['bin_ave'] = ave_reach
    #df_align['norm_bin_ave'] = normedbins
    return ave_reach, normedbins, ave_reach_


def trial_ave_heatmap_old(df,df_reaches,mouseid,binsize=0.020, start = -4.0,end = 2.0):
    #START IS TIME BEFORE REACHMAX
    #END IS TIME AFTER REACHMAX -- IF BEFORE REACHMAX THEN END MUST BE NEGATIVE
    #collapses mean fr along bins
    #initialize 
    df = df[df.mouse==mouseid]
    df_reaches=df_reaches[df_reaches.mouse==mouseid]
    print('creating heatmap for ' + str(mouseid) + '.....')

    try:
        mod_up, mod_down, maxfr_ts, minfr_ts = movement_mod(df,df_reaches)
        df['mod_up'] = mod_up
        df['mod_down'] = mod_down
        df['peak_up'] = maxfr_ts
        df['peak_down'] = minfr_ts

        print('movement modulated units found')

        edges=np.arange(start,end,binsize)
        num_bins=edges.shape[0]-1 #number of bins 
        byreach=np.zeros((len(df_reaches.rMax_t),num_bins))
        ave_reach_=np.zeros((len(df),num_bins)) #for tensor purposes
        ave_reach = []
        normedbins = []

        for i,times in enumerate(df.times): #for each unit 
            t = np.array(times) #for reach unit create an array of that unit's spike times
            for j,tmax in enumerate(df_reaches.rMax_t): #compare that unit's spike times to each reach max
                rd = np.array(t[(t >= tmax+start) & (t <= tmax+end)]) #find if that unit spiked within designated timeframes around reachmax
                edges=np.arange(tmax+start,tmax+end,binsize) #designated bins around this iteration of reachmax
                hist=np.histogram(rd,edges)[0] #bin spikes into timeframe
                byreach[j,:] = hist
            meanbinfr = np.mean(byreach,axis=0)
            ave_reach.append(meanbinfr)
        
            ave_reach_[i,:] = meanbinfr #for tensor purposes (neural trajectories)
        
            normbins = (meanbinfr-min(meanbinfr))/max(meanbinfr) #per dailey
            normedbins.append(normbins)
        
        print('plotting heatmap')

        df['binz'] = normedbins
        df_s= df.sort_values(by=['peak_up'])
        df_bins_ = df_s.binz
        df_heatmap = list(filter(any, df_bins_))
        fig, ax = plt.subplots(figsize=(20, 10))
        fig = sns.heatmap(df_heatmap)
        plt.title(str(mouseid))
        plt.show()
        plt.close()
    except:
        print('movement modulation units not found mouse ' +str(mouseid))
        pass


def countlist(lst): #to count consecutive numbers in movement-related code
    streak_count = []
    counter = 1
    for i in range(len(lst)):
        if i != (len(lst) - 1):
            diff = lst[i+1] - lst[i]
            if diff == 1:
                counter += 1
            else:
                streak_count.append(counter)
                counter = 1
        else:
            streak_count.append(counter)
    return(streak_count)



def movement_mod(df,df_reaches,startb = -1.0,endb = -0.5,starte = -0.5,ende = 0.5,binsize=0.001): #binsize 1 ms
    
    edgesb=np.arange(startb,endb,binsize)
    num_binsb=edgesb.shape[0]-1 #number of bins

    edgese=np.arange(starte,ende,binsize)
    num_binse=edgese.shape[0]-1 #number of bins

    byreach_b=np.zeros((len(df_reaches.rMax_t),num_binsb))
    byreach_e=np.zeros((len(df_reaches.rMax_t),num_binse))

    mod_up = []
    mod_down = []
    maxfr_ts = []
    minfr_ts = []
    maxfrs = []
    minfrs = []

    for i,times in enumerate(df.times): #for each unit 
        t = np.array(times) #for reach unit create an array of that unit's spike times
    
        for j,tmax in enumerate(df_reaches.rMax_t): 
        
            base = np.array(t[(t >= tmax+startb) & (t <= tmax+endb)])
            epoch = np.array(t[(t >= tmax+starte) & (t <= tmax+ende)])
        
            edgesb = np.arange(tmax+startb,tmax+endb,binsize)
            edgese = np.arange(tmax+starte,tmax+ende,binsize)
        
            histb=np.histogram(base,edgesb)[0] 
            histe=np.histogram(epoch,edgese)[0]
    
            byreach_b[j,:] = histb
            byreach_e[j,:] = histe
    
        meanbinfr_b = np.mean(byreach_b,axis=0)
        mfrb_ser = pd.Series(meanbinfr_b)        
        rolave_mfrb = mfrb_ser.rolling(100).sum() #takes sum of 100bins, shifts 1bin .. 100 bins binned at 1 ms = 100ms summed bins
        rolave_mfrb = np.array(rolave_mfrb.dropna())
        #lower,upper = sms.DescrStatsW(rolave_mfrb).tconfint_mean() #fix this
        upper = np.mean(rolave_mfrb) + (2.56*np.std(rolave_mfrb))
        lower = np.mean(rolave_mfrb) - (2.56*np.std(rolave_mfrb))

        meanbinfr_e = np.mean(byreach_e,axis=0)
        mfr_ser = pd.Series(meanbinfr_e)        
        rolave_mfr = mfr_ser.rolling(100).sum() #takes sum of 100bins, shifts 1bin 
        maxfr = np.max(rolave_mfr)/0.100 #peak firing rate
        minfr = np.min(rolave_mfr)/0.100 #peak firing rate
        maxfr_t = edgese[np.argmax(rolave_mfr)]-tmax #index of peak firing rate
        minfr_t = edgese[np.argmin(rolave_mfr)]-tmax #index of minimum firing rate
        
        maxfrs.append(maxfr)
        minfrs.append(minfr)
        maxfr_ts.append(maxfr_t)
        minfr_ts.append(minfr_t)

        mfr_a = np.array(rolave_mfr)
        up = np.where(mfr_a >= upper)[0].tolist()
        down = np.where(mfr_a <= lower)[0].tolist()

        consec_up = np.array(countlist(up))
        consec_down = np.array(countlist(down))

        if any(consec_up >= 50): #if there are 50 consecutive bins
            ups = 'yes'
            mod_up.append(ups)
        else:
            ups_ = 'no'
            mod_up.append(ups_)
    
        if any(consec_down >= 50): #if there are 50 consecutive bins
            downs = 'yes'
            mod_down.append(downs)
        else:
            downs_ = 'no'
            mod_down.append(downs_)

    return mod_up, mod_down, maxfr_ts, minfr_ts, maxfrs, minfrs


def movement_mod_heatmap(df,df_reaches,startb = -1.0,endb = -0.5,starte = -0.5,ende = 0.5,binsize=0.001): #binsize 1 ms
    
    edgesb=np.arange(startb,endb,binsize)
    num_binsb=edgesb.shape[0]-1 #number of bins

    edgese=np.arange(starte,ende,binsize)
    num_binse=edgese.shape[0]-1 #number of bins

    byreach_b=np.zeros((len(df_reaches.rMax_t),num_binsb))
    byreach_e=np.zeros((len(df_reaches.rMax_t),num_binse))

    mod_up = []
    mod_down = []
    maxfr_ts = []
    minfr_ts = []
    maxfrs = []
    minfrs = []

    for i,times in enumerate(df.times): #for each unit 
        t = np.array(times) #for reach unit create an array of that unit's spike times
    
        for j,tmax in enumerate(df_reaches.rMax_t): 
        
            base = np.array(t[(t >= tmax+startb) & (t <= tmax+endb)])
            epoch = np.array(t[(t >= tmax+starte) & (t <= tmax+ende)])
        
            edgesb = np.arange(tmax+startb,tmax+endb,binsize)
            edgese = np.arange(tmax+starte,tmax+ende,binsize)
        
            histb=np.histogram(base,edgesb)[0] 
            histe=np.histogram(epoch,edgese)[0]
    
            byreach_b[j,:] = histb
            byreach_e[j,:] = histe
    
        meanbinfr_b = np.mean(byreach_b,axis=0)
        mfrb_ser = pd.Series(meanbinfr_b)        
        rolave_mfrb = mfrb_ser.rolling(100).sum() #takes sum of 100bins, shifts 1bin .. 100 bins binned at 1 ms = 100ms summed bins
        rolave_mfrb = np.array(rolave_mfrb.dropna())
        #lower,upper = sms.DescrStatsW(rolave_mfrb).tconfint_mean() #fix this
        upper = np.mean(rolave_mfrb) + (2.56*np.std(rolave_mfrb))
        lower = np.mean(rolave_mfrb) - (2.56*np.std(rolave_mfrb))

        meanbinfr_e = np.mean(byreach_e,axis=0)
        mfr_ser = pd.Series(meanbinfr_e)        
        rolave_mfr = mfr_ser.rolling(100).sum() #takes sum of 100bins, shifts 1bin 
        #maxfr = np.max(rolave_mfr)/0.100 #peak firing rate
        #minfr = np.min(rolave_mfr)/0.100 #peak firing rate
        maxfr_t = edgese[np.argmax(rolave_mfr)]-tmax #index of peak firing rate
        minfr_t = edgese[np.argmin(rolave_mfr)]-tmax #index of minimum firing rate
        
        #maxfrs.append(maxfr)
        #minfrs.append(minfr)
        maxfr_ts.append(maxfr_t)
        minfr_ts.append(minfr_t)

        mfr_a = np.array(rolave_mfr)
        up = np.where(mfr_a >= upper)[0].tolist()
        down = np.where(mfr_a <= lower)[0].tolist()

        consec_up = np.array(countlist(up))
        consec_down = np.array(countlist(down))

        if any(consec_up >= 50): #if there are 50 consecutive bins
            ups = 'yes'
            mod_up.append(ups)
        else:
            ups_ = 'no'
            mod_up.append(ups_)
    
        if any(consec_down >= 50): #if there are 50 consecutive bins
            downs = 'yes'
            mod_down.append(downs)
        else:
            downs_ = 'no'
            mod_down.append(downs_)

    return mod_up, mod_down, maxfr_ts, minfr_ts



#mean removes and normalizes to std
def center(X):
    # X: ndarray, shape (n_features, n_samples)
    ss = StandardScaler(with_mean=True, with_std=True)
    Xc = ss.fit_transform(X.T).T
    return Xc



def pca(tens):
    for i in range(min(tens.shape[0], tens.shape[1])-1):
        pca = PCA(n_components=i)
        p=pca.fit_transform(tens.T).T
        ev = (pca.explained_variance_ / sum(pca.explained_variance_))
    return p,ev
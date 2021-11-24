import pandas as pd
import numpy as np
import glob
from neuropixels import generalephys as ephys
from neuropixels import generalephys_mua as ephys_mua
from neuropixels.generalephys import get_waveform_duration,get_waveform_PTratio,get_waveform_repolarizationslope,option234_positions
from scipy.cluster.vq import kmeans2
from neuropixels import sorting_quality_editing as sq
import seaborn as sns;sns.set_style("ticks")
import matplotlib.pyplot as plt
import h5py
import matplotlib.path as mpath
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import os,time



### utility code for loading kilosort outputs ####
#### adapted from dlab: ####

def df_from_phy_multimouse(folder,expnum='1',recnum='1',**kwargs):
    # list all animals within folder, iterate through each animal and get cluster information
    if 'est' not in folder:
        base_folder = os.path.basename(folder)
        cohort_ = os.path.basename(base_folder).split('_')[-2]
        mouse_  = os.path.basename(base_folder).split('_')[-1]
        #traverse down tree to data
        if 'open-ephys-neuropix' in base_folder:
            try:
                rec_folder = glob.glob(folder+'/*')[0]
            except:
                pass
    #        else:
    #            rec_folder = folder
    #            print(rec_folder)
        raw_path = os.path.join(rec_folder,'experiment1','recording'+str(recnum),'continuous')
        if len(glob.glob(raw_path+'/*100.0*'))>0:
            raw_path = glob.glob(raw_path+'/*100.0*')[0]
            print('loading from '+raw_path)
        else:
            print('could not find data folder for '+raw_path)
    if 'cohort' in kwargs.keys():
        cohort = kwargs['cohort']
    else:
        cohort = None 
    if 'mouse' in kwargs.keys():
        mouse = kwargs['mouse']
    else:
        mouse = None              
    # df = df_from_phy(raw_path,site_positions = ephys.option234_positions,cluster_file='KS2',cohort=cohort,mouse=mouse)
    
    path = raw_path
    #units = ephys.load_phy_template(path,cluster_file='KS2',site_positions=site_positions)
    if os.path.isfile(os.path.join(raw_path,'spike_clusters.npy')) :                  
        units = ephys.load_phy_template(path)
        #structures is a dictionary that defines the bounds of the structure e.g.:{'v1':(0,850), 'hpc':(850,2000)}
        mouse = [];experiment=[];cell = [];ypos = [];xpos = [];waveform=[];template=[];structure=[];times=[]
        index = []; count = 1; cohort = []
        probe_id=[]
        depth=[];#print(list(nwb_data.keys()));print(list(nwb_data['processing'].keys()));
        if os.path.isfile(os.path.join(raw_path,'cluster_group.tsv')) : 
            for unit in units.index:
                if 'probe' in kwargs.keys():
                    probe_id.extend([kwargs['probe']])
                else:
                    probe_id.extend(['A'])
                if 'mouse' in kwargs.keys():
                    mouse.extend([kwargs['mouse']])
                else:
                    mouse.extend([int(mouse_)])
                if 'experiment' in kwargs.keys():
                    experiment.extend([kwargs['experiment']])
                else:
                    experiment.extend(['placeholder'])
                if 'cohort' in kwargs.keys():
                    cohort.extend([kwargs['cohort']])
                else:
                    cohort.extend([cohort_])

            df = units
            df['mouse'] = mouse
            df['experiment'] = experiment
            df['probe'] = probe_id
            #     df['structure'] = structure
            df['cell'] = units.ClusterID
            df['cohort'] = cohort
            df['times'] = units['times']
            df['ypos'] = units['ypos']
            df['xpos'] = units['xpos']
            #         df['depth'] = xpos
            df['waveform'] = units['waveform_weights']
            df['template'] = units['template']
        
            return df
    
        else:
            pass




# to load a single recording file from one recording, one mouse
def df_from_phy(folder,expnum='1',recnum='1',**kwargs):
    # if 'est' not in folder:
 #       base_folder = os.path.basename(folder)
 #       cohort_ = os.path.basename(base_folder).split('_')[-2]
 #       mouse_  = os.path.basename(base_folder).split('_')[-1]

        #traverse down tree to data
 #       if 'open-ephys-neuropix' in base_folder:
 #           try:
 #               rec_folder = glob.glob(folder+'/*')[0]
 #           except:
 #               print(base_folder)
 #               return None
 #       else:
 #           rec_folder = folder
 #       raw_path = os.path.join(rec_folder,'experiment'+str(expnum),'recording'+str(recnum),'continuous')
 #       if len(glob.glob(raw_path+'/*100.0*'))>0:
 #           raw_path = glob.glob(raw_path+'/*100.0*')[0]
 #           print('loading from '+raw_path)
 #       else:
            
 #           print('could not find data folder for '+raw_path)
    raw_path=folder
 
    if 'cohort' in kwargs.keys():
        cohort = kwargs['cohort']
    else:
        cohort = None 
    if 'mouse' in kwargs.keys():
        mouse = kwargs['mouse']
    else:
        mouse = None              
    # df = df_from_phy(raw_path,site_positions = ephys.option234_positions,cluster_file='KS2',cohort=cohort,mouse=mouse)
    
    path = raw_path
    #units = ephys.load_phy_template(path,cluster_file='KS2',site_positions=site_positions)
    units = ephys.load_phy_template(path,site_positions=site_positions)
    #structures is a dictionary that defines the bounds of the structure e.g.:{'v1':(0,850), 'hpc':(850,2000)}
    mouse = [];experiment=[];cell = [];ypos = [];xpos = [];waveform=[];template=[];structure=[];times=[]
    index = []; count = 1; cohort = []
    probe_id=[]
    depth=[];#print(list(nwb_data.keys()));print(list(nwb_data['processing'].keys()));

    for unit in units.index:
        if 'probe' in kwargs.keys():
            probe_id.extend([kwargs['probe']])
        else:
            probe_id.extend(['A'])
        if 'mouse' in kwargs.keys():
            mouse.extend([kwargs['mouse']])
        else:
            mouse.extend([mouse_])
        if 'experiment' in kwargs.keys():
            experiment.extend([kwargs['experiment']])
        else:
            experiment.extend(['placeholder'])
        if 'cohort' in kwargs.keys():
            cohort.extend([kwargs['cohort']])
        else:
            cohort.extend([cohort_])

    df = units
    df['mouse'] = mouse
    df['experiment'] = experiment
    df['probe'] = probe_id
    #     df['structure'] = structure
    df['cell'] = units.index
    df['cohort'] = cohort
    df['times'] = units['times']
    df['ypos'] = units['ypos']
    df['xpos'] = units['xpos']
    #         df['depth'] = xpos
    df['waveform'] = units['waveform_weights']
    df['template'] = units['template']
    return df






### utility code for classifying waveforms

def get_peak_waveform_from_template(template):
    max = 0
    ind=0
    peak = np.zeros(np.shape(template.T)[0])
    for i,wv in enumerate(template.T):
        if np.max(np.abs(wv)) > max:
            max = np.max(np.abs(wv))
            ind = i
            peak = wv
    return peak


def classify_waveform_shape(df,plots=False,save_plots=False,basepath='',kmeans=0):
    durations = np.zeros(np.shape(df)[0])
    PTratio = np.zeros(np.shape(df)[0])
    repolarizationslope = np.zeros(np.shape(df)[0])
    for i,waveform in enumerate(df.waveform):
        # try:
        durations[i]=get_waveform_duration(waveform)
        PTratio[i]=get_waveform_PTratio(waveform)
        repolarizationslope[i]=get_waveform_repolarizationslope(waveform,window=18)
        # except:
        #     durations[i]=np.nan
        #     PTratio[i]=np.nan
        #     repolarizationslope[i]=np.nan
    df['waveform_duration'] = durations
    df['waveform_PTratio'] = PTratio
    df['waveform_repolarizationslope'] = repolarizationslope

    waveform_k = kmeans2(np.vstack(((durations-np.min(durations))/np.max((durations-np.min(durations))),
                                    (PTratio-np.min(PTratio))/np.max((PTratio-np.min(PTratio))),
                                    (repolarizationslope-np.min(repolarizationslope))/np.max((repolarizationslope-np.min(repolarizationslope))))).T,
                            2, iter=300, thresh=5e-6,minit='points')
    # waveform_k = kmeans2(np.vstack((durations/np.max(durations),PTratio/np.max(PTratio))).T, 2, iter=300, thresh=5e-6,minit='points')
    # waveform_k = kmeans2(np.vstack((durations/np.max(durations),(repolarizationslope-np.min(repolarizationslope))/np.max(repolarizationslope))).T, 2, iter=900, thresh=5e-7,minit='points')
    
    #assign fs and rs to the kmeans results
    if np.mean(durations[np.where(waveform_k[1]==0)[0]]) < np.mean(durations[np.where(waveform_k[1]==1)[0]]):
        fs_k = 0;rs_k = 1
        waveform_class_ids = ['fs','rs']
    else:
        rs_k = 0;fs_k = 1
        waveform_class_ids = ['rs','fs']
    waveform_class = [waveform_class_ids[k] for k in waveform_k[1]]

    #uncomment this to ignore the preceding kmeans and just split on the marginal distribution of durations
    if kmeans==0:
        waveform_class = ['fs' if duration < 0.0004 else 'rs' for i,duration in enumerate(durations) ]
    else:
        waveform_k = kmeans2(np.vstack(((durations-np.min(durations))/np.max((durations-np.min(durations))),
                                        (PTratio-np.min(PTratio))/np.max((PTratio-np.min(PTratio))),
                                        (repolarizationslope-np.min(repolarizationslope))/np.max((repolarizationslope-np.min(repolarizationslope))))).T,
                                kmeans, iter=300, thresh=5e-6,minit='points')
        # waveform_k = kmeans2(np.vstack((durations/np.max(durations),PTratio/np.max(PTratio))).T, 2, iter=300, thresh=5e-6,minit='points')
        # waveform_k = kmeans2(np.vstack((durations/np.max(durations),(repolarizationslope-np.min(repolarizationslope))/np.max(repolarizationslope))).T, 2, iter=900, thresh=5e-7,minit='points')
        
        #assign fs and rs to the kmeans results
        if np.mean(durations[np.where(waveform_k[1]==0)[0]]) < np.mean(durations[np.where(waveform_k[1]==1)[0]]):
            fs_k = 0;rs_k = 1
            waveform_class_ids = ['fs','rs']
        else:
            rs_k = 0;fs_k = 1
            waveform_class_ids = ['rs','fs']
        waveform_class = [waveform_class_ids[k] for k in waveform_k[1]]

    #force upwards spikes to have the own class, because we're not sure how they fit in this framework
    waveform_class = [waveform_class[i] if ratio < 1.0 else 'up' for i,ratio in enumerate(PTratio) ]
    df['waveform_class']=waveform_class

    #mark narrow upwards spikes as axons
    waveform_class = ['axon' if all([duration < 0.0004,waveform_class[i]=='up']) else waveform_class[i] for i,duration in enumerate(durations) ]
    df['waveform_class']=waveform_class

    # #mark narrow downward spike at the very bottom of cortex as axons
    #waveform_class = ['axon' if all([duration < 0.0004,waveform_class[i]=='fs',df['depth'][i+1] > 750, df['depth'][i+1]<1050]) else waveform_class[i] for i,duration in enumerate(durations) ]
    df['waveform_class']=waveform_class

    if plots:
        plot_waveform_classification(durations, PTratio, repolarizationslope,df,save_plots=save_plots,basepath=basepath)
    return df

def plot_waveform_classification(durations, PTratio, repolarizationslope, df,save_plots=False, basepath=''):
    f,ax = plt.subplots(1,3,figsize=(8,3))
    ax[0].plot(durations[np.where(df.waveform_class=='rs')[0]],PTratio[np.where(df.waveform_class=='rs')[0]],'o',ms=3.2)
    ax[0].plot(durations[np.where(df.waveform_class=='fs')[0]],PTratio[np.where(df.waveform_class=='fs')[0]],'o',ms=3.2)
    #ax[0].plot(durations[np.where(df.waveform_class=='up')[0]],PTratio[np.where(df.waveform_class=='up')[0]],'o',ms=3.2)
    ax[0].plot(durations[np.where(df.waveform_class=='axon')[0]],PTratio[np.where(df.waveform_class=='axon')[0]],'o',ms=3.2)
    ax[0].set_xlabel('width (sec)')
    ax[0].set_ylabel('peak/trough ratio')
    ax[1].plot(durations[np.where(df.waveform_class=='rs')[0]],repolarizationslope[np.where(df.waveform_class=='rs')[0]],'o',ms=3.2)
    ax[1].plot(durations[np.where(df.waveform_class=='fs')[0]],repolarizationslope[np.where(df.waveform_class=='fs')[0]],'o',ms=3.2)
    #ax[1].plot(durations[np.where(df.waveform_class=='up')[0]],repolarizationslope[np.where(df.waveform_class=='up')[0]],'o',ms=3.2)
    ax[1].plot(durations[np.where(df.waveform_class=='axon')[0]],repolarizationslope[np.where(df.waveform_class=='axon')[0]],'o',ms=3.2)
    ax[1].set_xlabel('width (sec)')
    ax[1].set_ylabel('repolarization slope')
    ax[2].plot(PTratio[np.where(df.waveform_class=='rs')[0]],repolarizationslope[np.where(df.waveform_class=='rs')[0]],'o',ms=3.2)
    ax[2].plot(PTratio[np.where(df.waveform_class=='fs')[0]],repolarizationslope[np.where(df.waveform_class=='fs')[0]],'o',ms=3.2)
    #ax[2].plot(PTratio[np.where(df.waveform_class=='up')[0]],repolarizationslope[np.where(df.waveform_class=='up')[0]],'o',ms=3.2)
    ax[2].plot(PTratio[np.where(df.waveform_class=='axon')[0]],repolarizationslope[np.where(df.waveform_class=='axon')[0]],'o',ms=3.2)
    ax[2].set_ylabel('repolarization slope')
    ax[2].set_xlabel('peak/trough ratio')
    ax[0].set_xlim(0.0,0.0015);ax[1].set_xlim(0.0,0.0015)
    ax[0].set_ylim(0,1.1);ax[2].set_xlim(0,1.1)
    plt.tight_layout()
    for axis in ax:
    #    ephys.cleanAxes(axis,bottomLabels=True,leftLabels=True)
        axis.locator_params(axis='x',nbins=4)
    ax[2].legend(loc='upper right')
    panelname = 'waveforms_clusters'
    plt.tight_layout()
    if save_plots:
        plt.gcf().savefig(os.path.join(basepath,'figures','panels',panelname+'.png'),fmt='png',dpi=300)
        plt.gcf().savefig(os.path.join(basepath,'figures','panels',panelname+'.eps'),fmt='eps')
        
    nbins = 36
    plt.hist(durations[np.where(df.waveform_class=='rs')[0]],range=(0,0.0015),bins=nbins)
    plt.hist(durations[np.where(df.waveform_class=='fs')[0]],range=(0,0.0015),bins=nbins)
    plt.hist(durations[np.where(df.waveform_class=='axon')[0]],range=(0,0.0015),bins=nbins)
    plt.figure()
    plt.hist((durations[np.where(df.waveform_class=='rs')[0]],durations[np.where(df.waveform_class=='fs')[0]],durations[np.where(df.waveform_class=='axon')[0]]),range=(0,0.0015),bins=nbins,stacked=True)
    #ephys.cleanAxes(plt.gca(),bottomLabels=True,leftLabels=True)
    plt.xlabel('waveform duration (sec)')
    plt.ylabel('neuron count')
    panelname = 'waveforms_durationhistogram'
    plt.tight_layout()
    if save_plots:
        plt.gcf().savefig(os.path.join(basepath,'figures','panels',panelname+'.png'),fmt='png',dpi=300)
        plt.gcf().savefig(os.path.join(basepath,'figures','panels',panelname+'.eps'),fmt='eps')
    
    plt.figure(figsize=(4,3))
    
    waveform_time = np.linspace(-1*np.where(df.waveform[1] > 0.)[0][0]/30000.,(len(df.waveform[1])-np.where(df.waveform[1] > 0.)[0][0])/30000.,len(df.waveform[1]))*1000
    #plot all
    for i,waveform in enumerate(df.waveform):
        #waveform_time = np.linspace(0,len(waveform)/30000.,len(waveform))*1000
        if df.waveform_class[i]=='rs':
            plt.plot(waveform_time,waveform/np.max(np.abs(waveform)),color=sns.color_palette()[0],alpha=0.01)
        if df.waveform_class[i]=='axon':#df.waveform_class.unique()[np.where(df.waveform_class=='axon')[0]]:
            plt.plot(waveform_time,waveform/np.max(np.abs(waveform)),color=sns.color_palette()[2],alpha=0.01)
        if df.waveform_class[i]=='fs':#df.waveform_class.unique()[np.where(df.waveform_class=='fs')[0]]:
            plt.plot(waveform_time,waveform/np.max(np.abs(waveform)),color=sns.color_palette()[1],alpha=0.01)
    # plot means, normalized
    for waveform_class in ['rs','fs','axon']:#df.waveform_class.unique():
        if waveform_class != 'up' and waveform_class!='axon':
            plt.plot(waveform_time,np.mean(df.waveform[df.waveform_class==waveform_class])/(np.max(np.abs(np.mean(df.waveform[df.waveform_class==waveform_class])))),lw=4)
        #plt.plot(waveform_time,np.mean(df.waveform[df.waveform_class==waveform_class])/(np.max(np.abs(np.mean(df.waveform[df.waveform_class==waveform_class])))),lw=2)
    # plt.plot(waveform_time,np.mean(df.waveform[df.waveform_class=='rs'])/(np.min(np.mean(df.waveform[df.waveform_class=='rs']))*-1),lw=2)
    # plt.plot(waveform_time,np.mean(df.waveform[df.waveform_class=='up'])/(np.max(np.mean(df.waveform[df.waveform_class=='up']))),lw=2)
    
    plt.title('RS: '+str(len(df.waveform_class[df.waveform_class=='rs']))+
                '   FS: '+str(len(df.waveform_class[df.waveform_class=='fs']))+
              '   axon: '+str(len(df.waveform_class[df.waveform_class=='axon'])))#+
    #            '   up:'+str(len(df.waveform_class[df.waveform_class=='up'])))
    
    
    plt.gca().set_xlim(-1.,1.4)
    plt.gca().legend(loc='upper left')
    #ephys.cleanAxes(plt.gca(),leftLabels=True,bottomLabels=True)
    plt.gca().set_ylabel('normalized amplitude',size=10)
    d=plt.gca().set_xlabel('time (msec)',size=10)
    panelname = 'waveforms_mean_peak'
    plt.tight_layout()
    if save_plots:
        plt.gcf().savefig(os.path.join(basepath,'figures','panels',panelname+'.png'),fmt='png',dpi=300)
        plt.gcf().savefig(os.path.join(basepath,'figures','panels',panelname+'.eps'),fmt='eps')





### utility code for depth estimation ###


def plot_results(chunk, 
                 power, 
                 in_range, 
                 values, 
                 nchannels, 
                 surface_chan, 
                 power_thresh, 
                 diff_thresh, 
                 figure_location):

    plt.figure(figsize=(5,10))
    plt.subplot(4,1,1)
    # plt.imshow(np.flipud((chunk).T), aspect='auto',vmin=-1000,vmax=1000)
    plt.imshow((chunk).T, aspect='auto',vmin=-1000,vmax=1000)
    plt.title('raw data (the chunk)')
    plt.xlabel('time (samples)')
    plt.ylabel('channels')
    

    plt.subplot(4,1,2)
    # plt.imshow(np.flipud(np.log10(power[in_range,:]).T), aspect='auto')
    plt.imshow(np.log10(power[in_range,:]).T, aspect='auto')
    plt.title('gamma power')
    plt.xlabel('time (samples)')
    plt.ylabel('channels')

    plt.subplot(4,1,3)
    plt.plot(values) 
    plt.plot([0,nchannels],[power_thresh,power_thresh],'--k') #kg input -0.1 for y axis
    plt.plot([surface_chan, surface_chan],[0.5, 2],'--r')
    plt.xlabel('channels')
    plt.ylabel('gamma power')
    
    plt.subplot(4,1,4)
    plt.plot(np.diff(values))
    plt.plot([0,nchannels],[diff_thresh,diff_thresh],'--k') #kg input -0.1 for y axis
    plt.plot([surface_chan, surface_chan],[diff_thresh, diff_thresh],'--r')
    plt.title(surface_chan)
    plt.xlabel('channels')
    plt.ylabel('power difference between channels')

    plt.show()
    plt.savefig(figure_location)
    plt.close()


# ===========================================================================
    ### utility code for batch sorting quality ###


def sorting_quality_multimouse(folder,expnum='1',recnum='1',channels=383,**kwargs):
    # list all animals within folder, iterate through each animal and get cluster information
    if 'est' not in folder:
        base_folder = os.path.basename(folder)
        cohort_ = os.path.basename(base_folder).split('_')[-2]
        mouse_  = os.path.basename(base_folder).split('_')[-1]
        #traverse down tree to data
        if 'open-ephys-neuropix' in base_folder:
            try:
                rec_folder = glob.glob(folder+'/*')[0]
            except:
                pass
    #        else:
    #            rec_folder = folder
    #            print(rec_folder)
        raw_path = os.path.join(rec_folder,'experiment1','recording'+str(recnum),'continuous')
        if len(glob.glob(raw_path+'/*100.0*'))>0:
            raw_path = glob.glob(raw_path+'/*100.0*')[0]
            print('loading from '+raw_path)
        else:
            print('could not find data folder for '+raw_path)
    if 'cohort' in kwargs.keys():
        cohort = kwargs['cohort']
    else:
        cohort = None 
    if 'mouse' in kwargs.keys():
        mouse = kwargs['mouse']
    else:
        mouse = None              

    directory = raw_path
    if os.path.isfile(os.path.join(raw_path,'cluster_group.tsv')) :
        channels2 = 'all'#[0,383]
        print(channels2)
        time_limits = None#[500.,600.]

        t0 = time.time()
        #quality = sq.masked_cluster_quality(directory,time_limits)
        #print('PCA quality took '+str(time.time()-t0)+' sec');t0 = time.time()
        isiV = sq.isiViolations(directory,time_limits)
        print('ISI quality took '+str(time.time()-t0)+' sec');t0 = time.time(); 
        #SN = sq.cluster_signalToNoise(directory,filename,time_limits, channels = 383)
        #print('SN quality took '+str(time.time()-t0)+' sec');t0 = time.time()

        cluster_groups = sq.read_cluster_groups_CSV(directory)  
        print(cluster_groups[2])

        print(isiV[0])

        cluster_group = []
        color = []
        for clu_id in isiV[0]:
            if clu_id in cluster_groups[0]:
                cluster_group.append('good')
                color.append(sns.color_palette()[1])
            else:
                if clu_id in cluster_groups[1]:
                    cluster_group.append('mua')
                    color.append(sns.color_palette()[0])
                else:
                    if clu_id in cluster_groups[2]:
                        cluster_group.append('unsorted')
                        color.append(sns.color_palette()[1])
                    else:
                        cluster_group.append('noise')
                        color.append(sns.color_palette()[1])
        mouse = []; cohort = []; trash = []; 
        for unit in cluster_group:
            if 'mouse' in kwargs.keys():
                mouse.extend([kwargs['mouse']])
            else:
                mouse.extend([int(mouse_)])
            if 'cohort' in kwargs.keys():
                cohort.extend([kwargs['cohort']])
            else:
                cohort.extend([cohort_])

        df = pd.DataFrame({
            'clusterID':isiV[0],
            'isi_violations':np.ones(len(isiV[1])) - isiV[1],
#            'sn_max':SN[1],
#            'sn_mean':SN[2],
#            'isolation_distance':quality[1],
#            'mahalanobis_contamination':np.ones(len(quality[2]))-quality[2],
#            'FLDA_dprime':quality[3]*-1,
            'cluster_group':cluster_group,
            'color':color})
        df['mouse'] = mouse
        df['cohort'] = cohort
        return df

    else:
        pass



def assign_sq_rank(df_sq):
    qual_rank = []

    for i in df_sq.linear_quality:
        if i>1.5: # i > 1.5
            qual_rank.append(1) 
        if i>1 and i<1.5: # 1 < i < 1.5
            qual_rank.append(2)
        if i > 0.5 and i < 1:
            qual_rank.append(3)
        if i > 0 and i < 0.5:
            qual_rank.append(4)
        if i < 0 and i > -0.5:
            qual_rank.append(5)
        if i < -0.5 and i > -1:
            qual_rank.append(6)
        if i <-1 and i > -1.5:
            qual_rank.append(7)
        if i <-1.5:
            qual_rank.append(8)
    df_sq['quality_rank'] = qual_rank
    return df_sq

def save_individual_mouse_pairplot(folder,df,df_sq,expnum='1',recnum='1',channels=383,**kwargs):
    # list all animals within folder, iterate through each animal and get cluster information
    if 'est' not in folder:
        base_folder = os.path.basename(folder)
        cohort_ = os.path.basename(base_folder).split('_')[-2]
        mouse_  = os.path.basename(base_folder).split('_')[-1]
        #traverse down tree to data
        if 'open-ephys-neuropix' in base_folder:
            try:
                rec_folder = glob.glob(folder+'/*')[0]
            except:
                pass
    #        else:
    #            rec_folder = folder
    #            print(rec_folder)
        raw_path = os.path.join(rec_folder,'experiment1','recording'+str(recnum),'continuous')
        if len(glob.glob(raw_path+'/*100.0*'))>0:
            raw_path = glob.glob(raw_path+'/*100.0*')[0]
            print('saving figure in '+raw_path)
        else:
            print('could not find data folder for '+raw_path)

    directory = raw_path
    df_sq = df_sq[df_sq.mouse==int(mouse_)]
    df_ = df[df.mouse==int(mouse_)]
    if os.path.isfile(os.path.join(raw_path,'cluster_group.tsv')) :
        fig = sns.pairplot(df_sq,
            diag_kind='kde',markers='o',hue='cluster_group')
        plt.title('sorting quality metric distributions for mouse' +str(mouse_))
        plt.show()
        fig.savefig(os.path.join(raw_path,'sorting_quality_metrics_distribution.png'))
        plt.close()

        fig = sns.jointplot(x = df_.overall_rate,y = df_.depth, hue = df_.cluster_group
           )
        plt.title('firing rate vs depth')
        plt.ylabel('depth (0 = deep)')
        plt.xlabel('firing rate')
        plt.show()
        plt.close()

        sns.catplot(data=df_, kind="swarm", x="waveform_class", y="overall_rate", hue="cluster_group")

        fig = sns.jointplot(x = df_.linear_quality,y = df_.depth, hue = df_.cluster_group)


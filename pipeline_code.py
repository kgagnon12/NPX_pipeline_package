import numpy as np
import matplotlib.pyplot as plt
import _pickle as pkl
import pandas as pd
import os,sys,glob, h5py, csv, time
import matplotlib.pyplot as plt
from neuropixels import utils




# get info for units in df
def get_unit_info(df):
# get waveform
    wave_ = []
    for i,template in df.template.items():
        wave_.append(utils.get_peak_waveform_from_template(np.array(template)))
    df.waveform=wave_

# get firing rate
    f_ = []
    for i,times in df.times.items():
        try:
            rate = float(len(times)/(times[-1] - times[0]))
            if rate < 400:
                f_.append(rate)
            else:
                f_.append(0.)
        except:
            f_.append(0.)
    df['overall_rate']=f_

    return df




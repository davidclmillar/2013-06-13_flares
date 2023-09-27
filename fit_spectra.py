import numpy as np
from helita.io import lp
import os
import kappa_fitting as k
import sstanalysis as sst
import pickle
import concurrent.futures
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

line='8542' # or 5250

wl0, wln = 4,7 # wavelengths to extract between
t0, tn   = 0,111 # time values to extract between

# read in data 
data, sp, C = sst.read_in_data(line) # read in intensity data

# or you can read in velocity results
#data = pickle.load(open('velocities.p','rb'))

# ----- mask ----- #
# this determines which pixels are useful (some don't contain any data)
# if you are using velocity data from get_velocities.py this will already have been done
mask = sst.get_mask()
mask_first=mask[:,:,0 ]
mask_last =mask[:,:,-1]
supermask = mask_last*mask_first
# ---------------- #
args = np.where(supermask==1) # the indices of target pixels


# get no cores
no_cores = os.cpu_count() - 2

# total no pixels
total = len(args[0])
#total = len(data[0]) # this is better if you loaded velocity results

# this list will have all of the minimum values that each core will begin at, once the array is flattened 
mins = np.zeros(shape=no_cores+1)
i = 0
for a in range(1,no_cores):
    if i < (total % no_cores):
        mins[a] = mins[a-1] + np.ceil(total/no_cores)
        i+=1
    else:
        mins[a] = mins[a-1] + np.floor(total/no_cores)
mins[-1] = total
mins = mins.astype(int)

# function to be used for each timeseries
def fit(I):
      # try fitting each spectrum model individually. failure tracked by filling with 1e10
      # model parameters are returned one per model, WRSes returned as a single object
    x = k.timeseries(I,dt=30.7)
    WRS_results = np.zeros(shape=(3))
    try:
        result = x.fit_M1(plot=False)
        model1_parameters = result[0]
        WRS_results[0] =  result[1]
    except:
        model1_parameters, WRS_results[0] = np.full(3,1e10), 1e10
    try:
        result = x.fit_M2(plot=False)
        model2_parameters = result[0]
        WRS_results[1] =  result[1]
    except:
        model2_parameters, WRS_results[1] = np.full(6,1e10), 1e10

    try:
        result = x.fit_kappa(plot=False)
        modelk_parameters = result[0]
        WRS_results[2] =  result[1]
    except:
        modelk_parameters, WRS_results[2] = np.full(6,1e10), 1e10

    return model1_parameters, model2_parameters, modelk_parameters, WRS_results

# depending on if you are doing intensity or velocity this bit will need changed
def function(mini,maxi):
    res = []
    for i in range(maxi-mini):
        # for intensities
        #row, col = args[0][i+mini], args[1][i+mini]
        #I_s = np.sum(data[row, col, t0:tn, 0, wl0:wln],axis=-1)
        # --------------------------------------------------------
        
        # for velocities
        #I_s = data[1][i+mini][0][t0:tn]# for velocities, the 0 index is the velocities
        # --------------------------------------------------------
        
        # for either
        res.append(fit(I_s)) # do the fitting
    return res


# parallel processing
with concurrent.futures.ProcessPoolExecutor(max_workers=no_cores) as executor:
    futures=[executor.submit(function, mins[j], mins[j+1]) for j in range(len(mins)-1)]
    for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        pass #This gets a progress bar

results=[res.result() for res in futures]

# indices (pixel locations)
argy = [(args[0][a],args[1][a]) for a in range(len(args[0]))]

# combining all the results
bigr = []
[bigr.extend(r) for r in results]

# save the results
save_this = [argy,bigr]
savepath='/where/to/save/'
pickle.dump(save_this, open(savepath+'fit_results.p',"wb"))

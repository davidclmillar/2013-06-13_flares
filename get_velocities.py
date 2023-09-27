import numpy as np
from helita.io import lp
import time
import os
import kappa_fitting as k
import sstanalysis as sst
import pickle
import concurrent.futures
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

start_time = time.time() # simple timer

# read in data
data, sp, C = sst.read_in_data('8542') #or '5250'

# ----- mask ----- #
# this prevents us analysing pixels which don't contain any actual data
mask = sst.get_mask()
supermask = mask[:,:,0]*mask[:,:,-1]
# ---------------- #

args = np.where(supermask==1) # gets the indices of target pixels

total = len(args[0]) # total no pixels

no_cores = os.cpu_count() - 1 # set number of cores to use for processing

# this splits up all of the target pixels so each core works on a roughly equal amount
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

# settings for the velocity measurement
wl0=8542 # or 5250
wlvs = sp[:-1] - wl0 # extract relevant part of wavelength array (use all of sp for 5250)
left, right = 3, 8 # the window the first fit is done over
# 4/7 for 5250. use 3/8 for 8542
xvals = wlvs[left:right] 
T=data.shape[2] # number of timesteps

# function for doing all velocities for one pixel
def get_vs(row,col,T=T):
    ans0=0 # tracks the last answer (starts at 0)
    result = np.zeros(shape=(T))
    dodge  = np.zeros(shape=(T)) # tracks how many non-standard line shapes encountered for each pixel
    for i in range(0,T):
        fail = False
        profi = data[row,col,i,0,:-1] # whole line profile
        profi=profi/np.max(np.abs(profi)) # normalise profile
        prof = profi[left:right] # extract the part for fitting
        r = np.polyfit(xvals,prof,2) # can use curve_fit if you like
        ans = -r[1]/2/r[0] # x0=-b/2a
        fitted = r[0]*xvals**2 + r[1]*xvals + r[2]
        chi2 = np.sum((fitted - prof)**2) # not really chi2, but similar
        # conditions for good fit
        if chi2 > 2e-3:
            fail = True
        if (ans > 0.2): # use 0.2 for 8542/0.1 for 5250
            fail = True
        if (ans < -0.2): # 0.2 for 8542/0.1 for 5250
            fail = True
        # if not a good fit, try again at a turning point
        if fail:
            fail=False
            dodge[i]=1
            dp = np.diff(profi)
            tps = np.where(dp[:-1]*dp[1:] < 0)[0]+1 # turning points
            if len(tps)==0:
                fail=True
                cent=0
                ans=1
            else:
                cent = tps[np.argmin(np.abs(wlvs[tps]-ans0))]

            if cent<1 or cent>10:
                ans=1.
            else:
                  # trying another fit
                newprof = profi[cent-1:cent+2]
                newxvals = wlvs[cent-1:cent+2]
                
                r = np.polyfit(newxvals,newprof,2)
                ans = -r[1]/2/r[0]
                
                newfitted = r[0]*xvals**2 + r[1]*xvals + r[2]
                newchi2 = np.sum((fitted - prof)**2)
                
                if newchi2 > 1e-3:
                    ans=1

            if (ans > 0.5): # 0.5 for 8542/0.1 for 5250
                fail = True
            if (ans < -0.5): # 0.5 for 8542/0.1 for 5250
                fail = True
            if fail:
                ans=ans0 # if the process didn't work, set to the same value as last
                
        result[i] =ans # save result as we go
        ans0=ans
    return result, dodge

# this is a function which each core will run simultaneously
def function(mini,maxi):
    res = []
    for i in range(maxi-mini):
        row, col = args[0][i+mini], args[1][i+mini]
        res.append(get_vs(row,col))
    return res

# parallel computation
with concurrent.futures.ProcessPoolExecutor(max_workers=no_cores) as executor:
    futures=[executor.submit(function, mins[j], mins[j+1]) for j in range(len(mins)-1)]
    for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        pass #This gets a progress bar

# get all results
results=[res.result() for res in futures]

# the indices (tracking which pixel is which)
argy = [(args[0][a],args[1][a]) for a in range(len(args[0]))]

#combine all results into one object
bigr = []
[bigr.extend(r) for r in results]

#---------- save results -----------------
save_this = [argy,bigr] #save indices and results together
savepath = 'your/path/'
if not os.path.exists(savepath):
    os.makedirs(savepath)
pickle.dump(save_this, open(savepath+'velocities.p',"wb"))

print('Completed after %.2f minutes'%((time.time()-start_time)/60))

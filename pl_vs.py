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

# read in data
data, sp, C = sst.read_in_data('8542')
#sp is the wavelength array


# ----- mask ----- #
mask = sst.get_mask()
supermask = mask[:,:,0]*mask[:,:,-1]
# ---------------- #

args = np.where(supermask==1)

# get no cores
no_cores = os.cpu_count() - 1
print('no cores : {}'.format(no_cores))
# total no pixels
total = len(args[0])
print('total pixels : {}'.format(total))

print('pixels per core = %.2f'%(total/no_cores)) 
print('So we need %d cores with %d pixels and %d cores with %d pixels.\n ' \
      %(total % no_cores, np.ceil(total/no_cores),no_cores - (total % no_cores), np.floor(total/no_cores)))

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

wl0=8542 # central wavelength
#wl0 = 5250
wlvs = sp[:-1] - wl0 #\delta \lambda
#left/right determines the width of the window to fit over
left, right = 3, 8 # 4, 7  for 5250. use 3, 8 for 8542
xvals = wlvs[left:right]
T=data.shape[2] # size of the time axis

def get_vs(row,col,T=T):
    ans0=0
    result = np.zeros(shape=(T)) # store the results
    dodge  = np.zeros(shape=(T)) # record if it is not a simple fit
    for i in range(0,T):
        fail = False
        profi = data[row,col,i,0,:-1] # extract the full spectral line profile
        profi=profi/np.max(np.abs(profi)) # normalise
        prof = profi[left:right] # select window for fitting
        r = np.polyfit(xvals,prof,2) # quadratic fit
        ans = -r[1]/2/r[0] # -b/2a
        fitted = r[0]*xvals**2 + r[1]*xvals + r[2]
        chi2 = np.sum((fitted - prof)**2) # calculate a goodness of fit
      
        if chi2 > 2e-3:
            fail = True
        if (ans > 0.2): # use 0.2 for 8542/0.1 for 5250
            fail = True
        if (ans < -0.2): # 0.2 for 8542/0.1 for 5250
            fail = True
        if fail:
            dodge[i]=1
            dp = np.diff(profi)
            tps = np.where(dp[:-1]*dp[1:] < 0)[0]+1 # identify turning points
            if len(tps)==0:
                fail=True
                cent=0
                ans=1
            else:
                cent = tps[np.argmin(np.abs(wlvs[tps]-ans0))] # new centre point for a second fit

            if cent<1 or cent>10: # if trying to fit too close to edge of profile
                ans=1.
            else:
              # second fit attempt
                r = np.polyfit(wlvs[cent-1:cent+2],profi[cent-1:cent+2],2)
                ans = -r[1]/2/r[0]

            if (ans > 0.5): # 0.5 for 8542/0.1 for 5250
                fail = True
            if (ans < -0.5): # 0.5 for 8542/0.1 for 5250
                fail = True
            if fail:
                ans=result[i-1] # set answer to previous value
                
        result[i] =ans
        ans0=ans 
    return result, dodge


def function(mini,maxi): # each core gets fed this function
    res = []
    for i in range(maxi-mini):
        if i%100==0:
            print('\r heyah'+str(i),end="")
        row, col = args[0][i+mini], args[1][i+mini]
        res.append(get_vs(row,col))
    return res

# parallel execution
with concurrent.futures.ProcessPoolExecutor(max_workers=no_cores) as executor:
    futures=[executor.submit(function, mins[j], mins[j+1]) for j in range(len(mins)-1)]
    for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        pass #This gets a progress bar
# get results for each core
results=[res.result() for res in futures]

#save the indices
argy = [(args[0][a],args[1][a]) for a in range(len(args[0]))]

#combine the indices (pixel values) and results into one thing
bigr = []
[bigr.extend(r) for r in results]

save_this = [argy,bigr]

savepath = './'

pickle.dump(save_this, open(savepath+'velocities.p',"wb"))

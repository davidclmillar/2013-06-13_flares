import numpy as np
from Waveletfunctions import wavelet, wave_signif
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.dates as dates
from datetime import datetime,timedelta


# ----------------------------------
# red + white noise background model
# -----------------------------------
def model1(f, A, alpha, C):

    if (A < 0) or (C < 0):
        return 0*f - 1e20
    else:
        return A*f**-alpha + C

# -----------------------------------
# gaussian bump model
# -----------------------------------
def model2(f, A, alpha, B, beta, sigma, C):

    if (A < 0) or (B < 0) or (C < 0):
        return 0*f - 1e20

    else:
        pl = A*f**-alpha
        norm = 1/np.sqrt(2*np.pi*sigma**2)
        exptop = -(np.log(f) - beta)**2
        expbot = 2*sigma**2

        gb = B*norm*np.exp(exptop/expbot)

        return pl + gb + C

# -------------------------------------
# kappa model
# --------------------------------------
def modelk(x , A , index , B , kappa , rho , C):

    ax = A*x**-index # the power law term

    if (A < 0) or (B < 0) or (kappa < 0) or (rho < 0) or (C < 0):
        f = 0*x - 1e10         # discard negative solutions

    else:
        if (kappa == 0) or (rho == 0):
            bx = 0         # no kappa term

        else:
            #if kappa > 1e2:
                #kappa = 1e2

            bx = B*(1 + (x**2)/(kappa*rho**2))**(-(kappa + 1)/2)
            #bx = 0

        f = ax + bx + C

    return f

# significance level calculation (the m factor)
def siglvl(x,N):

    return -np.log(1-x**(1/N))

# use this class for everything timeseries related. methods for fitting the spectrum models,
# plotting the FPS (sometimes called psd/psdt), and comparing model fits. 
class timeseries:
    def __init__(self, data, dt, norm=True):
        # original data
        self.data = data
        # timestep
        self.dt   = dt
        #normalised version
        s = self.data
        if norm==True:
            self.normalised = (s-np.mean(s))/np.std(s)
        else:
            self.normalised = data
        # psd
        self.n = len(s)
        apodized = self.normalised*np.hanning(self.n)
        FT = np.fft.fft(apodized)
        vari = np.var(apodized)
        psdt=(abs(FT)**2)
        if norm==True:
            psdt *= 1/vari
        psdt = psdt[1:int(self.n/2)]
        self.psd = psdt
        self.freqs = np.arange(self.n)/(self.dt*self.n)
        self.freqs = self.freqs[1:int(self.n/2)]

        # use wavelets to get the weighting function. adapted from auchere
        mother = 'MORLET'

        s0 = 2.0*self.dt
        noct = np.log(self.n)/np.log(2.0) - 1
        dj = 1/8.0
        j1 = np.fix(noct/dj)

        wave, period, scale, coi = wavelet(self.normalised, self.dt, pad=1, dj=dj, s0=s0, J1=j1, mother=mother)

        power = (abs(wave))**2.0

        global_ws = np.sum(power, axis=1)#/self.n
        interp = interp1d(period, global_ws,fill_value="extrapolate")
        self.weights = interp(1/self.freqs)
        self.sigmas = self.weights #**0.5
    def plot_psd(self,start_time=None):
      # start_time should be a datetime
        f, ax = plt.subplots(1,2,figsize=(10,3))
        
        if start_time is not None:
            minutes = dates.MinuteLocator()
            times = [start_time + timedelta(seconds=self.dt*x) for x in range(self.n)]
            x_vals = dates.date2num(times)
            myFmt = dates.DateFormatter('%H:%M')
            ax[0].plot_date(x_vals,self.normalised,linestyle='-',marker=None,color='k')
            ax[0].xaxis.set_major_formatter(myFmt)
            ax[0].xaxis.set_minor_locator(minutes)
        else:
            ax[0].plot(np.arange(self.n)*self.dt,self.normalised)
        
                
        ax[0].set_xlabel('Time [s]')
        ax[0].set_ylabel(r'Intensity [$\sigma$]')
        ax[1].step(1/self.freqs,self.psd)
        ax[1].set_xscale('log')
        ax[1].set_yscale('log')
        ax[1].set_xlabel('Period [s]')
        ax[1].set_ylabel(r'Power [$\sigma^2$]')
    def fit_kappa(self, plot=True, guess = np.array([6.12353659e-00, 0, 4.76849553e+02, 2.27399893e+04,
       4.83575040e-03, 3.54247390e+00]), **kwargs):

        popt, pcov  = curve_fit(modelk, xdata=self.freqs, ydata=self.psd, sigma=self.sigmas, absolute_sigma=True, p0=guess)#, bounds=limits2, method='trf')

        m = siglvl(0.999, len(self.freqs))

        WRS = np.sum(((modelk(self.freqs,*popt)-self.psd)/self.sigmas)**2)

        if plot==True:
            plt.figure(figsize=(7,4))
            plt.step(1/self.freqs, self.psd)

            plt.plot(1/self.freqs, self.weights)
            plt.plot(1/self.freqs, modelk(self.freqs,*popt))
            plt.plot(1/self.freqs,m*modelk(self.freqs,*popt))

            plt.yscale('log')
            plt.xscale('log')
            plt.text(10**2.5,10**3.5,str(WRS));
        return popt, WRS

    def fit_M1(self, plot=True, guess  = [1e-4, 1.5, 1e-1], **kwargs):

        limits = [[1e-12, 0.01, 1e-2],[1e4, 6,np.inf]]
        popt, pcov  = curve_fit(model1, xdata=self.freqs, ydata=self.psd, sigma=self.sigmas, absolute_sigma=True, p0=guess, bounds=limits)

        m = siglvl(0.999, len(self.freqs))
        WRS = np.sum(((model1(self.freqs,*popt)-self.psd)/self.sigmas)**2)

        if plot==True:
            plt.figure(figsize=(7,4))
            plt.step(1/self.freqs, self.psd)

            plt.plot(1/self.freqs, self.weights)
            plt.plot(1/self.freqs,model1(self.freqs,*popt))
            plt.plot(1/self.freqs,m*model1(self.freqs,*popt))

            plt.yscale('log')
            plt.xscale('log')

            plt.text(10**2.5,10**3.5,str(WRS));

        return popt, WRS

    def fit_M2(self, plot=True, guess=[1e-4, 1, 1e2, -5.1, 0.1, 1],limits = [[1e-12, 0.01, 1e2, -7, 0.1, 1e-2],[1e4, 6, 1e3, -3, 0.3, np.inf]], **kwargs):


        popt, pcov  = curve_fit(model2, xdata=self.freqs, ydata=self.psd, sigma=self.sigmas, absolute_sigma=True, p0=guess, bounds=limits)

        m = siglvl(0.9999, len(self.freqs))
        WRS = np.sum(((model2(self.freqs,*popt)-self.psd)/self.sigmas)**2)
        if plot==True:
            plt.figure(figsize=(7,4))
            plt.step(1/self.freqs, self.psd)

            plt.plot(1/self.freqs, self.weights)
            plt.plot(1/self.freqs, model2(self.freqs,*popt))
            plt.plot(1/self.freqs, m*model2(self.freqs,*(popt*[1,1,0,1,1,1])))
            plt.yscale('log')
            plt.xscale('log')
            plt.text(10**2.5,10**3.5,str(WRS));
        return popt, WRS


    def fit_all(self, plot=True,**kwargs):

        try:
            results_M1 = self.fit_M1(plot=plot)
        except:
            results_M1 = np.array([0,0,0]), 1e6            
        try:
            results_M2 = self.fit_M2(plot=plot)
        except:
            results_M2 = np.array([0,0,0,0,0,0]), 1e6
        try:
            results_kappa = self.fit_kappa(plot=plot)
        except:
            results_kappa = np.array([0,0,0,0,0,0]), 1e6
        
            
        WRS1 = results_M1[1]
        WRS2 = results_M2[1]
        WRSk = results_kappa[1]

        WRSes = [WRS1, WRS2, WRSk]

        results = [results_M1[0], results_M2[0], results_kappa[0],WRSes]
        self.results= results
        return results

    def compare_models(self, sig=0.999):
        
        #input the results from the fitting script made on 14.01.2020.
        #That is a list with [popt1, popt2, poptkappa, WRS]
        #WRS has the WRSs for all the models [1,2,kappa]
        
        results = self.results
        freqs = self.freqs
        popt1 = results[0]
        popt2 = results[1]
        poptk = results[2]

        WRSes = results[3]

    #loop around
        if True:
            if True:

                if True:
        # see if any F-tests need to be performed
                    WRS_list = WRSes
                    popts = np.array([popt1, popt2, poptk])

                    if (np.mean(popts[0])==1e-10) & (np.mean(popts[1])==1e-10) & (np.mean(popts[2])==1e-10):   # no fits successful
                        preferred = 0


                    elif (WRS_list[0]<=WRS_list[1]) & (WRS_list[0]<=WRS_list[2]):
                        preferred = 1


                    elif np.argmin(WRS_list)==1: # f test required
                        F_stat = ((WRS_list[0] - WRS_list[1])/(3))/((WRS_list[1])/(len(freqs) - 6))
                        #print(F_stat)
                        p_value = 1 - stats.f.cdf(F_stat, 3, len(freqs) - 6)
                        #print(p_value)
                        if p_value < (1-sig):
                            # test height of bump
                            params = popt2
                            m = siglvl(sig,len(freqs))
                            just_line = model2(freqs, *(params*[1,1,0,1,1,1]))
                            signif    = m*just_line
                            fit = model2(freqs, *params)
                            index = np.argmax(fit/just_line)

                            if fit[index] > signif[index]:
                                preferred=2

                            elif WRS_list[2] < WRS_list[0]:
                                #print("bump not high enough")
                                preferred = 3

                            else:
                                preferred = 1
                                #print("bump not high enough")
                        else:
                            preferred = 1

                    else:
                        F_stat = ((WRS_list[0] - WRS_list[2])/(3))/((WRS_list[2])/(len(freqs) - 6))
                        p_value = 1 - stats.f.cdf(F_stat, 3, len(freqs) - 6)
                        #print(p_value)
                        if p_value < (1-sig):
                            preferred = 3
                        else:
                            preferred  = 1

        display = {
            "model 1 goodness of fit":round(WRSes[0],2),
            "model 2 goodness of fit":round(WRSes[1],2),
            "kappa model goodness of fit":round(WRSes[2],2),
            "preferred model":preferred
        }
        self.preferred = preferred
        return display


def compare_models_v2(results, freqs, mask, sig=0.999):
#     """
#     input the results from the fitting script.
#     That is a list with [popt1, popt2, poptkappa, WRS]
#     WRS has the WRSs for all the models [1,2,kappa]
#     """
    
    popt1 = results[0]
    popt2 = results[1]
    poptk = results[2]
    
    WRSes = results[3]
    
    preferred = np.zeros_like(WRSes[0,:,:])
    
    #loop around
    for row in range(WRSes.shape[1]):
        for col in range(WRSes.shape[2]):
            
            if mask[row,col]==1:
    # see if any F-tests need to be performed
                WRS_list = WRSes[:,row,col]
                popts = np.array([popt1[:, row, col], popt2[:, row, col], poptk[:, row, col]])
               # check if all fits failed
                if (np.mean(popts[0])==1e-10) & (np.mean(popts[1])==1e-10) & (np.mean(popts[2])==1e-10):   # no fits successful
                    preferred[row, col] = 0   
                
                # if model1 is best
                elif (WRS_list[0]<=WRS_list[1]) & (WRS_list[0]<=WRS_list[2]): # 
                    preferred[row, col] = 1
                    
                # if model 2 seems best, do an f test and check height of bump if it passes
                elif np.argmin(WRS_list)==1:
                    F_stat = ((WRS_list[0] - WRS_list[1])/(3))/((WRS_list[1])/(len(freqs) - 6))
                    p_value = 1 - stats.f.cdf(F_stat, 3, len(freqs) - 6)
                    if p_value < (1-sig):
                        # test height of bump
                        params = popt2[:, row, col]
                        m = siglvl(sig,len(freqs))
                        just_line = model2(freqs, *(params*[1,1,0,1,1,1]))
                        signif    = m*just_line
                        fit = model2(freqs, *params)
                        index = np.argmax(fit/just_line)
                        #  if it failed the f test, assign m3 or m1 depending on which has the best WRS 
                        if (fit[index] > signif[index]) & (params[3] > -6.1) & (params[3] < -3.91):
                            preferred[row, col]=2
                        
                        elif WRS_list[2] < WRS_list[0]:
                            preferred[row, col] = 3
                            
                        else:
                            preferred[row, col] = 1
                            
                    else:
                        preferred[row, col] = 1
                # if m3 looks like the best, do an f-test
                else:
                    F_stat = ((WRS_list[0] - WRS_list[2])/(3))/((WRS_list[2])/(len(freqs) - 6))
                    p_value = 1 - stats.f.cdf(F_stat, 3, len(freqs) - 6)
                    if p_value < (1-sig):
                        preferred[row, col] = 3
                    else:
                        preferred[row,col]  = 1

    return preferred

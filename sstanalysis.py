import numpy as np
from helita.io import lp
import os
from scipy.io.idl import readsav
# functions which read in data, extract timesteps, find B field based on the weak field approximation
# file names and data paths will have to be changed if using these for different data

# read in data
def read_in_data(line,path='/path/to/data/'):
    if line=='5250':
        fn_im = path+'crispex.stokes.5250.08:20:38.time_corrected_rot.fcube'
        fn_sp = path+'crispex.stokes.5250.08:20:38.time_corrected_rot_sp.fcube'
        x = readsav(path+'tseries.5250.08:20:38.calib.sav')
        spec = readsav(path+'spectfile.5250.idlsave')
        lam0 = 5250.22 # Angstroms
        geff = 3.0 # lande factor
    elif line == '8542':
        fn_im = path+'crispex.stokes.8542.aligned_5250.08:20:38.time_corrected_rot.fcube'
        fn_sp = path+'crispex.stokes.8542.aligned_5250.08:20:38.time_corrected_rot_sp.fcube'
        x = readsav(path+'tseries.8542.08:20:38.calib.sav')
        spec = readsav(path+'spectfile.8542.idlsave')
        lam0 = 8542.09 #8542.09  # central wavelength in Angs
        geff = 1.10     # Effective Lande factor
    else:
        raise ValueError('line must be 8542 or 5250, got %s'%line)

    C = 4.6686e-13 * lam0**2 * geff   ## For wavelength in Angs

    hdr_im = lp.getheader(fn_im)
    hdr_sp = lp.getheader(fn_sp)

    nx = hdr_im[0][0]
    ny = hdr_im[0][1]
    nt = hdr_sp[0][1]
    nw = hdr_sp[0][0]

    data = lp.getdata(fn_im)
    data = np.reshape(data,[nx,ny,nt,4,nw])

    sp  = spec['spect_pos']

    return data, sp, C

def get_mask(line='8542',path='/path/to/data/'):

    if line=='5250':
        fn_im = path+'mask.5250.08:20:38.corrected_rot.icube'
    elif line == '8542':
        fn_im = path+'mask.8542.aligned_5250.08:20:38.corrected_rot.icube'
    else:
        raise ValueError('line must be 8542 or 5250, got %s'%line)

    hdr_im = lp.getheader(fn_im)
    data = lp.getdata(fn_im)
    
    return data

def get_times(mode='str',line='8542',path='/path/to/data/'):

    if line=='8542':
        fn = path+'tseries.8542.08:20:38.calib.sav'
    elif line=='5250':
        fn = path+'tseries.5250.08:20:38.calib.sav'
    else:
        raise ValueError('line must be 8542 or 5250, got %s'%line)
    d = readsav(fn)['time']
    d = [x.decode("utf-8") for x in d]

    return d
    
def get_aia_data(wl,path='/path/to/data/'):

    fn = path+"aia%s.aligned_5250.08:20:38.time_corrected_rot.icube"%(wl)
    hdr = lp.getheader(fn)
    nx = hdr[0][0]
    ny = hdr[0][1]
    nt = hdr[0][2]

    data = lp.getdata(fn)
    data = np.reshape(data,[nx,ny,nt])

    return data

def wfa_ca(localdata,sp):
  # localdata should be the of shape (lambda, stokes)
  # sp is the values of lambda
    cw = 8542.091
    geff = 1.1
    G = 1.21
    # inputs
    # wavelengths
    sp = sp[:-1]
    # local continuum value
    cont = localdata[0,-1]
    # Stokes params
    I = localdata[0,:-1]/cont
    Q = localdata[1,:-1]/cont
    U = localdata[2,:-1]/cont
    V = localdata[3,:-1]/cont

    # Blos
    #c value
    C1 = 4.669e-13*geff*cw**2
    derI = np.gradient(I)/np.gradient(sp)
    num=0
    den=0
    for ii in range(sp.size):
        num+=V[ii]*derI[ii]
        den+=derI[ii]**2

    Blos = num/C1/den

    # Bhor
    # c value
    C2 = 0.75*(4.669e-13*cw**2)**2*G
    ilim=[4,6]
    numQ=0
    denQ=0
    numU=0
    denU=0
    for ii in range(sp.size):
        if ii<ilim[0] or ii>ilim[1]:
            iw = sp[ii]-cw
            numQ+=Q[ii]*derI[ii]/iw
            numU+=U[ii]*derI[ii]/iw

            denQ+=(derI[ii]/iw)**2
            denU+=(derI[ii]/iw)**2

    denQ*=C2
    denU*=C2

    BhorQ2_2 = (numQ/denQ)**2
    BhorU2_2 = (numU/denU)**2

    Bhor = np.sqrt(np.sqrt(BhorQ2_2+BhorU2_2))

    #azimuth
    numX=0
    denX=0
    for ii in range(sp.size):
        numX+=U[ii]*derI[ii]
        denX+=Q[ii]*derI[ii]
    Bazi=0.5*np.arctan2(numX,denX)
    if Bazi<0: Bazi+=np.pi
        
    return np.array([Blos,Bhor,Bazi])

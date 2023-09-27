# 2013-06-13_flares
Code for analysis found in a paper about B-class flares on 2013-06-13

- ```sstanalysis.py```, ```Waveletfunctions.py``` and ```kappa_fitting.py``` contain lots of functions/classes used in the scripts.

- ```Waveletfunctions.py``` is code originally from [Torrence & Compo, 1998](https://ui.adsabs.harvard.edu/link_gateway/1998BAMS...79...61T/doi:10.1175/1520-0477(1998)079%3C0061:APGTWA%3E2.0.CO;2) which was [translated to python]("http://atoc.colorado.edu/research/wavelets/") by Evgeniya Predybaylo.

- use ```get_velocities.py``` to calculate Doppler shifts in CRISP data.

- ```fit_spectra.py``` runs spectrum fitting on intensity or velocity data.

- ```parallel_compare.py``` compares the results of the fits to get preferred models.

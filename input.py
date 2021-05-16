#!/usr/bin/env python
# -*- coding: utf-8 -*-   

from CAnac import nac_calc

T_start = 1 
T_end   = 1000 
    
# NAC calculations and Genration of standard input for HFNAMD or PYXAID
# bmin and bmax are actual band index in VASP, and should be same with the bmin and bmax in your NAMD simulation.
is_combine = True   #If generate standard input for HFNAMD or PYXAID
#iformat = "PYXAID" 
iformat = "HFNAMD"
bmin    = 166       
bmax    = 186         
potim   = 1         # Nuclear timestep, unit: fs 
    
# Time-overlap 
# bmin_stored bmax_stored are actual band index in VASP
# Use a large basis sets here if you would like to remove WAVECAR to save disk usage
# Or when you turn on the state reordering  
#bmin_stored = bmin - 10
#bmax_stored = bmax + 10
bmin_stored    = 50      
bmax_stored    = 200       
    

nproc   = 4         # Number of cores used in parallelization

is_gamma_version  = False  # Which VASP version is used!!  vasp_std False  vasp_gam True
is_reorder= True    # If turn on State Reordering   True or False
is_alle   = True    # If use All-electron wavefunction(require NORMALCAR) True or False
is_real   = True   # If rotate wavefunction to ensure NAC is real value. True or False
    
ikpt    = 1         #k-point index, starting from 1 to NKPTS
ispin   = 1         #spin index, 1 or 2

# Directories structure. Here, 0001 for 1st ionic step, 0002 for 2nd ionic step, etc.
# Don't forget the forward slash at the end.
Dirs = ['./%04d/' % (ii + 1) for ii in range(T_start-1, T_end)] 




# Don't change anything below if you are new to CA-NAC    
#########################################################################   
# For Pseudo NAC only. omin and omax are used for post-orthonormalization.
# In principle, you should use entire basis sets in VASP
icor    = 1
omin    = bmin_stored
omax    = bmax_stored

skip_file_verification  = False
skip_TDolap_calc = False 
skip_NAC_calc = False
onthefly_verification  = True
    
checking_dict={'skip_file_verification':skip_file_verification,'skip_TDolap_calc':skip_TDolap_calc,'skip_NAC_calc':skip_NAC_calc,'onthefly_verification':onthefly_verification}
    
    
nac_calc(Dirs, checking_dict, nproc, is_gamma_version, is_reorder, is_alle, is_real, is_combine,
         iformat, bmin, bmax,
         bmin_stored, bmax_stored, omin, omax,
         ikpt, ispin, icor, potim )

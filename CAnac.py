#!/usr/bin/env python
# -*- coding: utf-8 -*-   
import os
import sys
import numpy as np
import multiprocessing
from time import time
import mod_hungarian as hungarian
from vaspwfc import vaspwfc
from aeolap import PawProj_info,ae_aug_olap_martrix,test,realtime_checking
from spinorb import read_cproj_NormalCar

def version():
    print("CA-NAC 1.0.9_beta")
    print("Should you have any question, please contact wc_086@usc.edu")
    
def combine(runDirs,bmin_s,bmax_s,obmin,obmax, ispin, ikpt, potim,is_alle,is_reorder,is_real,iformat):
    
    Ev_To_Ry = 1.0 / 13.605662285137 
    Hbar_Ev = 0.6582119
    nbasis = bmax_s - bmin_s + 1
    
    nac = np.zeros((len(runDirs),nbasis**2),dtype=np.complex)
    eig = np.zeros((len(runDirs),nbasis))
    
    tag_ae = 'ae' if is_alle else 'ps'
    tag_rd = 'rd' if is_reorder else ''
    tag_rl = '_real_' if is_real else ''
    # in  
    nac_filename='nac_'  + tag_ae + tag_rd + '.npy'
    eig_filename='eig_'  + tag_ae + tag_rd + '.npy'
    # out
    nacre_filename='CAnac_' + str(obmin) + '_' + str(obmax)+'_' + 'ispin' + str(ispin) + '_' + 'k' + str(ikpt) + '_' + tag_ae + tag_rd + tag_rl + '_re.txt'
    nacim_filename='CAnac_' + str(obmin) + '_' + str(obmax)+'_' + 'ispin' + str(ispin) + '_' + 'k' + str(ikpt) + '_' + tag_ae + tag_rd + tag_rl + '_im.txt'
    eig_out_filename='CAeig_' + str(obmin) + '_' + str(obmax)+'_' + 'ispin' + str(ispin) + '_' + 'k' + str(ikpt) + '_' + tag_ae + tag_rd + '.txt'
    
    for i,dirs in enumerate(runDirs[:-1]):
        nac[i,:] = np.load(dirs+nac_filename).reshape(nbasis**2)
        eig[i,:] = np.load(dirs+eig_filename)
    
    if is_real:
        nac=np.abs(nac)*np.sign(nac.real)
        
    obasis = obmax - obmin + 1   
    nac = nac.reshape(-1, nbasis, nbasis)[:, obmin-bmin_s : obmin-bmin_s+obasis, obmin-bmin_s : obmin-bmin_s+obasis]
    nac = nac.reshape(-1, obasis*obasis)
    
    eig=eig[:, obmin-bmin_s : obmin-bmin_s+obasis]
    
        
    if iformat=='PYXAID' or iformat=='P':
       
        for i,dirs in enumerate(runDirs[:-2]):
            ham=(np.diag((eig[i,:]+eig[i+1,:])/2).flatten()+nac[i,:]/2.0/potim*-1j*Hbar_Ev).reshape(obasis,obasis)*Ev_To_Ry
            np.savetxt(dirs+'Ham%d_re' %(i),ham.real)
            np.savetxt(dirs+'Ham%d_im' %(i),ham.imag)
       
    
    else:
        np.savetxt(nacre_filename,nac.real[:-1,:])
        np.savetxt(nacim_filename,nac.imag[:-1,:])
        np.savetxt(eig_out_filename,eig[:-1,:]) 
    

def task_checking(Dirs, obmin, obmax, ispin, ikpt, is_alle):
    
    t1 = time()
    tag_ae='ae' if is_alle else 'ps'
    tdolap_filename='tdolap_' + str(obmin) + '_' + str(obmax) + '_' + str(ispin) + '_' + str(ikpt) + '_' + tag_ae + '.npy'
    
    task_Dirs = np.zeros(len(Dirs), dtype=np.bool)
    completed_Dirs = np.zeros(len(Dirs), dtype=np.bool)
    
    waveA_Dirs = np.zeros(len(Dirs), dtype=np.bool)
    waveB_Dirs = np.zeros(len(Dirs), dtype=np.bool)
    tdolap_Dirs = np.zeros(len(Dirs), dtype=np.bool)
    
    Dirs=np.array(Dirs)
    DirsA=None
    DirsB=None
        
    for i, rundir in enumerate(Dirs):
       
        if os.path.exists(rundir+tdolap_filename):  
            tdolap_Dirs[i] = True
            
        if os.path.exists(rundir+"WAVECAR"):
            waveA_Dirs[i] = True
            if i>0:
                waveB_Dirs[i-1] = True 
            
    #print (tdolap_Dirs)
    #print (waveA_Dirs)    
    
    tdolap_Dirs[-1]= True
    
    task_Dirs[waveA_Dirs & waveB_Dirs & ~tdolap_Dirs]=True
 
    completed_Dirs[tdolap_Dirs] = True
        
    
    # check if WAVECAR has been recalculated 
#  This routine currently has a conflict if you removed WAVECARs after generating intermediate tdolap files.
#    for i, rundir in enumerate(Dirs[:-1]):
        
#        if waveA_Dirs[i] & tdolap_Dirs[i] :
#            wav=os.stat(rundir+"WAVECAR")
#            tolap=os.stat(rundir+tdolap_filename) 
#            if tolap.st_mtime < wav.st_mtime:
#                    task_Dirs[i] = True
#                    completed_Dirs[i] = False
    
    completed_flag = True if False not in completed_Dirs else False
    
    
    if completed_flag:
        print ('Files integrity checking completed')
    else:
        wav_missing=np.union1d(Dirs[~tdolap_Dirs & ~waveA_Dirs],(Dirs[1:])[~tdolap_Dirs[:-1] & ~waveB_Dirs[:-1]])
        print ('%5d TDolaps to be calculated ' %task_Dirs[task_Dirs==True].shape)
        print ('%5d WAVECARs to be calculated (VASP)' %wav_missing.shape[0] )
        if wav_missing.shape[0] < 10:
            print('Please provide WAVECAR for these directories:', wav_missing)
    
    if True in task_Dirs:
        DirsA=(Dirs[:-1])[task_Dirs[:-1]]
        DirsB=(Dirs[1:])[task_Dirs[:-1]]
        
    #print (DirsA,DirsB)
    t2 = time()
    print ('2. Elapsed Time: %.4f [s] in integrity checking' % (t2 - t1))
    return DirsA,DirsB,completed_flag
    
def orthogon(cic):
    
    S = np.dot(cic.conj(),cic.T)

    Dsqrt= np.zeros_like(S, dtype=np.complex)
    T = np.zeros_like(S, dtype=np.complex)
    cio = np.zeros_like(cic, dtype=np.complex)
    
    D,V=np.linalg.eig(S)
    for ii in range(np.size(S,0)):
        Dsqrt[ii,ii]=1/np.sqrt(D[ii])
    T=np.dot(np.dot(V.conj(),Dsqrt.conj()),V.T)
    
    cio = np.dot(T,cic)
    
    return cio              
    
def reorder_td_olap(td_olap,perm1,perm2):
    
    td_olap_tmp=td_olap.copy()
     
    for i in range(len(perm1)):
        for j in range(len(perm2)):
            td_olap_tmp[perm1[i],perm2[j]]=td_olap[i,j]
     
    return td_olap_tmp 

def reorder_cc(cc,perm):
    
     cc_rd=cc.copy()
     for i in range(len(perm)):
         cc_rd[perm[i]]=cc[i]
         #cc_rd[i]=cc[perm[i]]
     
     return cc_rd
 
def reorder_eig(eig,perm):
    
     eig_rd=eig.copy()
     for i in range(len(perm)):
         eig_rd[perm[i]]=eig[i]
         #eig_rd[i]=eig[perm[i]]
     
     return eig_rd

def reorder_pij(pij,pji,perm):
     
     pij_rd=np.zeros_like(pij)
     pji_rd=np.zeros_like(pji)
     for i in range(len(perm)):
         for j in range(len(perm)):
              pij_rd[perm[i],perm[j]]=pij[i,j]
              pji_rd[perm[i],perm[j]]=pji[i,j]
              #pij_rd[i,j]=pij[perm[i],perm[j]]
              #pji_rd[i,j]=pji[perm[i],perm[j]]
     

     return pij_rd,pji_rd    

def reorder_verification(runDirs,is_alle):
    
    tag_ae='ae' if is_alle else 'ps'
    tag_rd='rd' 
    eig_filename='eig_' + tag_ae + tag_rd + '.npy'
    
    eig_diff=np.load(runDirs[-2]+eig_filename)-np.load(runDirs[0]+eig_filename)
    if np.max(np.abs(eig_diff)) > 0.3:
        print ('False reordering may exist!')
        print ('Please plot the energy profile and carefully check the state reording!' )

def phase_from_tdolap(td_olap,is_gamma):
    
    cor2 = np.diag(td_olap)
    cc2 = cor2/abs(cor2)

    if is_gamma:
        for ii in range(td_olap.shape[0]):
            cc2[ii]= 1.0 if cor2[ii].real>0 else -1.0

    cc1=np.ones_like(cc2) 

    return cc1,cc2    

def phasecor_apply(pij,pji,cc1,cc2,is_gamma,bmin_s,omin,nbasis):

    nacs=np.zeros_like(pij)
    for ii in range(nbasis):
        for jj in range(ii):

            ibi = ii + bmin_s - omin 
            ibj = jj + bmin_s - omin  

            pij[ii,jj] = pij[ii,jj]*cc1[ibi]*cc2[ibj].conj()
            pji[ii,jj] = pji[ii,jj]*cc2[ibi]*cc1[ibj].conj()
            
            #Sij[ii,jj] = Sij[ii,jj]*cc1[ibi]*cc1[ibj].conj()
            #Sji[ii,jj] = Sji[ii,jj]*cc2[ibi]*cc2[ibj].conj()
            
            #tmp = pij[ii,jj]-Sij[ii,jj]-pji[ii,jj] +Sji[ii,jj]
            tmp = pij[ii,jj]-pji[ii,jj]
            
            nacs[ii,jj] = tmp.real if is_gamma else tmp
            if is_gamma:
                nacs[jj,ii] = -nacs[ii,jj]
            else:
                nacs[jj,ii] = -np.conj(nacs[ii,jj])
     
    return nacs


def nac_from_tdolap(dirA, omin, omax, ispin=1, ikpt=1, is_reorder=False, is_alle=False, is_gamma=False):  
#    t1 = time()
    prefix = dirA
        
    tag_ae='ae' if is_alle else 'ps'
    tdolap_filename = prefix + 'tdolap_' + str(omin) + '_' + str(omax) + '_' + str(ispin) + '_' + str(ikpt) + '_' + tag_ae + '.npy' 
    tdeig_filename = prefix + 'eig_' + str(omin) + '_' + str(omax) + '_' + str(ispin) + '_' + str(ikpt) + '.npy' 
        
        
    EnT=np.load(tdeig_filename)
    td_olap=np.load(tdolap_filename)
    
    if is_reorder:
        #print(np.abs(td_olap))
        reorder_cost=np.real(td_olap.conj()*td_olap)
        res = hungarian.maximize(reorder_cost)
        perm1=np.array(res,dtype=np.int)[:,0]
        perm2=np.array(res,dtype=np.int)[:,1]
        td_olap_reorded = reorder_td_olap(td_olap,perm1,perm2)
        cc1,cc2 =phase_from_tdolap(td_olap_reorded,is_gamma)
#        t2 = time()
#        print ('2. Elapsed Time: %.4f [s] in reordering' % (t2 - t1))
#        t1 = t2    
    else:
        cc1,cc2 = phase_from_tdolap(td_olap,is_gamma)
        perm1 = None
        perm2 = None
        
    pij= td_olap_reorded  if is_reorder else td_olap
    pji=np.conj(np.transpose(pij))
#    t2 = time()
#    print ('3. Elapsed Time: %.4f [s]' % (t2 - t1))
    
    if is_reorder:
        for i in range(len(perm1)):
            if perm1[i] != perm2[i] :
                print ('Switched',dirA,perm1[i],perm2[i],"Energy difference",EnT[perm1[i]]-EnT[perm2[i]])
                print ('Switched',dirA,perm1[i],perm2[i],"Overlap",np.abs(td_olap[perm1[i],perm1[i]]),np.abs(td_olap[perm1[i],perm2[i]]))
    # return EnT, raw overlap 
    return EnT, pij, pji, cc1, cc2 ,perm1, perm2
    
    
def tdolap_from_vaspwfc(dirA, dirB, paw_info=None, is_alle=False,
                     bmin_s=None, bmax_s=None, omin=None, omax=None, 
                     ikpt=1, ispin=1, icor=1, OntheflyVerify = True):
    '''
    Calculate Nonadiabatic Couplings (NAC) from two WAVECARs
    <psi_i(t)| (psi_j(t+dt))> 
    
    inputs:
        waveA:  path of WAVECAR A
        waveB:  path of WAVECAR B
        gamma:  gamma version wavecar
        dt:     ionic time step, in [fs]          
        ikpt:   k-point index, starting from 1 to NKPTS
        ispin:  spin index, 1 or 2

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!! Note, this method is much slower than fortran code. !!!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!! Now, It is much faster than fortran code :)         !!!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    '''
    
    waveA=dirA+'WAVECAR'
    waveB=dirB+'WAVECAR'

    phi_i = vaspwfc(waveA)      # wavecar at t
    phi_j = vaspwfc(waveB)      # wavecar at t + dt
    
    normalcar_i = dirA + '/NormalCAR'
    normalcar_j = dirB + '/NormalCAR'

    print ('Calculating TD Overlap between <%s> and <%s>' % (waveA, waveB))

    assert phi_i._nbands == phi_j._nbands, '#bands not match!'
    assert phi_i._nplws[ikpt-1] == phi_j._nplws[ikpt-1], '#nplws not match!'

    

    bmin_s = 1 if bmin_s is None else bmin_s
    bmax_s = phi_i._nbands if bmax_s is None else bmax_s
    omin = bmin_s if (icor==1 or icor==5) else omin
    omax = bmax_s if (icor==1 or icor==5) else omax
    nbasis = bmax_s - bmin_s + 1
    obasis = omax - omin + 1    # Basis for orthogonalization

    nkpts=phi_i._nkpts
    nbands=phi_i._nbands

    ci_t   = phi_i.readBandCoeff(ispin, ikpt, omax, norm=False)
    cic_t = np.zeros([obasis] + list(ci_t.shape),dtype=np.complex)
    cic_tdt = np.zeros([obasis] + list(ci_t.shape),dtype=np.complex)
           
    cio_t = np.zeros([obasis] + list(ci_t.shape),dtype=np.complex)
    cio_tdt = np.zeros([obasis] + list(ci_t.shape),dtype=np.complex)
    


    print (cic_t.shape)
    
    
    t1 = time()
    
    if is_alle:
        cprojs1=read_cproj_NormalCar(normalcar_i)
        cprojs2=read_cproj_NormalCar(normalcar_j)

   
    #t2 = time()
    #print '2. Elapsed Time: %.4f [s] in reading croj' % (t2 - t1)
    #t1 = t2
    

    if is_alle:
        for ii in range(obasis):
            ib1 = ii + omin
            cic_t[ii,:]   = phi_i.readBandCoeff(ispin, ikpt, ib1, norm=False)
            cic_tdt[ii,:]   = phi_j.readBandCoeff(ispin, ikpt, ib1, norm=False)
    else:
        for ii in range(obasis):
            ib1 = ii + omin
            cic_t[ii,:]   = phi_i.readBandCoeff(ispin, ikpt, ib1, norm=True)
            cic_tdt[ii,:]   = phi_j.readBandCoeff(ispin, ikpt, ib1, norm=True)

    cio_t = orthogon(cic_t) if (icor==21 or icor == 25) else cic_t
    cio_tdt = orthogon(cic_tdt) if (icor==21 or icor== 25) else cic_tdt
    

    t2 = time()
    print ('2. Elapsed Time: %.4f [s] in reading wavefunction and projector' % (t2 - t1))
    t1 = t2


    
    td_olap=np.dot(cio_t.conj(),np.transpose(cio_tdt))
    
    if OntheflyVerify & is_alle:
        S_olap = np.dot(cio_t.conj(),np.transpose(cio_t))
        S_aug_olap = ae_aug_olap_martrix(bmin_s, bmax_s, cprojs1, cprojs1, paw_info, nkpts, nbands, ikpt, ispin)
        S_olap = S_olap + S_aug_olap
        
        realtime_checking(S_olap, dirA)
    t2 = time()
    print ('2. Elapsed Time: %.4f [s] in overlap' % (t2 - t1))
    t1 = t2


    if is_alle:
        td_aug_olap=ae_aug_olap_martrix(bmin_s, bmax_s, cprojs1, cprojs2, paw_info, nkpts, nbands, ikpt, ispin)
        td_olap = td_olap + td_aug_olap
    
        t2 = time()
        print ('2. Elapsed Time: %.4f [s] in aug_overlap' % (t2 - t1))
        t1 = t2

#  # EnT = (phi_i._bands[ispin-1,ikpt-1,:] + phi_j._bands[ispin-1,ikpt-1,:]) / 2.
    EnT = phi_i._bands[ispin-1, ikpt-1, bmin_s-1:bmax_s]
    
    # close the wavecar
    phi_i._wfc.close()
    phi_j._wfc.close()
    

    return EnT, td_olap

def parallel_tdolap_calc(dirA, dirB, checking_dict, nproc=None, is_alle=False, 
                      bmin_s=None, bmax_s=None,omin=None, omax=None,
                      ikpt=1, ispin=1, icor=1 ):
    '''
    Parallel calculation of TD overlaps using python multiprocessing package.
    '''
    import multiprocessing

    nproc = multiprocessing.cpu_count() if nproc is None else nproc
    pool = multiprocessing.Pool(processes=nproc)
    results = []
   
    
    
    if is_alle:
        test(bmin_s, bmax_s, dirA[0])
        paw_info=PawProj_info(dirA[0])
    else:
        paw_info=None
 
    for w1, w2 in zip(dirA, dirB):
       
        res = pool.apply_async(tdolap_from_vaspwfc, (w1, w2, paw_info, is_alle, bmin_s, bmax_s, omin, omax, ikpt, ispin, icor, checking_dict['onthefly_verification']))
        results.append(res)

    for ii in range(len(dirA)):

        et, td_olap = results[ii].get()
        
        #Writing
        prefix = dirA[ii]
        
        tag_ae='ae' if is_alle else 'ps'

        tdolap_filename = prefix + 'tdolap_' + str(omin) + '_' + str(omax) + '_' + str(ispin) + '_' + str(ikpt) + '_' + tag_ae + '.npy' 
        tdeig_filename = prefix + 'eig_' + str(omin) + '_' + str(omax) + '_' + str(ispin) + '_' + str(ikpt) + '.npy' 
        
        
        np.save(tdeig_filename, et)
        np.save(tdolap_filename, td_olap)

############################################################
############################################################


def parallel_nac_calc(runDirs, nproc=None, is_gamma=False, is_reorder=False, is_alle=False, 
                      bmin_s=None, bmax_s=None,omin=None, omax=None,
                      ikpt=1, ispin=1, icor=1):
    '''
    Parallel calculation of NAC using python multiprocessing package.
    '''
   

    nproc = multiprocessing.cpu_count() if nproc is None else nproc
    pool = multiprocessing.Pool(processes=nproc)
    results = []
   
     
    

 
    for w1 in runDirs:
        res = pool.apply_async(nac_from_tdolap, (w1, omin, omax, ispin, ikpt, is_reorder, is_alle, is_gamma))
        results.append(res)

    for ii in range(len(runDirs)-1):

        et, pij, pji, cc1, cc2 ,perm1, perm2 = results[ii].get()
        
        
        
        if is_reorder:
            if ii == 0:
                cc1 = np.ones_like(cc2)
                perm_cum =np.arange(len(cc2))
            else:
                cc2 = reorder_cc(cc2, perm_cum) 
                cc1 = cc_next.copy()
                cc2 = cc1 * cc2    
        
 
            et = reorder_eig(et, perm_cum) 
            pij,pji = reorder_pij(pij, pji, perm_cum) 
        else:
            if (icor == 1 or icor == 21) :
                if ii == 0:
                    cc1 = np.ones_like(cc2)
                else:
                    cc1 = cc_next.copy()
                    cc2 = cc1 * cc2    
        
        nc = phasecor_apply(pij, pji, cc1, cc2, is_gamma, bmin_s, omin, bmax_s - bmin_s + 1) 
        
        #Writing
        prefix = runDirs[ii]
        
        tag_ae='ae' if is_alle else 'ps'
        tag_rd='rd' if is_reorder else ''
      
        nac_filename=prefix+'nac_'  + tag_ae + tag_rd + '.npy'
        eig_filename=prefix+'eig_'  + tag_ae + tag_rd + '.npy'
        
        
        np.save(eig_filename, et)
        np.save(nac_filename, nc)

        
        
        
        cc_next = cc2.copy()
        
        if is_reorder:
           perm_filename=prefix+'perm_'  + tag_ae + tag_rd + '.txt'
           np.savetxt(perm_filename, perm_cum[np.newaxis, :])
            
           perm_t=perm_cum.copy()
           for i in range(len(perm_t)):
               perm_t[i]=perm_cum[perm2[i]]
           perm_cum=perm_t.copy()
           
   
############################################################
############################################################



def nac_calc(runDirs, checking_dict, nproc=None, is_gamma=False, is_reorder=False, is_alle=False, is_real=False, is_combine=False,
             iformat='HFNAMD', ibmin=None, ibmax=None,
             bmin_s=None, bmax_s=None,omin=None, omax=None,
             ikpt=1, ispin=1, icor=1, potim=1.0):
    
    
    if is_alle == True and is_gamma == True:
        print("Currently, all-electron NAC does not support gamma-version WAVECAR")
        sys.exit(0)
  
    skip_file_verification  = checking_dict['skip_file_verification']
    skip_TDolap_calc = checking_dict['skip_TDolap_calc']
    skip_NAC_calc = checking_dict['skip_NAC_calc']
    onthefly_verification  = checking_dict['onthefly_verification']
    
    version()
    
    if not skip_file_verification:
        print ("Checking Files Integrity")
        DirA,DirB,completed_flag = task_checking(runDirs, omin, omax, ispin, ikpt, is_alle)
    
        if DirA is not None:
            print ("Starting TDolap Calculations")
            parallel_tdolap_calc(DirA, DirB, checking_dict, nproc, is_alle, bmin_s, bmax_s, omin, omax, ikpt, ispin, icor)
            DirA,DirB,completed_flag = task_checking(runDirs, omin, omax, ispin, ikpt, is_alle)
    
        if completed_flag: 
            print ("Starting CA-NAC")
            parallel_nac_calc(runDirs, nproc, is_gamma, is_reorder, is_alle, bmin_s, bmax_s, omin, omax, ikpt, ispin, icor)
            print ("CA-NAC Calculations is done")
            if is_combine:
                print ("Generating Standard Input for ", iformat)
                combine(runDirs, bmin_s, bmax_s, ibmin, ibmax, ispin, ikpt, potim, is_alle, is_reorder, is_real,iformat)
                if is_reorder:
                    print("The state tracking( is_reorder = True) is turned on, please check the reordering carefully")
                    reorder_verification(runDirs, is_alle)
                if not is_real:
                    print("The current NACs are complex value (is_real = False), which are not supported by Hefei-NAMD and PYXAID(default integrator) ")
                    print("Please be aware of that!!!")

        
        else:
            print('WAVECAR generation are not finished or TDolap files are incomplete' ) 
    
    if skip_TDolap_calc:
        if skip_NAC_calc:
            if is_combine:
                print ("Generating Standard Input for ", iformat)
                combine(runDirs, bmin_s, bmax_s, ibmin, ibmax, ispin, ikpt, potim, is_alle, is_reorder, is_real, iformat)
                if is_reorder:
                    reorder_verification(runDirs, is_alle)
        else:
            print ("Starting CA-NAC")
            parallel_nac_calc(runDirs, nproc, is_gamma, is_reorder, is_alle, bmin_s, bmax_s, omin, omax, ikpt, ispin, icor)
            print ("CA-NAC Calculations is done")
            if is_combine:
                print ("Generating Standard Input for ", iformat)
                combine(runDirs, bmin_s, bmax_s, ibmin, ibmax, ispin, ikpt, potim, is_alle, is_reorder, is_real,iformat)
                if is_reorder:
                    print("The state tracking(is_reorder = True) is turned on, please check the reordering carefully")
                    reorder_verification(runDirs, is_alle)
                if not is_real:
                    print("The current NACs are complex value (is_real = False), which are not supported by Hefei-NAMD and PYXAID(default integrator) ")
                    print("Please be aware of that!!!")

    
if __name__ == '__main__':
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
    bmin_stored    = 166       
    bmax_stored    = 186       
    

    nproc   = 4         # Number of cores used in parallelization

    is_gamma_version  = False  # Which VASP version is used!!  vasp_std False  vasp_gam True
    is_reorder= True    # If turn on State Reordering   True or False
    is_alle   = True    # If use All-electron wavefunction(require NORMALCAR) True or False
    is_real   = True   # If rotate wavefunction to ensure NAC is real value. True or False
    
    ikpt    = 1         #k-point index, starting from 1 to NKPTS
    ispin   = 1         #spin index, 1 or 2

# Directories structure. Here, 0001 for 1st ionic step, 0002 for 2nd ionic step, etc.
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




        
        
    


import numpy as np
from paw import pawpotcar
import re
import time
from vaspwfc import vaspwfc
from paw import nonlq, nonlr
from ase.io import read, write
from spinorb import read_cproj_NormalCar
import sys


def read_diffovlap(datastr):
    data = datastr.strip().split('\n')
    #nmax = int(data[0].split()[0])
    #print data
    grid_start_idx = data.index(" augmentation charges (non sperical)") + 1
    diffovlap_data = np.array([ x for line in data[grid_start_idx:] for x in line.strip().split() if not re.match(r'\ \w+', line) ], dtype=float)
    return diffovlap_data


class PawProj_info(object):
    def __init__(self,dir0):
         
        self.ae_difq_setup(dir0)

    def proj_lm_gen(self,_proj_l):
    
        _proj_l_seq=[]
        _proj_m_seq=[]

        for ii in range(len(_proj_l)):        
            _proj_m_seq.extend(range(_proj_l[ii]*2 + 1))
            _proj_l_seq.extend([_proj_l[ii]] * (_proj_l[ii]* 2 + 1))

        return _proj_l_seq,_proj_m_seq

#pawpp = [pawpotcar(potstr) for potstr in
#        open(potcar).read().split('End of Dataset')[:-1]]

#print pawpp[0].proj_l

    def ae_difq_setup(self,dir0):

        poscar = read(dir0+'POSCAR')
        wfc = vaspwfc(dir0+'WAVECAR')
        potcar = dir0+'POTCAR'
        
        ikpt = 1
        p1 = nonlq(poscar, wfc._encut, k=wfc._kvecs[ikpt-1])

        t0 = time.time()

        difq=[]
        proj_tot=[]
        rotate_idx_tot=[]
        proj_l_seq=[]
        proj_m_seq=[]
        difq_ii_complete=[]
        difq_ij_complete=[]
        difq_ii=[]
        difq_ij=[]

        for potstr in open(potcar).read().split('End of Dataset')[:-1] :
 
            pawpp = pawpotcar(potstr)
            
            pawpp.get_Qij()
            
            print ("Projectrors:",pawpp.proj_l)
            nproj_l = pawpp.proj_l.shape[0]
            
            res=np.unique(pawpp.proj_l,return_counts=True)
            assert max(res[1]) <= 2, 'Error, projectors are not supported!'
            
            p_tot=np.sum(pawpp.proj_l*2+1)

          
    
            #print ('psov',psov,'aeov',aeov)


            #aug_chg_part=potstr.split('uccopancies in atom')[0] 
            #dif_olap=read_diffovlap(aug_chg_part)

            #dif_olap=dif_olap.reshape(pawpp.proj_l.shape[0],pawpp.proj_l.shape[0])

            res = self.proj_lm_gen(pawpp.proj_l.tolist())
    
            proj_l_seq.extend([res[0]])
            proj_m_seq.extend([res[1]])

            proj_tot.extend([p_tot])
    
            
            
            #print dif_olap.shape
            rotate_idx=np.arange(p_tot,dtype=np.int) 
            single_idx=[]
            for i in range(p_tot):
                if i == 0 and res[0][i]==res[0][i+res[0][i]*2+1]:
                    rotate_idx[i]= i+(res[0][i]*2+1)
                elif i-(res[0][i]*2+1) >=0 and res[0][i]==res[0][i-res[0][i]*2-1]:
                    rotate_idx[i]= i-(res[0][i]*2+1)
                elif i+(res[0][i]*2+1) <p_tot and res[0][i]==res[0][i+res[0][i]*2+1]:
                    rotate_idx[i]= i+(res[0][i]*2+1)
                else:
                    single_idx.extend([i])            
            #print rotate_idx
            rotate_idx_tot.extend([rotate_idx])
            
            #print "Qij",pawpp.paw_qij
            qij_ii = np.diag(pawpp.paw_qij)
            tmp = pawpp.paw_qij[:,rotate_idx]
            tmp2 = tmp
            tmp2[single_idx,single_idx]=0.0
            #tmp2[:,single_idx]=0.0
            qij_ij = np.diag(tmp2)
            
            #print qij_ii,qij_ij
            
            

            #tmp=[]
            #full_tmp=[]
            #for i,itot_l in enumerate(pawpp.proj_l.tolist()):
             #    tmp=[]
             #    for j, jtot_l in enumerate(pawpp.proj_l.tolist()):
             #        tmp=tmp+([dif_olap[i,j].tolist()]*(jtot_l*2+1))
             #   full_tmp=full_tmp+(tmp*(itot_l*2+1))
    
            #difq.extend([np.array(full_tmp).reshape(p_tot,p_tot)])
            difq_ii.extend([qij_ii])
            difq_ij.extend([qij_ij])
                               
        
        proj_cum=0
        r_idx=[]
        for i in p1.element_idx:
            r_idx +=(np.array(rotate_idx_tot[i],dtype=np.int)+proj_cum).tolist()
            difq_ii_complete += difq_ii[i].tolist()
            difq_ij_complete += difq_ij[i].tolist()
            proj_cum=proj_cum + proj_tot[i]
        #print difq_ii_complete
        #print "difqij",difq_ij_complete
            
        
        
        #difq_complete=[]
        #for i in p1.element_idx:
        #    difq_complete.extend([difq[i]])
 
        t1 = time.time()
        
        self.natoms = p1.natoms
        self.element_idx = p1.element_idx
        self.difq_ii = difq_ii_complete
        self.difq_ij = difq_ij_complete
        self.l_seq = proj_l_seq
        self.m_seq = proj_m_seq
        self.tot = proj_tot
        self.rotate_idx = r_idx
        print ('1. Elapsed Time: %.4f [s] in Qij Construction' % (t1 - t0))
        
def ae_aug_olap_martrix(bmin,bmax,cprojs1,cprojs2,proj_info,nkpts,nbands,ikpt=1,ispin=1):
    
     nbasis=bmax-bmin+1
     aug_olap_matrix=np.zeros((nbasis,nbasis),dtype=np.complex)
     
     index_min=bmin - 1 + nbands * (ikpt - 1) + nbands * nkpts * (ispin - 1)
     index_max=bmax     + nbands * (ikpt - 1) + nbands * nkpts * (ispin - 1)

     cproj1=cprojs1[index_min:index_max]
     cproj2=cprojs2[index_min:index_max] 
     ctmp=np.zeros_like(cproj2)
     
     tmp=np.dot(cproj1.conj()*proj_info.difq_ii,cproj2.transpose())
     for i,j in enumerate(proj_info.rotate_idx):
         ctmp[:,i]=cproj2[:,j]
     
     cproj2_rotate=ctmp
     aug_olap_matrix= tmp + np.dot(cproj1.conj()*proj_info.difq_ij,cproj2_rotate.transpose())
   
     return aug_olap_matrix


def test(bmin=5,bmax=40,dir0='./'):

    nbasis=bmax-bmin+1
    ikpt=1 
    ispin=1
    print (dir0)
    proj=PawProj_info(dir0)
    cprojs1=read_cproj_NormalCar(dir0+'NormalCAR')
    print('cprojs1.shape',cprojs1.shape)
    cprojs2=read_cproj_NormalCar(dir0+'NormalCAR')
    
    wfc=vaspwfc(dir0+'WAVECAR')

    nkpts= wfc._nkpts
    nbands= wfc._nbands
    cptwf = wfc.readBandCoeff(iband=bmax, ikpt=ikpt, ispin=ispin)
    
    wfc_coef = np.zeros([nbasis] + list(cptwf.shape),dtype=np.complex)
    td_olap = np.zeros((nbasis,nbasis),dtype=np.complex)
    
    aug_olap=ae_aug_olap_martrix(bmin,bmax,cprojs1,cprojs2,proj,nkpts,nbands,ikpt,ispin)
    
    for i in range(nbasis):
        nband=bmin+i
        wfc_coef[i] = wfc.readBandCoeff(iband=nband, ikpt=ikpt)
    
    
    for i in range(nbasis):
        for j in range(nbasis):
            td_olap[i,j]=np.sum(wfc_coef[i].conj()*wfc_coef[j])
            
    diff_olap = td_olap+aug_olap - np.identity(nbasis)
    
    error_max = np.max(np.abs(diff_olap))
    
    if error_max >1e-5 :
        print ("Error exists in Projector or Qij")
        print ("Please check NormalCAR, POTCAR or WAVECAR")
        print ("Pseudo Overlap \n",td_olap)
        print ("AE Overlap \n",td_olap+aug_olap)
        sys.exit(1)
    else:
        print ("Qij checking completed")
        
        
def realtime_checking(s_olap,dir):
              
    diff_olap = s_olap - np.identity(s_olap.shape[0])
    
    error_max = np.max(np.abs(diff_olap))
    
    if error_max >1e-5 :
        print ("Error in directory:",dir)
        print ("np.abs(diff_olap)",np.abs(diff_olap))
        print ("S_overlap is not a identity matrix")
        print ("Please check NormalCAR, POTCAR or WAVECAR")
        print ("Please check covergency in SCF calculation")
        sys.exit(1)

    
     
if __name__ == '__main__':
    test()

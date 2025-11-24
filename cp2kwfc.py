import numpy as np
from time import time

class cp2kwfc(object):


#    def __init__(self, proj_name, iband, ispin, norm=True):
        
#        self._pname = proj_name
#        self.fname = proj_name+'-WFN_%05d_%d-1_0.cube' %iband %ispin
#        self.norm = norm
#        self.ispin = ispin

    def __init__(self, wfc):
        
        
        self.fname = wfc
#        self.ispin = ispin
        
    def read_wfc_r(self, norm=True):
        with open(self.fname,'r') as wfc_file:
            lines = wfc_file.readlines()
            natoms=int((lines[2].split())[0])
            ngx=int((lines[3].split())[0])
            ngy=int((lines[4].split())[0])
            ngz=int((lines[5].split())[0])

            posinfo = np.array ( [x for line in lines[6:6+natoms] for x in line.split()] , dtype=np.float64)

            self.natoms = natoms 
            self.ngx = ngx 
            self.ngy = ngy 
            self.ngz = ngz
            self.size = ngx*ngy*ngz
            self.posinfo = posinfo
            self.head = lines[:6+natoms] 

            cg = np.array ( [x for line in lines[6+natoms:] for x in line.split() ] , dtype=np.float64)
            assert cg.shape[0] == self.size, 'Error in reading wavefunction' 
            if norm:
                cg /= np.linalg.norm(cg)
            wfc_file.close()
        return cg


    def write_rho_r(self,ncol=5):
        
        
        with open('spin.cube','w') as file:
            cg = self.read_wfc_r()
            fmt = "%13.5E"
            rho = cg*cg
            
            nrow = self.size // ncol
            rho_1 = rho[:nrow*ncol].reshape((nrow,ncol))  
            rho_2 = rho[nrow*ncol:]  
   
            file.write(''.join([xx for xx in self.head])) 

            file.write( '\n'.join([''.join([fmt % xx for xx in row ]) for row in rho_1 ]) )
            file.write( '\n' + ''.join([fmt % xx for xx in rho_2 ]) )

def read_eig(filename):
    o_spin=0
    u_spin=0
    eig1=[]
    eig2=[]
    au_to_ev = 2.72113838565563E+01
    
    with open(filename,'r') as f:
        while True:
            line = f.readline()
            if "Eigenvalues" in line:
                if "occupied" in line and "unoccupied" not in line:
                    f.readline()
                    tmp = f.readline()
                
                    if o_spin==0:
                        while "Fermi" not in tmp and tmp.split():
                            eig1 += tmp.split()
                            tmp = f.readline()
                    else:
                        while "Fermi" not in tmp and tmp.split():
                            eig2 += tmp.split()
                            tmp = f.readline()
                    o_spin = o_spin + 1
                if "unoccupied" in line:
                    f.readline()
                    f.readline()
                    tmp = f.readline()
        
                    if u_spin==0:
                        while tmp.split():
                            eig1+=tmp.split()
                            tmp = f.readline()
                    else:
                        while tmp.split():
                            eig2 += tmp.split()
                            tmp = f.readline()
                    u_spin = u_spin + 1
            if not line:
                break
    
    eig1=np.array(eig1,dtype=np.float64) * au_to_ev
    eig2=np.array(eig2,dtype=np.float64) * au_to_ev
    
    return eig1,eig2



def tdolap_from_cp2kwfc(dirA, dirB, proj_name, outfile,
                     bmin=None, bmax=None, 
                     ispin=1 ):
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
    
    
        
    print ('Calculating TD Overlap between <%s> and <%s>' % (dirA, dirB))

    waveA=dirA+proj_name+'-WFN_%05d' %bmin + '_%d' %ispin +  '-1_0.cube'
    waveB=dirB+proj_name+'-WFN_%05d' %bmin + '_%d' %ispin +  '-1_0.cube'
  

    phi_i = cp2kwfc(waveA)      # wavecar at t
    phi_j = cp2kwfc(waveB)      # wavecar at t + dt
    
    ci_t   = phi_i.read_wfc_r(norm=True)
    ci_tdt   = phi_j.read_wfc_r(norm=True)
    
    assert phi_i.size == phi_j.size, '#ngrids not match!'

    

    bmin = 1 if bmin is None else bmin
    bmax = phi_i._nbands if bmax is None else bmax

    nbasis = bmax - bmin + 1


   
    cic_t = np.zeros([nbasis] + list(ci_t.shape),dtype=np.complex)
    cic_tdt = np.zeros([nbasis] + list(ci_t.shape),dtype=np.complex)
           
    print (cic_t.shape)
    
    
    t1 = time()
    
 
    #t2 = time()
    #print '2. Elapsed Time: %.4f [s] in reading croj' % (t2 - t1)
    #t1 = t2
    
 
    for ii in range(nbasis):
        ib1 = ii + bmin
        waveA=dirA+proj_name+'-WFN_%05d' %ib1 + '_%d' %ispin +  '-1_0.cube'
        waveB=dirB+proj_name+'-WFN_%05d' %ib1 + '_%d' %ispin +  '-1_0.cube'

        phi_i = cp2kwfc(waveA)      # wavecar at t
        phi_j = cp2kwfc(waveB)      # wavecar at t + dt

        cic_t[ii,:]   = phi_i.read_wfc_r(norm=True)
        cic_tdt[ii,:]   = phi_j.read_wfc_r(norm=True)

     

    t2 = time()
    print ('2. Elapsed Time: %.4f [s] in reading wavefunction and projector' % (t2 - t1))
    t1 = t2


    
    td_olap=np.dot(cic_t.conj(),np.transpose(cic_tdt))
    
    #if OntheflyVerify & is_alle:
    #    S_olap=np.dot(cio_t.conj(),np.transpose(cio_t))
    #    S_aug_olap=ae_aug_olap_martrix(bmin,bmax,cprojs1,cprojs1,paw_info,nkpts,nbands,ikpt,ispin)
    #    S_olap = S_olap + S_aug_olap
        
    #    realtime_checking(S_olap,dirA)
    t2 = time()
    print ('2. Elapsed Time: %.4f [s] in overlap' % (t2 - t1))
    t1 = t2



#  # EnT = (phi_i._bands[ispin-1,ikpt-1,:] + phi_j._bands[ispin-1,ikpt-1,:]) / 2.
    eig1,eig2= read_eig(dirA+outfile)
    EnT = eig1[bmin-1:bmax] if ispin==1 else eig2[bmin-1:bmax]
    

    return EnT, td_olap


#def cp2k_test(wfc="H2O-WFN_00258_2-1_0.cube",eig="h2o.out"):
def cp2k_test(wfc="79Na2O-2NaCl-20CaO-100SiO2-32PbS-WFN_02976_1-1_0.cube",eig="output.out"):
    phi_i = cp2kwfc(wfc)
    cg = phi_i.read_wfc_r(norm=True)
    print(np.sum(cg*cg))
    eig1,eig2 = read_eig(eig)
    print(eig1)
    print(eig2)
        


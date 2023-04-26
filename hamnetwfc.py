import numpy as np
import os

def dummy(): pass

class hamnetwfc(object):

  def __init__(self, dir:str) -> None:
    '''
    Now only support gamma calcuation, single spin.
    Since we use exact diag method. 
    We shall always get all the eigenvalues & eigenvectors.
    nbands = norbitals
    '''
    self._dname = dir
    self._fname = os.path.join(dir, 'wfc.npy')
    self._ename = os.path.join(dir, 'eigen.npy')
    self._lgam = True

    if not os.path.isfile(self._fname):
      raise IOError("%s doen't exist." % self._fname)
    if not os.path.isfile(self._ename):
      raise IOError("%s doen't exist." % self._ename)
    
    self.readWF()

  def isGammaWfc(self):
    return True if self._lgam else False
  
  def readWF(self):
    self._wfc:np.ndarray = np.load(self._fname) # shape: (nbands, nbands)
    self._wfc.close = dummy
    self._nkpts = 1
    self._nspin = 1
    self._nbands = self._wfc.shape[1]
    self._nplws = [self._wfc.shape[1]]*self._nkpts
    self._kvecs = np.zeros((self._nkpts, 3), dtype=float)
    # energy
    self._bands = np.load(self._ename).reshape(self._nspin, self._nkpts, -1) # shape: (nspin, nkpts, nbands)

  def readBandCoeff(self, ispin=1, ikpt=1, iband=1, norm=False):
    '''
    Read the NAO coefficients of specified KS states.
    '''
    self.checkIndex(ispin, ikpt, iband)
    cg = self._wfc[iband-1]
    if norm:
        cg /= np.linalg.norm(cg)
    return cg

  def checkIndex(self, ispin, ikpt, iband):
    '''
    Check if the index is valid!
    '''
    assert 1 <= ispin <= self._nspin,  'Invalid spin index!'
    assert 1 <= ikpt <= self._nkpts,  'Invalid kpoint index!'
    assert 1 <= iband <= self._nbands, 'Invalid band index!'

import numpy as np
import os

ry2ev = 13.605662285137 
def dummpy(): pass

class abacuswfc(object):

  def __init__(self, dir:str, lgamma=False) -> None:
    self._dname = dir
    self._lgam = lgamma
    self._nspin = 1

    if not os.path.isdir(self._dname):
      raise IOError(f'{dir} is not a directory.')
    self.readWF()

  def isGammaWfc(self):
    return True if self._lgam else False

  def readWF(s):
    try:
      with open(os.path.join(s._dname, 'kpoints')) as _kfnm:
        s._nkpts = int(_kfnm.readline().split()[-1])
    except:
      raise IOError('Cannot open kpoints file.')
    # assuming all kpoints use the same number of orbitals.
    init = True
    # Loop over kpoints
    for ik in range(s._nkpts):
      _wfnm = open(os.path.join(s._dname, f'LOWF_K_{ik}.dat'))
      _wfnm.readline() # k index
      kvec = list(map(float, _wfnm.readline().split()))
      nbnd = int(_wfnm.readline().split()[0])
      norb = int(_wfnm.readline().split()[0])
      if init:
        s._nbands = nbnd
        s._nplw = norb * np.ones(s._nkpts, dtype=int)
        s._kvecs = np.zeros((s._nkpts, 3), dtype=float)
        s._bands = np.zeros((s._nspin,s._nkpts,nbnd), dtype=np.float64) # energy nspin, nkpts, nbands
        s._occs = np.zeros((s._nspin,s._nkpts,nbnd), dtype=np.float64) # occ nspin, nkpts, nbands
        s._wfc = np.zeros((s._nspin,s._nkpts,nbnd,norb), dtype=np.float64) # shape: nspin, nkpts, nbands, norbitals
        init = False
      s._kvecs[ik] = kvec
      # Loop over bands
      for ib in range(nbnd):
        _wfnm.readline() # band index
        s._bands[0,ik,ib] = float(_wfnm.readline().split()[0])
        s._occs[0,ik,ib] = float(_wfnm.readline().split()[0])
        tmp = []
        # Loop over orbitals
        for io in range(norb, 10):
          tmp.extend(_wfnm.readline().split())
        s._wfc[0,ik,ib] = np.asarray(tmp, dtype=np.float64)
      _wfnm.close()
    s._bands *= ry2ev
    s._wfc.close = dummpy

  def readBandCoeff(self, ispin=1, ikpt=1, iband=1, norm=False):
    '''
    Read the NAO coefficients of specified KS states.
    '''
    self.checkIndex(ispin, ikpt, iband)
    cg = self._wfc[ispin-1,ikpt-1,iband-1]
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

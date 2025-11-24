import numpy as np
import os

# only support nspin=1 for now.

ry2ev = 13.605662285137 

class abacuswfc(object):
  '''
  @property: _wfnm: TextIOWrapper. file handle.
  @property: _nkpts: int.
  @property: _lgam: bool.
  @property: _nspin: 1
  @property: _nbands: int. number of bands.
  @property: _nplws: list[int]. shape: (nkpt, nuotot). wfc coefficient number for each kpoint.
  @property: _kvecs: ndarray[float64]. shape: (nkpt, 3).
  @property: _bands: ndarray[float64]. shape: (nspin, nkpt, nbands). band energy.
  @property: _occs:  ndarray[float64]. shape: (nspin, nkpt, nbands). occuapation.
  @property: _wfc:   ndarray[float64]. shape: (nspin, nkpt, nbands, norbs). wfc coefficients.
  '''

  def __init__(self, dir:str, lgamma=False) -> None:
    self._dname = dir
    self._lgam = lgamma
    self._nspin = 1
    self._nowK = -1

    if not os.path.isdir(self._dname):
      raise IOError(f'{dir} is not a directory.')
    self.readWFHeader()
    self.readWF(ik=0)

  def isGammaWfc(self):
    return True if self._lgam else False
  
  def readWFHeader(self):
    # get kpts
    try:
      with open(os.path.join(self._dname, 'kpoints')) as _kfnm:
        self._nkpts = int(_kfnm.readline().split()[-1])
        _kfnm.readline()
        self._kvecs = np.zeros((self._nkpts, 3), dtype=np.float64)
        for ik in range(self._nkpts):
          tmp = _kfnm.readline().split()
          self._kvecs[ik] = list(map(float, tmp[1:4]))
    except:
      raise IOError('Cannot open kpoints file.')

  def readWF(s, ik=0):
    # assuming all kpoints use the same number of orbitals.
    s._nowK = ik

    _wfnm = open(os.path.join(s._dname, f'LOWF_K_{ik+1}.dat'))
    _wfnm.readline() # k index
    kvec = list(map(float, _wfnm.readline().split()))
    nbnd = int(_wfnm.readline().split()[0])
    norb = int(_wfnm.readline().split()[0])
    if not hasattr(s, '_nbands'):
      s._nbands = nbnd
      s._nplws = s._nkpts * [norb]
      s._bands = np.zeros((s._nspin,s._nkpts,nbnd), dtype=np.float64) # energy nspin, nkpts, nbands
      s._occs = np.zeros((s._nspin,s._nkpts,nbnd), dtype=np.float64) # occ nspin, nkpts, nbands
      if s.isGammaWfc():
        s._wfc = np.zeros((s._nspin,s._nkpts,nbnd,norb), dtype=np.float32) # shape: nspin, nkpts, nbands, norbitals
      else:
        s._wfc = np.zeros((s._nspin,s._nkpts,nbnd,norb), dtype=np.complex64) # shape: nspin, nkpts, nbands, norbitals

    # Loop over bands
    for ib in range(nbnd):
      _wfnm.readline() # band index
      s._bands[0,ik,ib] = float(_wfnm.readline().split()[0])
      s._occs[0,ik,ib] = float(_wfnm.readline().split()[0])
      # Loop over orbitals
      tmp = []
      if s.isGammaWfc():
        for io in range(norb, 10):
          tmp.extend(_wfnm.readline().split())
        tmp = np.asarray(tmp, dtype=np.float32)
        s._wfc[0,ik,ib] = tmp
      else:
        for io in range(0, norb*2, 10):
          tmp.extend(_wfnm.readline().split())
        tmp = np.asarray(tmp, dtype=np.float32)
        s._wfc[0,ik,ib] = tmp[0::2] + 1j * tmp[1::2]
      
    _wfnm.close()
    s._bands *= ry2ev

  def readBandCoeff(self, ispin=1, ikpt=1, iband=1, norm=False):
    '''
    Read the NAO coefficients of specified KS states.
    '''
    self.checkIndex(ispin, ikpt, iband)
    if self._nowK != ikpt - 1:
      self.readWF(ikpt - 1)

    cg = self._wfc[ispin-1,ikpt-1,iband-1]
    # if norm:
    #     cg /= np.linalg.norm(cg)
    return cg

  def checkIndex(self, ispin, ikpt, iband):
    '''
    Check if the index is valid!
    '''
    assert 1 <= ispin <= self._nspin,  'Invalid spin index!'
    assert 1 <= ikpt <= self._nkpts,  'Invalid kpoint index!'
    assert 1 <= iband <= self._nbands, 'Invalid band index!'

if __name__ == '__main__':
  wfc = abacuswfc('.', lgamma=False)
  b = wfc.readBandCoeff(ispin=1, ikpt=1, iband=1)
  print(wfc._kvecs)
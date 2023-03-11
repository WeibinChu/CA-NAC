import numpy as np
import os
import io

# int = np.int32 = integer(kind=4)
# float = np.float64 = real(kind=8)

def fromfortran(fp, dtype, count=1):
  pos = fp.tell()
  length, = np.fromfile(fp, dtype=np.int32, count=1)
  dump = []
  if isinstance(dtype, list):
    for itype, icount in zip(dtype, count):
      dump.append(np.fromfile(fp, dtype=itype, count=icount))
  else:
    dump = np.fromfile(fp, dtype=dtype, count=count)
  fp.seek(pos + length + 8)
  return dump


class siestawfc(object):
  '''
  Class for processing SIESTA wavefunction stored in SystemName.WFSX

  The format of SIESTA WFSX is: 
    nk, gamma
    nspin
    nuotot
    iaorb(:), labelfis(:), iphorb(:), cnfigfio(:), symfio(:)
    Loop over k-points
      Loop over spin
        nk, k(3)
        ispin
        nwflist
      End Loop over spin
      Loop over bands 
      ! for LCAO basis, one orbital corresponds to one band?
        indwf
        energy
        psi(:,:)
      End Loop over bands
    Loop over k-points
  '''

  def __init__(self, fnm='WFSX', lgamma=False) -> None:
    self._fname = fnm
    self._dname = os.path.dirname(fnm)
    if self._dname == '':
      self._dname = '.'

    self._lgam = lgamma

    try:
      self._wfc = open(self._fname, 'rb')
    except:
      raise IOError('Failed to open %s' % self._fname)

    # read the basic information
    self.readWFHeader()
    # read the band information
    self.readWFBand()

  def isGammaWfc(self):
    return True if self._lgam else False

  def readWFHeader(self):
    self._wfc.seek(0)
    dump = fromfortran(self._wfc, dtype=np.int32, count=2)
    self._nkpts = int(dump[0])
    self._lgam = bool(dump[1])
    self._nspin, = fromfortran(self._wfc, dtype=np.int32, count=1)
    self._nuotot, = fromfortran(self._wfc, dtype=np.int32, count=1) # nuotot
    self._nspin = min(4,self._nspin)
    nuotot = self._nuotot
    self._iaorb    = np.zeros(nuotot, dtype=int)
    self._labelfis = np.zeros([nuotot,20], dtype=int)
    self._iphorb   = np.zeros(nuotot, dtype=int)
    self._cnfigfio = np.zeros(nuotot, dtype=int)
    self._symfio   = np.zeros([nuotot,20], dtype=int)
    pos = self._wfc.tell()
    length, = np.fromfile(self._wfc, dtype=np.int32, count=1)
    for idx in range(nuotot):
      self._iaorb[idx]    = np.fromfile(self._wfc, dtype=np.int32, count=1)
      self._labelfis[idx] = (np.fromfile(self._wfc, dtype=np.uint8, count=20))
      dump                = np.fromfile(self._wfc, dtype=np.int32, count=2)
      self._iphorb[idx]   = dump[0]
      self._cnfigfio[idx] = dump[1]
      self._symfio[idx]   = (np.fromfile(self._wfc, dtype=np.uint8, count=20))
    self._wfc.seek(pos + length + 8)

  def readWFBand(self):
    self._nplws = [self._nuotot]*self._nkpts # useless
    self._kvecs = np.zeros((self._nkpts, 3), dtype=float)
    self._bands = [[[]]*self._nkpts]*self._nspin # energy
    self._recs = [[[]]*self._nkpts]*self._nspin
    for iik in range(self._nkpts):
      for iispin in range(self._nspin):
        dump = fromfortran(self._wfc, 
                           dtype=[np.int32, np.float64],
                           count=[1,3])
        self._kvecs[iik,:] = dump[1]
        dump = self._wfc.tell()
        fromfortran(self._wfc, dtype=np.int32, count=1)
        dump, = fromfortran(self._wfc, dtype=np.int32, count=1)
        if not hasattr(self, '_nbands'):
          self._nbands = dump
        elif self._nbands != dump:
          raise ValueError("nbands doesn't match between different kpts/spins.")

        dumpB, dumpR = [], []
        for iw in range(self._nbands):
          fromfortran(self._wfc, dtype=np.int32, count=1)
          dumpB.append(fromfortran(self._wfc, dtype=np.float64, count=1)[0])
          pos = self._wfc.tell()
          length, = np.fromfile(self._wfc, dtype=np.int32, count=1)
          dumpR.append(pos)
          self._wfc.seek(pos + length + 8) # length = 4*self._nuotot*self._nspin
        self._bands[iispin][iik] = dumpB
        self._recs[iispin][iik] = dumpR
    self._bands = np.array(self._bands, dtype=np.float32)
    a = 1

  def get_ps_wfc(self, *args, **kwargs):
    '''
    Alias for the wfc_r method.
    '''
    return self.wfc_r(*args, **kwargs)
  
  def wfc_r(self):
    pass

  def readBandCoeff(self, ispin=1, ikpt=1, iband=1, norm=False):
    '''
    Read the NAO coefficients of specified KS states.
    '''

    self.checkIndex(ispin, ikpt, iband)

    rec = self.whereRec(ispin, ikpt, iband)
    self._wfc.seek(rec)

    cg = fromfortran(self._wfc, dtype=np.float32, count=self._nuotot*self._nspin)[0]. \
          reshape(-1, self._nspin).swapaxes(0, 1).flatten()

    # cg = np.asarray(dump, dtype=np.complex128)
    if norm:
        cg /= np.linalg.norm(cg)
    return cg

  def whereRec(self, ispin=1, ikpt=1, iband=1):
    '''
    Return the rec position for specified KS state.
    '''
    self.checkIndex(ispin, ikpt, iband)
    return self._recs[ispin-1][ikpt-1][iband-1]

  def checkIndex(self, ispin, ikpt, iband):
    '''
    Check if the index is valid!
    '''
    assert 1 <= ispin <= self._nspin,  'Invalid spin index!'
    assert 1 <= ikpt <= self._nkpts,  'Invalid kpoint index!'
    assert 1 <= iband <= self._nbands, 'Invalid band index!'

if __name__ == '__main__':
  wfc = siestawfc(r'C:\Users\zhang\UserSpace\Github\HamNet\utils\Cs2BBX6.fullBZ.WFSX')


    
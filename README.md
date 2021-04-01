# CA-NAC 
Concentric Approximation - Nonadiabatic Coupling

is a code base to accurately and efficiently evaluate nonadiabatic coupling (NAC), a crucial quantity in nonadiabatic molecular dynamics (NAMD), under the PAW formalism. Specifically, it's used to extract the planewave coefficients from `VASP` pseudo-wavefunction file `WAVECAR` and  PAW projector coefficients from `VASP` projector file `NORMALCAR(optional)` to calculate pseudo-NAC (PS-NAC) and exact-NAC `AKA "all-electron" under PAW` (AE-NAC).  In addition, other advanced techniques like phase correction and state reordering are also implemented in this code.

## Prerequisites
*  Python3
*  ASE
*  Scipy
*  Numpy
*  VaspBandUnfolding [https://github.com/QijingZheng/VaspBandUnfolding]
* (Optional）For AE-NAC only，you need to modify VASP source code slightly and recompile it. The original part in VASP is not coded properly. A patch is available upon request by email (Please make sure you have legal access to VASP)

## Known issues
1. For AE-NAC only, currently, it does not support the `NORMALCAR` that is generated with gamma-only version VASP. Alternatively, you can use standard verision VASP to generate `WAVECAR` and `NORMALCAR` with only one gamma point in `KPOINTS`. 
2. For spin-polarized AE-NAC only, currently, the spin down part in `NORMALCAR` is not correct. Will fix it soon.

## Reference

Citation is much appreciated. 👍

* **Main CA-NAC paper**

  **Concentric Approximation for Fast and Accurate Numerical Evaluation of Nonadiabatic Coupling with Projector Augmented-Wave Pseudopotentials**

  Weibin Chu and Oleg V. Prezhdo

  _The Journal of Physical Chemistry Letters_ **2021** 12 (12), 3082-3089 

* **Evaluation of AE-NAC and PS-NAC. PS-NAC fails in the system with transition metal**

  **Accurate Computation of Nonadiabatic Coupling with Projector Augmented-Wave Pseudopotentials**

  Weibin Chu, Qijing Zheng, Alexey V. Akimov, Jin Zhao, Wissam A. Saidi, and Oleg V. Prezhdo

  _The Journal of Physical Chemistry Letters_ **2020** 11 (23), 10073-10080 




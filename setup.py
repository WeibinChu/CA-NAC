#!/usr/bin/env python

from distutils.core import setup

setup(
    name         = 'CA-NAC',
    version      = '1.2.0_beta',
    description  = 'Concentric Approximation - Nonadiabatic Coupling.',
    author       = 'Weibin Chu',
    author_email = 'wbchu@fudan.edu.cn',
    url          = 'https://github.com/WeibinChu/CA-NAC',
    py_modules   = [
        'abacuswfc', 'aeolap', 'CAnac', 'cp2kwfc',
        'hamnetwfc', 'hamgnnhugewfc', 'mod_hungarian', 'siestawfc'
    ],
)
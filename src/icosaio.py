#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on 2013.10.18.

@author: András Olasz
@version: 2015.07.06.dev.1
'''

import subprocess
import numpy as np
from elements import ELEMENTS


def getxyz(fname, exclude=['']):
    #TODO: Openbabel should be used through pybel; however, no success installing/compiling
    #  openbabel with python bindings in a conda environment.
    proc = subprocess.Popen(["obabel", fname, "-oxyz"], stdout=subprocess.PIPE)
    line = True
    numatoms = False
    while line:
        line = proc.stdout.readline().decode()
        if not numatoms:
            i = 0
            numatoms = int(line.strip())
            geom = np.empty((numatoms, 4), dtype=np.float64)
            proc.stdout.readline()  # comment line is discarded
        elif i < numatoms:
            words = line.split()
            symbol = words[0]
            words[0] = ELEMENTS[symbol].number
            geom[i] = [float(i) for i in words]
            i += 1
    return geom

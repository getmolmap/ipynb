#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on 2013.10.18.

@author: András Olasz
@version: 2015.07.06.dev.1
'''

import numpy as np
from elements import ELEMENTS

def getxyz(fname, exclude=['']):
    print(fname)
    with open(fname, 'r') as f:
        numatoms = int(f.readline().strip())
        geom = np.empty((numatoms, 4), dtype=np.float64)
        f.readline() #comment line is discarded
        for i in range(numatoms):
            line = f.readline()
            pieces = line.split()
            symbol = pieces[0]
            pieces[0] = ELEMENTS[symbol].number
            geom[i] = [float(i) for i in pieces]
        return geom

def getxyz_v1(fname):
    with open(fname, 'r') as f:
        numatoms = int(f.readline().strip())
        coords = np.empty((numatoms, 3), dtype=np.float64)
        symbols = []
        f.readline() #comment line is discarded
        for i in range(numatoms):
            line = f.readline()
            pieces = line.split()
            symbols.append(pieces[0])
            coords[i] = [float(i) for i in pieces[1:4]]
        return (symbols, coords)


def writemesh(fname, verts, faces, edges=None):
    with open(fname, 'w') as f:
        part1 = '\n'.join([' '.join([str(v_) for v_ in vert]) for vert in verts])
        part2 = '\n'.join([' '.join([str(f_) for f_ in face]) for face in faces])
        f.write('verts\n' + part1 + '\nfaces\n' + part2)

def savexlsx(**kwargs):
    data = kwargs['data']
    fname = kwargs['fname']
    from xlsxwriter.workbook import Workbook
    workbook = Workbook(fname)
    worksheet = workbook.add_worksheet('worksheet1')
    format1 = workbook.add_format()
    format1.set_num_format('0.000')
    # Write a 2D array.
    array1 = np.arange(15).reshape(3, 5)
    for row, data in enumerate(array1):
        worksheet.write_row(row, 0, data)
    # Write a 1D array.
    array2 = np.array([6, 7, 8])
    worksheet.write_column('A5', array2)
    workbook.close()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2013.10.18.

@author: András Olasz
@version: 2016.01.05.dev.2

"""
#TODO: 1. Egyéni atomi rádiuszok

import time
import math
import sys, os
import array
from collections import defaultdict
import numpy as np
# import numpy.ma as ma
import scipy
import scipy.spatial
import xlsxwriter
import pandas as pd
# from scipy.spatial import cKDTree
# from scipy.spatial import distance as sp_dis
try:
    from elements import ELEMENTS
except ImportError:
    if './progdata' not in sys.path:
        sys.path.append('../progdata')
    from elements import ELEMENTS
import icosaio
import icosphere as ico
# import scipy.spatial as sp
sp = scipy


class lazyattr(object):
    """Lazy object attribute whose value is computed on first access."""
    __slots__ = ['func']

    def __init__(self, func):
        self.func = func

    def __get__(self, instance, owner):
        result = self.func(instance)
        if result is NotImplemented:
            return getattr(super(owner, instance), self.func.__name__)
        setattr(instance, self.func.__name__, result)
        return result


class Logger():
    def __init__(self, **kw):
        self.debug_level = kw.get('debug_level', 0)
        if kw.get('table', False):
            self.df_column_names = ('Description', '1', )
            self.df = pd.DataFrame(columns=self.df_column_names)
            self.df_row = 0
            self.tags_to_html = ('file_name', 'coverage', 'centrum', 'angle')
        else:
            self.df = False
        out = kw.get('out', '../results/getmolmap_results') + '.xlsx'
        sheet_name = kw.get('sheet_name', 'getMolMap')
        self.workbook = xlsxwriter.Workbook(out)
        self.worksheet = self.workbook.add_worksheet(sheet_name)
        self.worksheet.set_column('A:A', 24)
        self.worksheet.set_column('B:D', 16)
        self.row = 0
        tags = {'file_name': 'File Name',
                'sub': 'IcoSphere Subdivision',
                'radius': 'Cut Radius',
                'rad_type': 'Type of Atomic Radii',
                'atom_scale': 'Scale Factor of Atomic Radii',
                'excludes': 'Excluded Elements',
                'coverage': 'Buried Volume',
                'centrum': 'Centrum',
                'cone_atoms': 'Atoms on the Inverse Cone',
                'angle': 'Inverse Cone Angle',}
        self.tags = tags
        self.decimals = defaultdict(int, {'radius': 3,
                                          'atom_scale': 2,
                                          'coverage': 3,
                                          'angle': 1, })


    def log(self, **kw):
        tags = self.tags
        tag = kw['tag']
        value = kw['value']
        plain_format = self.workbook.add_format({'bold': False})
        cell_format = self.workbook.add_format({'align': 'center'})
        decimals = self.decimals[tag]
        options = {'level': 1, 'hidden': False}
        if decimals:
            xlsx_num_format = '0.{}'.format('0' * decimals)
            num_format = self.workbook.add_format({'num_format': xlsx_num_format,
                                                   'align': 'center', })
            cell_format = num_format
        elif tag == 'centrum':
            value = [''.join([str(x) for x in value])]
        if tag in ('file_name', 'coverage'):
            options = {'level': 0, 'collapsed': False}
        if tag == 'file_name':
            plain_format.set_top(2)
            cell_format.set_top(2)

        self.worksheet.set_row(self.row, None, plain_format, options)
        self.worksheet.write(self.row, 0, self.tags[tag], plain_format)
        self.worksheet.write_row(self.row, 1, value, cell_format)
        self.row += 1
        if 'df' in dir(self):
            if tag in self.tags_to_html:
                row_values = [''] * (len(self.df_column_names) - 1)
                for i in range(len(value)):
                    if decimals > 0:
                        row_values[i] = '{:0.{}f}'.format(value[i], decimals)
#                        row_values[i] = round(value[i], decimals)
                    else:
                        row_values[i] = value[i]
                self.df.loc[self.df_row] = [tags[tag]] + row_values
                self.df_row += 1

        if self.debug_level:
            print('{} ({}):'.format(tags[tag], kw['tag']), *kw['value'])


    def close(self):
        self.workbook.close()
#        def hover(hover_color="#ffff99"):
#            return dict(selector="tr:hover",
#                        props=[("background-color", "%s" % hover_color)])
#
#        styles = [hover()]
        return str(self.df.to_html(index=False, index_names=False, header=False))
#        df.style.set_properties(color="white", align="right")


class MolMap():
    def __init__(self, **kwargs):
        defaults = {'sub': 6,
                    'rad_type': 'vdwrad',  # Chose from covrad, atmrad, vdwrad
                    # TODO implement custom atomic radii
                    'rad_scale': 1.0,  # Scale factor of chosen rad_type
                    'radius': 0.,
                    'excludes': [],
                    'num_angles': 1,}
        self.__dict__.update(defaults)
        self.__dict__.update(kwargs)

    @lazyattr
    def excluded_anums(self):
        return [ELEMENTS[symbol].number for symbol in self.excludes]

    @lazyattr
    def path(self):
        return os.path.join(self.fold, self.file_name)

    @lazyattr
    def geom(self):
        return icosaio.getxyz(self.path)

    @lazyattr
    def coords(self):
        return self.geom[:,1:]

    @lazyattr
    def anums(self):
        return self.geom[:,0]

    @lazyattr
    def symbols(self):
        return np.array([ELEMENTS[anum].symbol for anum in self.anums])

    @lazyattr
    def atomrads(self):
        if self.rad_type == 'vdwrad':
            return np.array([self.rad_scale * ELEMENTS[anum].vdwrad for anum in self.geom[:, 0]])
        elif self.rad_type == 'covrad':
            return np.array([self.rad_scale * ELEMENTS[anum].covrad for anum in self.geom[:, 0]])
        elif self.rad_type == 'atmrad':
            return np.array([self.rad_scale * ELEMENTS[anum].atmrad for anum in self.geom[:, 0]])

    @lazyattr
    def centrum(self):
        """Return the proprties of the central atom.

        Returns: (int:atom_number, str:symbol, np.array:coords)
        """
        i = self.centrum_num
        return [i, self.symbols[i - 1], self.coords[i - 1]]

    @lazyattr
    def excludemask(self):
        return np.in1d(self.geom[:,0], self.excluded_anums)

    def get3coneatms(self, p):
        mxgeom = self.xgeom[~self.mask]
        mc_atoms = self.c_atoms[~self.mask]
        p_angles = np.arccos(np.clip(np.dot(mxgeom[:, 1:], p), -1, 1))
#        Using the formula angle = 2 * arcsin(chord / (2*R)):
        atom_view_angles = 2 * np.arcsin(mc_atoms / 2)
        nearest3angles = np.argpartition(p_angles - atom_view_angles, 3)
#        return the indices relative to the original geom
        return np.arange(len(self.geom))[~self.mask][nearest3angles[:3]]
#        return mxgeom[nearest3angles]

    def get_xgeom(self):
        mask = self.excludemask.copy()
        n = self.centrum[0]
        atomrads = self.atomrads
        radius = self.radius
        mask[n - 1] = True
        xgeom = self.geom.copy()
        xgeom[:,1:] = self.geom[:,1:] - self.geom[n - 1,1:]
        distances = np.linalg.norm(xgeom[:, 1:], axis=1)
#        xatomrads = self.atomrads[~mask]
        with np.errstate(divide='ignore'):
            vperd = atomrads / distances
#        If an atom is closer to centrum than its vdW radius, make its half apperture 90 degrees
        vperd[vperd > 1.0] = 1.0
        half_apertures = np.arcsin(vperd)
#        print('\ngamma:', half_apertures[0])
        if radius > 0.0:
            for i in range(len(xgeom)):
                if distances[i] >= radius + atomrads[i]:
                    mask[i] = True  # Mask out (erase) atoms further than R + atomrad
                elif distances[i] < math.sqrt(radius ** 2 + atomrads[i] ** 2):
                    pass  # Atoms this close are already computed in the simple arcsin(vperd) way
                else:
                    half_apertures[i] = math.acos((radius**2 + distances[i]**2 - atomrads[i]**2) /
                                                  (2 * radius * distances[i]))
#                    This is the twighlight zone. Atoms at this special distance need special
#                    treatment. The radius of the observation zone cuts into their
#                    atomrad sphere in such a way that the remaining object's half aperture becomes
#                    smaller than the arcsin(vperd) value. The length of the chord is used in the
#                    formula.
        self.mask = mask
        with np.errstate(divide='ignore', invalid='ignore'):
            xgeom[:, 1:] = np.divide(xgeom[:, 1:], distances[:, np.newaxis])
            c_atoms = 2 * np.sin(np.divide(half_apertures, 2))
        self.xgeom = xgeom
        self.c_atoms = c_atoms
        return xgeom, c_atoms


def calc(**kw):
    # Sanitize input arguments
    debug_level = int(kw.get('debug_level', 0))
    kwargs = {}
    file_names = [str(file_name) for file_name in kw['file_names']]
    atom_types = [str(atom_type) for atom_type in kw['atom_types']]
    centrum_nums = [list(set([int(num) for num in nums if int(num) != 0])) for nums in kw['centrum_nums']]
    kwargs['fold'] = os.path.abspath(str(kw['fold']))
    sub = kwargs['sub'] = int(kw['sub'])
    rad_type = kwargs['rad_type'] = str(kw['rad_type'])
    rad_scales = [float(rad_scale) for rad_scale in kw['rad_scales']]
    radii = [[float(radius) for radius in radiuses] for radiuses in kw['radii']]
    kwargs['excludes']= excludes = [str(element) for element in kw['excludes']]
    num_angles = [int(num) for num in kw['num_angles']]
    out = os.path.abspath(os.path.join(str(kw['output_folder']), str(kw['output_name'])))
    logger = Logger(out=out, debug_level=debug_level, table=str(kw.get('table', False)))
    log = logger.log
    if debug_level:
        print('debug_level', debug_level)
        print('generating icosphere...')
#    atomtype = 'special1'
    start0 = time.time()
    sphere = ico.IcoSphere(sub, 1, hdf5_path=str(kw.get('hdf5_path', '../progdata')))
    end = time.time()
    verts = sphere.verts
    numverts = len(verts)
    edge_length = 1 / math.sin(2 * math.pi / 5) / 2 ** sub
    if debug_level:
        print('icosphere done in {} s'.format(end - start0))
        print('egde length = {}'.format(edge_length))
        print('number of vertices: {}'.format(numverts))
#    generating KD-tree:
    start = time.time()
    tree = sp.spatial.cKDTree(verts, leafsize=32)
    end = time.time()
    if debug_level:
        print('KD-tree done in', end - start, 's\n')
        print('sub', sub)
    start = time.time()
    if debug_level:
        print('processing...')
        print(file_names, atom_types, centrum_nums, rad_scales, radii, num_angles)
    for file_name, atom_type, cent_nums, rad_scale, radiuses, num_angle in zip(
      file_names, atom_types, centrum_nums, rad_scales, radii, num_angles):
        kwargs['file_name'] = file_name
        kwargs['atom_type'] = atom_type
        kwargs['num_angle'] = num_angle
        log(tag='file_name', value=[file_name])
        log(tag='sub', value=[sub])
        if debug_level:
            print('processing', file_name)
        for radius in radiuses:
            kwargs['radius'] = radius
            kwargs['rad_scale'] = rad_scale
#            TODO: replace tags, with their key value of the `tags` dictionary of the `Logger`
            log(tag='radius', value=[radius])
            log(tag='rad_type', value=[rad_type])
            log(tag='atom_scale', value=[rad_scale])
            log(tag='excludes', value=excludes)
            for centrum_num in cent_nums:
                kwargs['centrum_num'] = centrum_num
                mm = MolMap(**kwargs)
                mm.get_xgeom()
                hits = array.array('L')
                for atom, c_atom in zip(mm.xgeom[~mm.mask], mm.c_atoms[~mm.mask]):
                    hits.extend(tree.query_ball_point(atom[1:], c_atom))
                hits = np.unique(np.array(hits, dtype=np.uint32))
                numhits = len(hits)
                coverage = numhits / numverts
                log(tag='centrum', value=mm.centrum[1::-1])
                log(tag='coverage', value=[coverage])
                if coverage == 1.0:
                    log(tag='angle', value=[0.])
                    log(tag='cone_atoms', value=[np.NaN])
                    continue
                for a in range(num_angle):
                    if a > 0:
                        hits = np.union1d(hits, tree.query_ball_point(point_a, chord))
                    hit_tree = sp.spatial.cKDTree(verts[hits, :], leafsize=32)
                    misses = np.setdiff1d(np.arange(numverts, dtype=np.uint32),
                                          hits,
                                          assume_unique=True)
#                    Find all the minimum distances between missed points and hit points.
                    min_lengths = hit_tree.query(verts[misses, :])
#                    The maximum of these minimum distances is equal to the radius of the inscribed
#                    circle of the "biggest" missed patch on the sphere. Get the index of this point on
#                    the sphere of observation.
#                    TODO: Find the top 3 patches and return the 3 defining atoms of their cones
                    maxminpos = np.argmax(min_lengths[0])
                    point_a = verts[misses, :][maxminpos, :]
                    rads3 = hit_tree.query(point_a, k=3)
                    chord = np.average(rads3[0])
                    half_app = math.asin(chord / 2)
                    half_app_deg = half_app / math.pi * 180.
                    try:
                        atoms = mm.get3coneatms(point_a)
                        idxs = [ELEMENTS[mm.geom[i, 0]].symbol + str(i + 1) for i in atoms]
                    except ValueError:
                        idxs = [['']]
                    # angle in degrees!!!
                    log(tag='angle', value=[half_app_deg])
                    log(tag='cone_atoms', value=idxs)
    end = time.time()
    if debug_level:
        print('All done in {} s'.format(end - start0))
    return logger.close()


def main(**new_kwargs):
    '''Tester function.
    '''
#TODO: minimal angle of substrate/solvent molecule
#TODO: xlsxwriter
#TODO: error estimation, error propagation study

    kwargs = {'file_names': ['iprc.xyz', 'iprc2.xyz'],
              'atom_types': ['Pt'] * 2,
              'centrum_nums': [[5, ]] * 2,
              'fold': os.path.abspath('../demo_molecules'),
              # 'fold': os.path.abspath('D:\myworkspace\getMolMap\moldata\\20140816_gesub'),
              'sub' : 6,
              'rad_type' : 'covrad', # Chose from covrad, atmrad, vdwrad
              'rad_scales' : [1.17] * 2, # Scale factor of chosen rad_type
              'radii': [[0.,]] * 2,
              'excludes' : ['H', ],
              'num_angles': [2] * 2,
              'output_folder': '../results',
              'output_name': 'getmolmap_results',
              'table': True,
              'debug_level': 1,}

    kwargs.update(new_kwargs)
    return calc(**kwargs)

if __name__ == '__main__':
    main()

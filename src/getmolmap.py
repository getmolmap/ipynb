#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2013.10.18.

@author: András Olasz
@version: 2016.01.05.dev.2

"""
#TODO: 3. Egyéni atomi rádiuszok

import time
import math
import os
import array
from collections import defaultdict
import numpy as np
# import numpy.ma as ma
import scipy
import scipy.spatial
import xlsxwriter
# from scipy.spatial import cKDTree
# from scipy.spatial import distance as sp_dis
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


class Logger():
    def __init__(self, **kw):
        self.debug_level = kw.get('debug_level', 0)
        out = kw.get('out', '../result/getmolmap_results') + '.xlsx'
        sheet_name = kw.get('sheet_name', 'getMolMap')
        self.workbook = xlsxwriter.Workbook(out)
        self.worksheet = self.workbook.add_worksheet(sheet_name)
        self.row = 0
        tags = ['File Name', 'IcoSphere Subdivision', 'Cut Radius', 'Type of Atomic Radii',
                'Scale Factor of Atomic Radii', 'Excluded Elements', 'Buried Volume',
                'Centrum', 'Atoms on the Inverse Cone', 'Inverse Cone Angle']
        self.tags = tags
        self.decimals = defaultdict(int, {tags[2]: 3, tags[4]: 2, tags[6]: 3, tags[9]: 1})

    def log(self, **kw):
        tag = kw['tag']
        value = kw['value']
        self.worksheet.write(self.row, 0, tag)
        self.decimals.get
        decimals = self.decimals[tag]
        if decimals:
            num_format = self.workbook.add_format()
            num_format.set_num_format('0.{}'.format('0' * decimals))
            self.worksheet.write(self.row, 1, value[0])
        elif tag in ('Excluded Elements', 'Atoms on the Inverse Cone'):
            for i, v in enumerate(value):
                self.worksheet.write(self.row, i + 1, value[0])
        else:
            self.worksheet.write(self.row, 1, value[0])
        if self.debug_level:
            print('{}:'.format(kw['tag']), *kw['value'])
        self.row += 1

    def close(self):
        self.workbook.close()


def calc(**kw):
    # Sanitize input arguments
    debug_level = str(kw.get('debug_level', 0))
    kwargs = {}
    file_names = [str(file_name) for file_name in kw['file_names']]
    atom_types = [str(atom_type) for atom_type in kw['atom_types']]
    centrum_nums = [[int(num) for num in nums] for nums in kw['centrum_nums']]
    kwargs['fold'] = os.path.abspath(str(kw['fold']))
    sub = kwargs['sub'] = int(kw['sub'])
    rad_type = kwargs['rad_type'] = str(kw['rad_type'])
    rad_scales = [float(rad_scale) for rad_scale in kw['rad_scales']]
    radii = [[float(radius) for radius in radiuses] for radiuses in kw['radii']]
    excludes = [[str(ex) for ex in exclude] for exclude in kw['excludes']]
    num_angles = [int(num) for num in kw['num_angles']]
    out = os.path.abspath(os.path.join(str(kw['output_folder']), str(kw['output_name'])))
    logger = Logger(out=out, debug_level=debug_level)
    log = logger.log
    if debug_level:
        print('generating icosphere...')
#    atomtype = 'special1'
    start = time.time()
    sphere = ico.IcoSphere(sub, 1)
    end = time.time()
    verts = sphere.verts
    numverts = len(verts)
    edge_length = 1 / math.sin(2 * math.pi / 5) / 2 ** sub
    if debug_level:
        print('icosphere done in {} s'.format(end - start))
        print('egde length = {}'.format(edge_length))
        print('number of vertices: {}'.format(numverts))
#    generating KD-tree:
    start = time.time()
    tree = sp.spatial.cKDTree(verts, leafsize=32)
    end = time.time()
    if debug_level:
        print('KD-tree done in', end - start, 's\n')
    start = time.time()
    for file_name, atom_type, cent_nums, rad_scale, radiuses, excluded, num_angle in zip(
      file_names, atom_types, centrum_nums, rad_scales, radii, excludes, num_angles):
        kwargs['file_name'] = file_name
        kwargs['atom_type'] = atom_type
        kwargs['excludes'] = excluded
        kwargs['num_angle'] = num_angle
        log(tag='File Name', value=[file_name])
        log(tag='IcoSphere Subdivision', value=[sub])
        for radius in radiuses:
            kwargs['radius'] = radius
            kwargs['rad_scale'] = rad_scale
            log(tag='Cut Radius', value=[radius])
            log(tag='Type of Atomic Radii', value=[rad_type])
            log(tag='Scale Factor of Atomic Radii', value=[rad_scale])
            log(tag='Excluded Elements', value=excluded)
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
                log(tag='Buried Volume', value=[coverage])
                log(tag='Centrum', value=[mm.centrum[1::-1]])
                if coverage == 1.0:
                    log(tag='Atoms on the Inverse Cone', value=[np.NaN])
                    log(tag='Inverse Cone Angle', value=[0.])
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
                    log(tag='Atoms on the Inverse Cone', value=idxs)
                    log(tag='Inverse Cone Angle', value=[half_app_deg])
    end = time.time()
    if debug_level:
        print('All done in {} s'.format(end - start))
    logger.close()


def main(**new_kwargs):
    '''Tester function.
    '''
#TODO: minimal angle of substrate/solvent molecule
#TODO: xlsxwriter
#TODO: error estimation, error propagation study
    import os

    kwargs = {'file_names': ['iprc.xyz', 'iprc2.xyz'],
              'atom_types': ['Pt'] * 2,
              'centrum_nums': [[5, ]] * 2,
              'fold': os.path.abspath('../demo_molecules'),
              # 'fold': os.path.abspath('D:\myworkspace\getMolMap\moldata\\20140816_gesub'),
              'sub' : 7,
              'rad_type' : 'covrad', # Chose from covrad, atmrad, vdwrad
              'rad_scales' : [1.17] * 2, # Scale factor of chosen rad_type
              'radii': [[0.,]] * 2,
              'excludes' : [['H'], ] * 2,
              'num_angles': [2] * 2,
              'output_folder': '../results',
              'output_name': 'getmolmap_results',
              'debug_level': 1,}

    kwargs.update(new_kwargs)
    calc(**kwargs)

if __name__ == '__main__':
    main()

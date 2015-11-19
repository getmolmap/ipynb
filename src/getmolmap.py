#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on 2013.10.18.

@author: András Olasz
@version: 2015.07.15.dev.1

Legnagyobb részecske:
Megkeressük az összes kontúron lévő 3szöget. (az, amelyik fehér és van fekete szomszédja).
És a maradékon újra megkeressük a kontúrokat. Az utolsó fehér pont a nyerő. Erre megnézzük az
összes távolságot az eredeti kontúrtól.
'''

#TODO: 1. Atomi rádiuszok skálafaktorral
#TODO: 2. Egyéb atomi rádiuszok
#TODO: 3. Egyéni atomi rádiuszok

import time
import math
import os
import array
import numpy as np
import numpy.ma as ma
import scipy
import scipy.spatial
# from scipy.spatial import cKDTree
# from scipy.spatial import distance as sp_dis
from elements import ELEMENTS
import icosaio
import icosphere as ico
# import scipy.spatial as sp
sp = scipy

def atomfilter(geom, atomtype):
    anums = geom[:, 0]
    symbols = [ELEMENTS[anum].symbol for anum in anums]
    coords = geom[:, 1:]
    results = []
    if atomtype == 'special1':
        #Find Pd atoms, and  the closest atom to Pd
        for i, symbol, coord in zip(range(len(symbols)), symbols, coords):
            if symbol.lower() == 'pd':
                results.append((i+1, symbol, coord))
        nearest = results[0]
        mindist = 10.0
        for i, symbol, coord in zip(range(len(symbols)), symbols, coords):
            dist = distance(results[0][2], coord)
            if symbol.lower() != 'pd' and dist < mindist:
                mindist = dist
                nearest = (i+1, symbol, coord)
#                 print(symbol.lower(), i+1, mindist)
        if nearest[1].lower() not in ('si', 'c', 'p'):
            print('Warning, nearest atom to Pd is:', nearest[1])
        results.append(nearest)
    elif atomtype in ['Si', 'Ge']:
        results = []
        carbons = []
        for i, symbol, coord in zip(range(len(symbols)), symbols, coords):
            if symbol == atomtype:
                results.append((i+1, symbol, coord))
            elif symbol == 'C':
#                 print('Carbon found.')
                carbons.append((i+1, symbol, coord))
        if len(results) <= 2:
            return results
        iso_coords = [atom[2] for atom in results]
        carbon_coords = [atom[2] for atom in carbons]
        maxbond = ELEMENTS[atomtype].vdwrad * 1.3
        maxbondC = 2.2
        nCneighbors = np.zeros(len(results))
        selfneighbors = np.zeros(len(results))
        if len(carbon_coords) > 1:
            for i, atom in zip(range(len(results)), results):
                coord = atom[2]
                dists = sp.spatial.distance.cdist(np.array([coord]), np.array(carbon_coords))
                nCneighbors[i] = len(dists[dists <= maxbondC])
            if nCneighbors.min() == 0:
                results = [result for i, result in enumerate(results) if nCneighbors[i] == 0]
                return results
        for i, atom in zip(range(len(results)), results):
            coord = atom[2]
            coords2 = iso_coords.copy()
            del coords2[i]
            dists = sp.spatial.distance.cdist(np.array([coord]), np.array(coords2))
            n = len(dists[dists < maxbond])
            selfneighbors[i] = n
        results = [result for i, result in enumerate(results) if selfneighbors[i] > 0]
    else:
        for i, symbol, coord in zip(range(len(symbols)), symbols, coords):
            if symbol == atomtype:
                results.append((i+1, symbol, coord))
    return results

def distance(a, b):
    return math.sqrt(sum([(b[i] - a[i])**2 for i in range(len(a))]))

def get_polar(verts):
    xyz = np.array(verts)
    ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[:,0]**2 + xyz[:,1]**2
    #ptsnew[:,3] = np.sqrt(xy + xyz[:,2]**2)
    ptsnew[:,4] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    ptsnew[:,5] = np.arctan2(xyz[:,1], xyz[:,0])
    return ptsnew

def polar_ang_diff(p1, p2):
    theta = p2[0] - p1[0]
    if theta < 0.:
        theta = theta + math.pi
    phi = p2[1] - p1[1]
    if phi > math.pi:
        phi - math.pi * 2.0
    elif phi < -1.0 * math.pi:
        phi + math.pi * 2.0
    return [phi, theta]

def get_boundary_nodes(G, pentagons):
#     if len(G) == 88:
#         print(G.degree())/
    return [i[0] for i in G.degree_iter() if i[1] < 6] #or (i[1] in pentagons and i[1] < 5)]

def mask_centrum(geom, n, atomtype=None):
    symbols = geom[0]
    coords = geom[1]
    xcoords = ma.array(coords)
    xcoords[n - 1] = ma.masked
    return(symbols, xcoords)

def move_origin_to_centrum(geom, n, atomtype=None):
    xgeom = geom.copy()
    centrum = xgeom[n -1]
    xgeom[:, 1:] = xgeom[:, 1:] - centrum[1:]
#     xgeom = ma.array(xgeom)
#     xgeom[n - 1] = ma.masked
#     print(xgeom)
    xgeom = np.delete(xgeom, n - 1, axis=0) #remove centrum instead of masking it out
    return xgeom

def get_xgeom(geom, n, atomtype, radius):
    xgeom = move_origin_to_centrum(geom, n, atomtype)
    distances = np.linalg.norm(xgeom[:, 1:], axis=1)
#         print('\ndistances:', distances[0])
    vdwrads = np.array([ELEMENTS[anum].vdwrad for anum in xgeom[:, 0]])
    vperd = vdwrads / distances
    #If an atom is closer than its vdW radius, make its half apperture 90 degrees
    vperd[vperd > 1.0] = 1.0
#         print('\nvdwrads:', vdwrads / distances)
#         for i in range(len(vdwrads)):
#             _x = np.arcsin(vdwrads[i] / distances[i])
#             if np.isnan(_x):
#                 print('\nXXX', vdwrads[i], distances[i], vdwrads[i] / distances[i])
    half_apertures = np.arcsin(vperd)
#         print('\ngamma:', half_apertures[0])
    if radius > 0.0:
        for i in range(len(xgeom)):
            if distances[i] >= radius + vdwrads[i]:
                half_apertures[i] = 0.0
            elif distances[i] < math.sqrt(radius ** 2 + vdwrads[i] ** 2):
                pass
            else:
                half_apertures[i] = math.acos((radius**2 + distances[i]**2 - vdwrads[i]**2) /
                                              (2 * radius * distances[i]))
#               half_alpha = np.arccos((radius**2 + vdwrads**2 - distances**2) / (2 * radius * vdwrads))
    xgeom[:, 1:] = xgeom[:, 1:] / distances[:, np.newaxis]
    c_vdws = 2 * np.sin(half_apertures / 2)

    return xgeom, c_vdws

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
                    'radii': [0.,],
                    'excludes': [],
                    'num_angles': 1,}
        self.__dict__.update(defaults)
        self.__dict__.update(kwargs)

    @lazyattr
    def excluded_anums(self):
        return [ELEMENTS[symbol].number for symbol in self.excludes]

    @lazyattr
    def path(self):
        return os.path.join(self.fold, self.fn)

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
    def centrums(self):
        return atomfilter(self.geom, self.atomtype)

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
        #return the indices relative to the original geom
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
#                    special treatment. The radius of the observation zone cuts into their
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

def getmolmap(**kw):
    fold = kw['fold']
    sub = int(kw['sub'])
    print('generating icosphere...')
#    atomtype = 'special1'
    start = time.time()
    sphere = ico.IcoSphere(sub, 1)
    end = time.time()
    verts = sphere.verts
    numverts = len(verts)
    edge_length = 1 / math.sin(2 * math.pi / 5) / 2 ** sub
    print('icosphere done in {} s'.format(end - start))
    print('egde length = {}'.format(edge_length))
    print('number of vertices: {}'.format(numverts))
#    generating KD-tree:
    start = time.time()
    tree = sp.spatial.cKDTree(verts, leafsize=32)
    end = time.time()
    print('KD-tree done in', end - start, 's\n')
    start = time.time()
    for fn in os.listdir(fold):  # ['X903G1.xyz']:
        mm = MolMap(**kw)
        mm.geom = icosaio.getxyz(os.path.join(fold, fn), mm.excludes)
        centrums = atomfilter(mm.geom, mm.atomtype)
        for radius in mm.radii:
            mm.radius = radius
            print(fn, radius, end=' ')
            for centrum in centrums:
                mm.centrum = centrum
                n = centrum[0]
                centrum_symbol = centrum[1]
                print('{}{}'.format(centrum_symbol, n), end=' ')
                xgeom, c_vdws = mm.get_xgeom()
#                A hit is where the shadow cone of an atom is cast on the sphere of observation
                hits = array.array('L')
#                TODO: rewrite collecting hits as setting False values in a boolean array
#                Find all the points on the sphere of observation, which are in the collective
#                shadow of of shadow cones. A point can be counted several times.
                for atom, c_vdw in zip(xgeom[~mm.mask], c_vdws[~mm.mask]):
                    hits.extend(tree.query_ball_point(atom[1:], c_vdw))
#                    tree.query_ball_point          # delete this line?
#                Make a unique list of all points in the shadow (hits)
                hits = np.unique(np.array(hits, dtype=np.uint32))
                numhits = len(hits)
                coverage = numhits / numverts
                print('{}'.format(coverage), end=' ')
#                If coverage is 100% no need to calculate maximum uncovered area
                if coverage == 1.0:
                    print('0.0', end=' ')
                    continue
                for a in range(mm.num_angles):
                    if a > 0:
#                        r3 = []
#                        for i in range(3):
#                            mi = hit_tree.data[rads3[1][i], :]
#                            P1 = np.array([1,0,0]); P2 = np.array([2,0,0])
#                            r3.append(sp.spatial.distance.euclidean(point_a, mi))
#                        r3 = np.array(r3)
#                        r = np.average(r3)
#                        print('zero={}'.format(chord - r), end=' ')
#                        print('r={}'.format(r), end=' ')
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
                        idxs = ''.join([ELEMENTS[mm.geom[i, 0]].symbol + str(i + 1) for i in atoms])
                    except ValueError:
                        idxs = ''
                    print('angle{}={}'.format(idxs, half_app_deg), end=' ')  # angle in degrees!!!

            print('')
    end = time.time()
    print('All done in {} s'.format(end - start))


def main(**new_kwargs):
    '''TODO:
    A különböző felosztások relatív hibáinak átlaga és maximuma.
    >>>> Analitikus finomítás
    >>>> xls

    Az atomok projekciójánál az adott atom pontja körül addig irtjuk a pontokat nx.node_boundary-val,
    amíg a boundary node hibahatáron belül közelíti a projektált vdW sugarat.
    Esetleg kapásból kivághatunk egy theta+-rádiusz, fi+-rádiusz darabot és csak erre vizsgálódunk.
    A pontokat sorba lehetne rendezni növekvő theta, phi sorrendben. Sőt, esetleg analitikusan is
    leképezhetők valahogy.
    '''
#TODO: include H atoms
#TODO: oldószer minimális szöge

    import os

    kwargs = {'atomtype' : 'Ge',
              'fold' : os.path.abspath('D:\myworkspace\getMolMap\moldata\\20140816_gesub'),
              'sub' : 6,
              'rad_type' : 'covrad', # Chose from covrad, atmrad, vdwrad
              'rad_scale' : 1.17, # Scale factor of chosen rad_type
              'radii': [0.,],
              'excludes' : [],
              'num_angles': 1}

    kwargs.update(new_kwargs)
    getmolmap(**kwargs)

if __name__ == '__main__':
    main()
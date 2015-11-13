#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 2013.10.25.

@author: András Olasz
@version: 2015.05.21.dev.1

This is a python/numpy remake of the C# code of Andreas Kahler
@see: http://blog.andreaskahler.com/2009/06/creating-icosphere-mesh-in-code.html
'''

import numpy as np
import numpy.ma as ma
import math
import time
from itertools import combinations

# def normalize(a):
#     for v in a:

class IcoSphere():
    '''
    Return the vertices and faces of an icosphere as two numpy arrays. Vertices lie on a unit
    sphere
    Spherical coordinates convention follows geographic notation (r, theta, lambda), where
    r: positive real, theta: +-pi/2 from xy plane to z+, lamdba: +-pi from xz plane to y+.
    '''
    def __init__(self, subdivision=2, size=1.0):
        self._create_icosahedron()
        self.subdivision = 0
        self.subdivide(subdivision)

    def _create_icosahedron(self):
        '''
        Return the vertices and faces of an icosahedron as two numpy arrays.
        '''
#         t1 = time.time()
        t = (1.0 + math.sqrt(5.0)) / 2.0
        verts = ma.array([
                          [-1,  t,  0],
                          [ 1,  t,  0],
                          [-1, -t,  0],
                          [ 1, -t,  0],
                          [ 0, -1,  t],
                          [ 0,  1,  t],
                          [ 0, -1, -t],
                          [ 0,  1, -t],
                          [ t,  0, -1],
                          [ t,  0,  1],
                          [-t,  0, -1],
                          [-t,  0,  1],
                          ], dtype=np.float64)
        #Let's put the vertices on the unitsphere
        length = math.sqrt(t * t + 1)
        self.verts = verts / length
        #Create faces from triangles
        self.faces = ma.array([
                              [ 0, 11,  5], #Five faces around point 0
                              [ 0,  5,  1],
                              [ 0,  1,  7],
                              [ 0,  7, 10],
                              [ 0, 10, 11],
                              [ 1,  5,  9], #Adjacent faces to the previous 5 faces
                              [ 5, 11,  4],
                              [11, 10,  2],
                              [10,  7,  6],
                              [ 7,  1,  8],
                              [ 3,  9,  4], #Five faces around point 3 which is opposite to point 0
                              [ 3,  4,  2],
                              [ 3,  2,  6],
                              [ 3,  6,  8],
                              [ 3,  8,  9],
                              [ 4,  9,  5], #Adjacent faces to the previous 5 faces
                              [ 2,  4, 11],
                              [ 6,  2, 10],
                              [ 8,  6,  7],
                              [ 9,  8,  1],
                               ], dtype=np.uint8)
#         t2 = time.time()
#         print('frequency: 0')
#         print('vertex count:', len(self.verts))
#         print('face count:', len(self.faces))
#         print('çreation time:', t2 - t1, '\n')

    def _create_icosahedron_spherical(self):
        '''
        Return the vertices and faces of an icosahedron as two numpy arrays in spherical
        coordinates (theta and phi only, no radius).
        '''
        Pi = math.pi
        G = 0.2 * math.pi
        T = math.atan(0.5)
        verts = ma.array([
                      [ Pi, 0],
                      [  T, -4 * G],
                      [  T, -2 * G],
                      [  T,  0 * G],
                      [  T,  2 * G],
                      [  T,  4 * G],
                      [ -T, -5 * G],
                      [ -T, -3 * G],
                      [ -T, -1 * G],
                      [ -T,  1 * G],
                      [ -T,  3 * G],
                      [-Pi,  0],
                      ], dtype=np.float64)

    def _blender_plot(self):
        import bpy
#         import bmesh
#         verts, faces = readmesh(r'd:\workspace\icosapy\results\test.mesh')
#         D = bpy.data
#         C = bpy.context
        verts = [[float(v) for v in list(vert)] for vert in self.verts]
        faces = [[int(f) for f in list(face)] for face in self.faces]
        me = bpy.data.meshes.new("MyIcosphereMesh")    # create a new mesh
        ob = bpy.data.objects.new("MyIcosphere", me)   # create an object with that mesh
        ob.location = bpy.context.scene.cursor_location   # position object at 3d-cursor
        bpy.context.scene.objects.link(ob)                # Link object to scene
        # Fill the mesh with verts, edges, faces
        me.from_pydata(verts, [], faces)      # edges or faces should be [], or you ask for problems
        me.update(calc_edges=True)            # Update mesh with new data
        self.me = me
        self.ob = ob

    def _blender_delete(self):
        import bpy
        #select object
        self.ob.select = True
        bpy.ops.object.delete()

#         self.obj.user_clear()
#         bpy.context.scene.objects.active = self.obj # Change active object
#         bpy.ops.object.delete() # Delete object
#         bpy.data.objects.remove(self.obj) # Delete object on datablock
#         self.me.user_clear()
#         bpy.data.meshes.remove(self.me)

    def subdivide(self, subdivision):
        for _s in range(int(subdivision)):
#             t1 = time.time()
            self.subdivision = self.subdivision + 1
            numverts = len(self.verts)
            numfaces = len(self.faces)
            numfaces2 = numfaces * 4        #Each subdivision creates four triangles in place of one.
#             numedges2 = numfaces2 * 3 / 2 #Each triangle has three sides and each side is shared by
#                                           #two triangles
            numverts2 = numverts * 4 - 6    #Euler's Formula: F+V-E = 2
            #Chose an unsigned integer type that will be enough to count all vertices
            #It will give you an error if even uint64 is not enough to count all verts
            self._shift = int(math.log(numverts2, 2)) + 1
            for n, np_type in zip((8, 16, 32, 64), (np.uint8, np.uint16, np.uint32, np.uint64)):
                if self._shift <= n:
                    break
            oldfaces = self.faces.copy()
            self.faces = np.empty([numfaces2, 3], dtype=np_type)
            self._vertindex = numverts - 1
#             print('frequency:', self.subdivision)
#             print('vertex count:', numverts2)
#             print('face count:', numfaces2)
            self.verts = np.resize(self.verts, (numverts2, 3))
            self._vertcache = {}
            #Calculate length of the new midpoint vectors every time otherwise numerical noise
            #propagates to significant figures after a few subdivisions
            for i, face in enumerate(oldfaces):
#                 print('new face, i:', i)
                midpoints = [self.getmiddlepoint(*sorted(p)) for p in combinations(face, 2)]
                newfaces = (
                            (midpoints[0], midpoints[1], face[0]),
                            (midpoints[1], midpoints[2], face[2]),
                            (midpoints[0], midpoints[2], face[1]),
                            (midpoints[0], midpoints[1], midpoints[2]),
                            )
                self.faces[i*4:i*4+4] = newfaces
        del oldfaces; del self.faces
#             t2 = time.time()
#             print('creation time:', t2 - t1, '\n')
#                 if i == 20:
#                     break
#             self._blender_delete()
#         dist = spdist.cdist(self.verts, np.array([[0., 0., 0.]], dtype=np.float64), 'euclidean')
#         dist = spdist.squareform(cond_dist)
#         print(dist)
#         print('mean:', dist.mean())
#         print('var:', dist.var())
#         print('plotting...')
#         self._blender_plot()
#         print('plotting done', '\n'*5, )
#             time.sleep(3)


    def getmiddlepoint(self, p1, p2):
        '''
        important: p1 < p2
        '''
#         print('midpoint between', p1, 'and', p2, ':')
#         print('vertindex', self._vertindex)
        index = (p1 << self._shift) + p2
#         print('index', index)
        midpoint = self._vertcache.get(index, False)
#         print('midpoint', midpoint)
        if midpoint:
            return midpoint
        else:
            self._vertindex += 1
            newpoint = (self.verts[p1] + self.verts[p2]) / 2.0
            self.verts[self._vertindex] = (newpoint) / np.linalg.norm(newpoint)
            self._vertcache[index] = self._vertindex
            return self._vertindex

    def get_spherical(self):
        #replace math functions with numpy equivalents
        self.spherical = np.empty([len(self.verts), 2], dtype=np.float64)
        # The radial distance is always 1.
        # The polar angle (theta).
        #theta = math.acos(self.verts[2])
        np.arccos(self.verts[:, 2], self.spherical[:, 0])
        # The azimuth.
        # phi = math.atan2(self.verts[1], self.verts[0])
        np.arctan2(self.verts[:, 1], self.verts[:, 0], self.spherical[:, 1])


    def get_angle(self, p1, p2):
        """ Returns the angle in radians between vectors 'v1' and 'v2':: >>>
        angle_between((1, 0, 0), (0, 1, 0)) 1.5707963267948966 >>>
        angle_between((1, 0, 0), (1, 0, 0)) 0.0 >>>
        angle_between((1, 0, 0), (-1, 0, 0)) 3.141592653589793 """
        v1, v2 = self.verts[p1], self.verts[p2]
        angle = np.arccos(np.dot(v1, v2))
        if np.isnan(angle):
            if (v1 == v2).all():
                return 0.0
            else:
                return np.pi
        return angle
#        np.isclose(a, b, rtol, atol, equal_nan)


def main():
    t1 = time.time()
    ico = IcoSphere(7, 1)
    t2 = time.time()
#     print(len(ico.faces))
#     print(len(ico.verts))
    print('subdivision done in', t2 - t1, 'sec')
    t1 = time.time()
    ico.xyz2spherical()
    t2 = time.time()
    print('spherical transformation done in', t2 - t1, 'sec')
if __name__ == '__main__':
    main()

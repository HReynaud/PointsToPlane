import os
import numpy as np
import sys
import time
import subprocess
import SimpleITK as sitk
from PIL import Image

def saveJPG(arr, path):
    arr = arr.astype(np.uint8)
    Image.fromarray(arr).save(path, quality=100)

class Point():
    def __init__(self, x, y, z, arr):
        self._x = x
        self._y = y
        self._z = z

        sx, sy, sz = arr.shape

        self.max_x = sx
        self.max_y = sy
        self.max_z = sz

    def __str__(self):
        return "Point ("+str(self._x)+","+str(self._y)+","+str(self._z)+")"

    def p_coords(self):
        return np.array([self._x, self._y, self._z])
    
    def coords(self):
        return np.array([self._x*self.max_x, self._y*self.max_y, self._z*self.max_z])
    
    def rot_ax(self, num=1):
        for _ in range(num):
            tmp = self._x
            self._x = self._y
            self._y = self._z
            self._z = tmp
        return self

    @property
    def x(self):
        return self._x*self.max_x
    
    @property
    def y(self):
        return self._y*self.max_y
    
    @property
    def z(self):
        return self._z*self.max_z

def draw_sphere(arr, point, r=3, color = 255):
    i,j,k = np.indices(arr.shape)
    dist = np.sqrt( (point.x-i)**2 + (point.y-j)**2 + (point.z-k)**2)
    arr[dist < r] = color
    return arr

def get_plane_coefs(pointA, pointB, pointC):
    p1 = pointA.coords()
    p2 = pointB.coords()
    p3 = pointC.coords()

    # These two vectors are in the plane
    v1 = p3 - p1
    v2 = p2 - p1

    # the cross product is a vector normal to the plane
    cp = np.cross(v1, v2)
    a, b, c = cp

    # This evaluates a * x3 + b * y3 + c * z3 which equals d
    d = np.dot(cp, p3)

    return a, b, c, d

def get_plane(arr, coefs, oob_black=True):
    a, b, c, d = coefs
    sx, sy, sz = arr.shape
    main_ax = np.argmax([abs(a), abs(b), abs(c)])

    if main_ax == 0:
        Y, Z = np.meshgrid(np.arange(sy), np.arange(sz), indexing='ij')
        X = (d - b * Y - c * Z) / a

        X = X.round().astype(np.int)
        P = X.copy()
        S = sx-1

        X[X <= 0] = 0
        X[X >= sx] = sx-1

    elif main_ax==1:
        X, Z = np.meshgrid(np.arange(sx), np.arange(sz), indexing='ij')
        Y = (d - a * X - c * Z) / b

        Y = Y.round().astype(np.int)
        P = Y.copy()
        S = sy-1

        Y[Y <= 0] = 0
        Y[Y >= sy] = sy-1
    
    elif main_ax==2:
        X, Y = np.meshgrid(np.arange(sx), np.arange(sy), indexing='ij')
        Z = (d - a * X - b * Y) / c

        Z = Z.round().astype(np.int)
        P = Z.copy()
        S = sz-1

        Z[Z <= 0] = 0
        Z[Z >= sz] = sz-1
    
    plane = arr[X, Y, Z]
    if oob_black == True:
        plane[P < 0] = 0
        plane[P > S] = 0

    return plane, (X, Y, Z)

def save_mpl(dotA, dotB, dotC, X, Y, Z):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X.flatten(),
            Y.flatten(),
            Z.flatten(), 'bo ')

    ax.plot(*zip(dotA.p_coords(), dotB.p_coords(), dotC.p_coords()), color='r', linestyle=' ', marker='o')
    ax.view_init(0, 22)
    plt.tight_layout()
    plt.savefig('/vol/biomedic3/hjr119/XCAT/samples/planeMPL.jpg')

ct_itk = sitk.ReadImage("/vol/biomedic3/hjr119/XCAT/generation/samp0_CT_1.nii.gz")
ct_arr = sitk.GetArrayFromImage(ct_itk)
# ct_arr = np.moveaxis(ct_arr, 0, -1)
sx, sy, sz = ct_arr.shape
ct_arr = ct_arr/ct_arr.max()*255
print(ct_arr.min(), ct_arr.max(), ct_arr.shape)


dotA = Point(0, 0, .5, ct_arr)
dotB = Point(0, .5, 0, ct_arr)
dotC = Point(.5, 0, 0, ct_arr)

if True:
    r = 5
    ct_arr = draw_sphere(ct_arr, dotA, r=r)
    ct_arr = draw_sphere(ct_arr, dotB, r=r)
    ct_arr = draw_sphere(ct_arr, dotC, r=r)

coefs = get_plane_coefs(dotA, dotB, dotC)
print(coefs)

plane, (X, Y, Z) = get_plane(ct_arr, coefs)
print(coefs, plane.shape)
ct_arr[X, Y, Z] = 255

# name = "plane_"+str(hA)[:3]+"_"+str(hB)[:3]+"_"+str(hC)[:3]+".jpg"
name = "tmp.jpg"
saveJPG(plane, os.path.join("/vol/biomedic3/hjr119/XCAT/samples",name))

saveJPG(ct_arr[sx//2, :, :], "/vol/biomedic3/hjr119/XCAT/samples/ax1.jpg")
saveJPG(ct_arr[:, sy//2, :], "/vol/biomedic3/hjr119/XCAT/samples/ax2.jpg")
saveJPG(ct_arr[:, :, sz//2], "/vol/biomedic3/hjr119/XCAT/samples/ax3.jpg")

# save_mpl(dotA, dotB, dotC, X, Y, Z)

print("done")


import os
import numpy as np
import sys
import time
import subprocess
import SimpleITK as sitk
from PIL import Image
from numpy.core.defchararray import translate
from numpy.core.fromnumeric import transpose
from numpy.lib.function_base import flip
from scipy import ndimage, misc, signal
from seg_processing import load_clean_seg
from skimage.draw import polygon, circle
import cv2

PATH="/vol/biomedic3/hjr119/XCAT/tmp_video/"
COUNT=1

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
        return "P("+str(np.round(self._x,3))+","+str(np.round(self._y,3))+","+str(np.round(self._z,3))+")"

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

class ImageTransformation():
    def __init__(self) -> None:
        self.tl = [] # transformation list

    def add(self, name, value):
        assert name in ['flip', 'rot']
        
        if len(self.tl) > 0 and self.tl[-1][0] == name:
            if name == 'flip':
                if value in self.tl[-1][1]:
                    self.tl[-1][1].remove(value)
                    if len(self.tl[-1][1]) == 0:
                        del self.tl[-1]
                else:
                    self.tl[-1][1].add(value)

            elif name == 'rot':
                self.tl[-1][1] = (self.tl[-1][1] + value)%4
                if self.tl[-1][1] == 0:
                    del self.tl[-1]

        else:
            if name == 'flip':
                self.tl.append([name,{value}])

            elif name == 'rot':
                self.tl.append([name,value])
     
    def __call__(self, img):
        for t in self.tl:
            if t[0] == 'flip':
                if 0 in t[1]:
                    img = np.flip(img, axis=0)
                if 1 in t[1]:
                    img = np.flip(img, axis=1)
            elif t[0] == 'rot':
                img = np.rot90(img, k=t[1])
        return img
    
    def __str__(self) -> str:
        string = []
        for i, t in enumerate(self.tl):
                string.append(f'{i}: {t[0]} {t[1]}')
        return str(string)

class PlaneSampler():
    def __init__(self, volume) -> None:
        self.volume = volume
        
        self.prev_ax = None
        self.coefs = None
        self.coordinates = None
        self.plane = None
        self.flips = []
        self.rots = 0
        self.transpose = False

        self.trans = ImageTransformation()
    
    def _get_plane_coefs(self, pointA, pointB, pointC):
        p1 = pointA.coords()
        p2 = pointB.coords()
        p3 = pointC.coords()

        # These two vectors are in the plane
        v1 = p3 - p1
        v2 = p2 - p1

        # The cross product is a vector normal to the plane
        cp = np.cross(v1, v2)
        a, b, c = cp

        # This evaluates a * x3 + b * y3 + c * z3 which equals d
        d = np.dot(cp, p3)

        self.coefs = a,b,c,d

        return a, b, c, d

    def _get_plane(self, coefs, oob_black=True, count=None):
        a, b, c, d = coefs
        sx, sy, sz = self.volume.shape
        main_ax = np.argmax([abs(a), abs(b), abs(c)])

        sa, sb, sc = np.roll([sx, sy, sz],shift=(2-main_ax))
        na, nb, nc = np.roll([a, b, c],   shift=(2-main_ax))        

        A,B = np.meshgrid(np.arange(sa), np.arange(sb), indexing='ij')
        C = (d - na * A - nb * B) / nc

        C = C.round().astype(np.int)
        P = C.copy()
        S = sc-1

        C[C <= 0]  = 0
        C[C >= sc] = sc-1

        # Solves weird issue with np.roll on list of arrays
        ABC = np.stack((A, B, C))
        idxX, idxY, idxZ = np.roll([0, 1, 2],shift=(main_ax-2))
        X, Y, Z = ABC[idxX], ABC[idxY], ABC[idxZ]

        plane = self.volume[X, Y, Z]

        if oob_black == True:
            plane[P < 0] = 0
            plane[P > S] = 0
        
        # Plane adjustements:
        if main_ax == 2:
            plane = np.rot90(plane, k=1)

        if self.prev_ax == 2 and main_ax==1:
            self.trans.add('flip', 0)
        elif self.prev_ax == 0 and main_ax==2:
            self.trans.add('flip', 0)
        elif self.prev_ax == 2 and main_ax==0:
            self.trans.add('flip', 0)
            self.trans.add('flip', 1)
        elif self.prev_ax == 0 and main_ax==1:
            if ['flip', {0}] in self.trans.tl:
                self.trans.add('rot', -1)
            else:
                self.trans.add('rot', 1)
        elif self.prev_ax == 1 and main_ax==0:
            self.trans.add('rot', -1)
            self.trans.add('flip', 0)

        # print(self.trans)
        plane = self.trans(plane)

        # if count is not None:
        #     # print(count," a",a," b",b," c",c," d",d," main_ax",main_ax, X[0,0],X[-1,-1],Y[0,0],Y[-1,-1],Z[0,0],Z[-1,-1])
        #     print(count," main_ax",main_ax, self.flips, self.rots)
        #     saveJPG(plane, os.path.join(PATH, ("000"+str(count))[-3:]+".jpg" ))

        self.last_plane = plane
        self.coordinates = X,Y,Z
        self.prev_ax = main_ax
        return plane

    def __call__(self, pointA, pointB, pointC, count=None):
        self._get_plane_coefs(pointA, pointB, pointC)
        self._get_plane(self.coefs, count=count)
        return self.last_plane

def draw_sphere(arr, point, r=3, color = 255):
    i,j,k = np.indices(arr.shape)
    dist = np.sqrt( (point.x-i)**2 + (point.y-j)**2 + (point.z-k)**2)
    arr[dist < r] = color
    return arr

def get_plane_coefs(pointA, pointB, pointC):
    p1 = pointA.coords()
    p2 = pointB.coords()
    p3 = pointC.coords()

    print(pointA, pointB, pointC, sep=", ", end=", ")

    # These two vectors are in the plane
    v1 = p3 - p1
    v2 = p2 - p1

    # the cross product is a vector normal to the plane
    cp = np.cross(v1, v2)
    a, b, c = cp

    # This evaluates a * x3 + b * y3 + c * z3 which equals d
    d = np.dot(cp, p3)

    return a, b, c, d

def get_plane(arr, coefs, oob_black=True, count=None, prev_ax=None):
    a, b, c, d = coefs
    sx, sy, sz = arr.shape
    main_ax = np.argmax([abs(a), abs(b), abs(c)])

    if main_ax == 0:
        Y, Z = np.meshgrid(np.arange(sy), np.arange(sz), indexing='ij')
        X = (d - b * Y - c * Z) / a

        X = X.round().astype(np.int)
        P = X.copy()
        S = sx-1

        X[X <= 0]  = 0
        X[X >= sx] = sx-1

    elif main_ax==1:
        Z, X = np.meshgrid(np.arange(sz), np.arange(sx), indexing='ij')
        Y = (d - a * X - c * Z) / b

        Y = Y.round().astype(np.int)
        P = Y.copy()
        S = sy-1

        Y[Y <= 0]  = 0
        Y[Y >= sy] = sy-1
    
    elif main_ax==2:
        X, Y = np.meshgrid(np.arange(sx), np.arange(sy), indexing='ij')
        Z = (d - a * X - b * Y) / c

        Z = Z.round().astype(np.int)
        P = Z.copy()
        S = sz-1

        Z[Z <= 0]  = 0
        Z[Z >= sz] = sz-1
    
    plane = arr[X, Y, Z]
    if oob_black == True:
        plane[P < 0] = 0
        plane[P > S] = 0
    
    
    if main_ax==1:
        plane = np.rot90(plane, k=-1)


    if count is not None:
        # print(count," a",a," b",b," c",c," d",d," main_ax",main_ax, X[0,0],X[-1,-1],Y[0,0],Y[-1,-1],Z[0,0],Z[-1,-1])
        print(count," main_ax",main_ax, X[0,0],X[-1,-1],Y[0,0],Y[-1,-1],Z[0,0],Z[-1,-1])

        saveJPG(plane, os.path.join(PATH, ("000"+str(count))[-3:]+".jpg" ))

    return plane #, (X, Y, Z)

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

def sample_dot(limits):
    assert len(limits) == 3
    coords = []
    for ax in limits:
        if type(ax) == type(tuple()):
            v = (np.random.rand()*(max(ax)-min(ax))) + min(ax)
            coords.append(np.round(v, 3))
        else:
            coords.append(ax)
    
    return coords

def sample_planes(ct_arr, ranges, rep = 10, dest="/vol/biomedic3/hjr119/XCAT/samples/planes", spe_name = "samp", processings=""):
    
    mask = get_us_mask(ct_arr[0]) if "mask" in processings else None
    ct_arr = add_salt_peper(ct_arr) if "salt" in processings else ct_arr

    for i in range(rep):
        dotA        = Point(*sample_dot(ranges[0]), ct_arr)
        dotB        = Point(*sample_dot(ranges[1]), ct_arr)
        dotC        = Point(*sample_dot(ranges[2]), ct_arr)
        coefs       = get_plane_coefs(dotA, dotB, dotC)
        plane, _    = get_plane(ct_arr, coefs)
        if "rotate" in processings:
            angle       = np.round(np.random.rand()*10+40)
            plane       = ndimage.rotate(plane, angle, reshape=False, order=0, mode='nearest')
        if "kernel" in processings:
            plane = apply_kernel(plane)
        if "intensity" in processings:
            plane = rand_intensity(plane)
        if "mask" in processings:
            plane       = plane*mask
        if "patch" in processings:
            plane       = hide_patch(plane)
        name        = spe_name+ ("0000"+str(i))[-4:]  + ".jpg"
        print(name, dotA, dotB, dotC,)
        saveJPG(plane, os.path.join(dest,name))

def intensity_scaling(ndarr, pmin=None, pmax=None, nmin=None, nmax=None):
    pmin = pmin if pmin != None else ndarr.min()
    pmax = pmax if pmax != None else ndarr.max()
    nmin = nmin if nmin != None else pmin
    nmax = nmax if nmax != None else pmax
    
    ndarr[ndarr<pmin] = pmin
    ndarr[ndarr>pmax] = pmax
    ndarr = (ndarr-pmin)/(pmax-pmin)
    ndarr = ndarr*(nmax-nmin)+nmin
    return ndarr    

def sample_from_volumes(root = "/vol/biomedic3/hjr119/XCAT/generation", dest="/vol/biomedic3/hjr119/XCAT/samples/planes2", n=300, mode="CT", processings=""):
    samps = os.listdir(root)
    samps.sort()
    samps_clean = [samp for samp in samps if "samp" in samp and mode in samp and ".nii.gz" in samp]

    if not os.path.exists(dest):
        os.makedirs(dest, exist_ok=True)

    # For old xcat params
    # rangeA = ((0.85, 1) , 0, (0.7,0.92))    #DONE
    # rangeB = ((0.3,0.43), 1, 0) 
    # rangeC = ((0.3,0.43), 1, 1)     

    # For new xcat params
    # rangeA = ((.85, .95) , 0, (0.7, 0.8))
    # rangeB = ((.45, .6)  , 1, 0)
    # rangeC = ((.45, .55) , 1, 1)

    # For rot param
    rangeA = ((0.85,1.0) , 0, (0.7, 1.0))    #DONE
    rangeB = ((0.45,0.70), 1, 0) 
    rangeC = ((0.45,0.70), 1, 1)   

    for samp in samps_clean:
        print(samp)
        path = os.path.join(root, samp)
        
        if "CT" in mode:
            ct_arr = sitk.GetArrayFromImage(sitk.ReadImage(path))
            ct_arr = ct_arr/ct_arr.max()*255
            # data = intensity_scaling(ct_arr, pmin=150, pmax=200, nmin=0, nmax=255)
            data = intensity_scaling(ct_arr, pmin=150, pmax=200, nmin=0, nmax=220)+35

        elif "SEG" in mode:
            data = load_clean_seg(path)

        sample_planes(data, (rangeA, rangeB, rangeC), rep=n//len(samps_clean), dest=dest, spe_name=samp[:6], processings=processings)

def add_salt_peper(arr):
    noise = np.random.rand(*arr.shape)
    proba = 0.1
    disrupt = 1.5
    arr[noise < proba] = arr[noise < proba]/disrupt
    arr[noise > 1-proba] = arr[noise > 1-proba]*disrupt
    arr[arr>255] = 255
    return arr

def hide_patch(img, mins=0.1, maxs=.4, p=0.5, patch_value=0):
    if p > np.random.rand():
        sx, sy = img.shape
        size = int((np.random.rand() * maxs-mins + mins)*sx)-1
        px   = int(np.random.rand() * (sx-size))
        py   = int(np.random.rand() * (sy-size))
        img[px:(px+size),py:(py+size)] = patch_value
    
    return img

def get_us_mask(arr):
    assert arr.ndim == 2

    sx, sy = arr.shape # H, W

    mask = np.ones_like(arr)*1

    # Bottom circular shape
    rr, cc = circle(0, int(sy//2), sy-1)
    rr[rr >= sy] = sy-1
    cc[cc >= sy] = sy-1
    rr[rr < 0] = 0
    cc[cc < 0] = 0
    mask[rr, cc] = 0
    # Left triangle
    r = (0, 0, sx*5/9)
    c = (0, sy//2, 0)
    rr, cc = polygon(r, c)
    mask[rr, cc] = 1
    # Right triangle
    r = (0,      0, sx*5/9)
    c = (sy//2, sy-1, sy-1)
    rr, cc = polygon(r, c)
    mask[rr, cc] = 1

    mask = (1-mask).astype(np.bool)

    return mask
        
def apply_kernel(img, p=0.5):
    if p > np.random.rand():
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]]) * 1/10
        return signal.convolve2d(img, kernel, mode='same', boundary='fill', fillvalue=0)
    else:
        return img

def rand_intensity(img):
    thresh = np.random.rand()*255
    img = img+thresh
    img[img > 255] = img[img > 255]-255
    return img

def full_rot(arr, axis=0):

    sampler = PlaneSampler(arr)
    
    dotA = Point(*np.roll((0, .5, .5), shift=axis), arr)
    dotB = Point(*np.roll((1, .5, .5), shift=axis), arr)
    step = 0.1
    planes = []
    c=0
    for i in np.arange(0,1,step):
        c+=1
        
        dotC = Point(*np.roll((.5, i, 0), shift=axis), arr)
        planes.append(sampler(dotA, dotB, dotC, count=c))
        # planes.append(get_plane(arr, get_plane_coefs(dotA, dotB, dotC),count=c))
    for i in np.arange(0,1,step):
        c+=1
        dotC = Point(*np.roll((.5, 1, i), shift=axis), arr)
        planes.append(sampler(dotA, dotB, dotC, count=c))
        #planes.append(get_plane(arr, get_plane_coefs(dotA, dotB, dotC),count=c))
    for i in np.arange(0,1,step):
        c+=1
        dotC = Point(*np.roll((.5, 1-i, 1), shift=axis), arr)
        planes.append(sampler(dotA, dotB, dotC, count=c))
        #planes.append(get_plane(arr, get_plane_coefs(dotA, dotB, dotC),count=c))
    for i in np.arange(0,1,step):
        c+=1
        dotC = Point(*np.roll((.5, 0, 1-i), shift=axis), arr)
        planes.append(sampler(dotA, dotB, dotC, count=c))
        #planes.append(get_plane(arr, get_plane_coefs(dotA, dotB, dotC),count=c))
    
    return np.array(planes)

def arr2video(arr, foutput, fps = 10):
    size = (arr.shape[-1],arr.shape[-1])
    out = cv2.VideoWriter(foutput, cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)
    for i in range(arr.shape[0]):
        data = arr[i].astype(np.uint8)
        out.write(data)
    out.release()

if __name__ == "__main__":


    # sample_from_volumes(dest="/vol/biomedic3/hjr119/XCAT/samples/planes2", n=20, mode="CT", processings="salt,,patch,mask,intensity") # mask, salt, patch
    # exit()

    ct_itk = sitk.ReadImage("/vol/biomedic3/hjr119/XCAT/generation/samp0_1_CT.nii.gz")
    ct_arr = sitk.GetArrayFromImage(ct_itk)
    ct_arr = ct_arr/ct_arr.max()*255
    ct_arr = intensity_scaling(ct_arr, pmin=150, pmax=200, nmin=0, nmax=255)
    
    # planes = full_rot(ct_arr, axis=0)
    # arr2video(planes, foutput="/vol/biomedic3/hjr119/XCAT/rot0.mp4", fps=5)

    # planes = full_rot(ct_arr, axis=1)
    # arr2video(planes, foutput="/vol/biomedic3/hjr119/XCAT/rot1.mp4", fps=5)

    planes = full_rot(ct_arr, axis=2)
    arr2video(planes, foutput="/vol/biomedic3/hjr119/XCAT/rot2.mp4", fps=5)

    
    # ct_itk = sitk.ReadImage("/vol/biomedic3/hjr119/XCAT/generation/default_512_CT_1.nii.gz")
    # ct_arr = sitk.GetArrayFromImage(ct_itk)
    # # ct_arr = np.moveaxis(ct_arr, 0, -1)
    # sx, sy, sz = ct_arr.shape
    # ct_arr = ct_arr/ct_arr.max()*255
    # ct_arr = intensity_scaling(ct_arr, pmin=150, pmax=200, nmin=0, nmax=255)
    # print(ct_arr.min(), ct_arr.max(), ct_arr.shape)


    # # dotA = Point(0, 0, .5, ct_arr)
    # # dotB = Point(0, .5, 0, ct_arr)
    # # dotC = Point(.5, 0, 0, ct_arr)

    # # Parameters for 4ch sampling
    # # axial, coronal, sagittal
    # rangeA = ((0.85, 1) , 0, (0.7,0.92))    #DONE
    # rangeB = ((0.3,0.43), 1, 0) 
    # rangeC = ((0.3,0.43), 1, 1)   

    # # rangeA = ((0, 1), 0, (0, 1))
    # # rangeB = ((0, 1), 1, (0, 1)) 
    # # rangeC = ((0, 1), 1, (0, 1))   

    # # previ = time.time()

    # # sample_planes(ct_arr, (rangeA, rangeB, rangeC), rep=30)

    # # after = time.time()

    # # print("Generated in", after-previ, "sec.")

    # # exit()

    # # dotA = Point(*sample_dot(rangeA), ct_arr)
    # # dotB = Point(*sample_dot(rangeB), ct_arr)
    # # dotC = Point(*sample_dot(rangeC), ct_arr)

    # dotA = Point(1.1, 0, 1.1, ct_arr)
    # dotB = Point(0.3, 1.1, 0, ct_arr)
    # dotC = Point(0.3, 1.1, 1.1, ct_arr)

    # print(dotA, dotB, dotC, sep='\n')

    # # if False:
    # #     r = 5
    # #     ct_arr = draw_sphere(ct_arr, dotA, r=r)
    # #     ct_arr = draw_sphere(ct_arr, dotB, r=r)
    # #     ct_arr = draw_sphere(ct_arr, dotC, r=r)

    # coefs = get_plane_coefs(dotA, dotB, dotC)
    # # # print(coefs)

    # plane, (X, Y, Z) = get_plane(ct_arr, coefs)
    # # # print(coefs, plane.shape)
    # # # print(plane.min(), plane.max())

    # # angle = np.round(np.random.rand()*10+40)
    # # # print("Angle", angle)
    # # plane = ndimage.rotate(plane, angle, reshape=False)

    # # # ct_arr[X, Y, Z] = 255 # Draws white lines on the images

    # # # name = "plane_"+str(hA)[:3]+"_"+str(hB)[:3]+"_"+str(hC)[:3]+".jpg"
    # name = "tmp.jpg"
    # saveJPG(plane, os.path.join("/vol/biomedic3/hjr119/XCAT/samples","outside.jpg"))

    # saveJPG(ct_arr[sx//2, :, :], "/vol/biomedic3/hjr119/XCAT/samples/ax1.jpg")
    # saveJPG(ct_arr[:, sy//2, :], "/vol/biomedic3/hjr119/XCAT/samples/ax2.jpg")
    # saveJPG(ct_arr[:, :, sz//2], "/vol/biomedic3/hjr119/XCAT/samples/ax3.jpg")

    # # save_mpl(dotA, dotB, dotC, X, Y, Z)

    # print("done")


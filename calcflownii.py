#!/usr/bin/env python3
# calcflow.py:  calculate flow in a blood vessel, using nifti images

# NOTE:  must run this using pythonw for the key_press_event to show up

import os
import morphsnakes
import dicom
import numpy as np
import nibabel as nib
from statistics import mean
from matplotlib import pyplot as ppl
from skimage.morphology import convex_hull_image, binary_dilation
import json
from pylab import rcParams
import subprocess

rcParams['figure.figsize'] = 8, 8

def circle_levelset(shape, center, sqradius):
    """Build a binary function with a circle as the 0.5-levelset."""
    grid = np.mgrid[list(map(slice, shape))].T - center
    phi = sqradius - np.sqrt(np.sum(grid.T**2, 0))
    u = np.float_(phi > 0)
    return u


# read the nifti images

# imgpath='/bme006/hamilton/Projects/2017-Alzheimers-Synergy/TestScans//ALZHEART_TEST_121516_HUMAN_PHANTOM/20161215/'
# magimgs= nib.load(imgpath+'MR0017/MR0017_Brain_2D_PhCon_20161215085506_17.nii')
# pcimgs = nib.load(imgpath+'MR0018/MR0018_Brain_2D_PhCon_20161215085506_18.nii')

imgpath='/bme006/hamilton/Projects/2017-Alzheimers-Synergy/TestScans//ALZHEART_TEST_121516_HUMAN_PHANTOM/output/'
magimgs= nib.load(imgpath+'Brain2DPhCon_rs17.nii.gz')
pcimgs = nib.load(imgpath+'Brain2DPhCon_rs18.nii.gz')

magdata = magimgs.get_data()
pcdata  = pcimgs.get_data()

shp = pcdata.shape

nrows = shp[0]
ncols = shp[1]
nfrms = shp[3]

# get the venc from the info file
tmpstr = subprocess.check_output(['grep', '-i', 'nvel', imgpath+'Brain2DPhCon_rs18.info'])

vencstr = tmpstr.split()[-1]   #  venc is last

print('vencstr = ', vencstr, ' cm/s')

venc = float(vencstr)*10   # convert to mm/s

print('venc = ', venc, ' mm/s')

# setup nparray as 4D
imgs = np.zeros((nrows, ncols, nfrms, 2))

imgs[:,:,:,0] = magdata[:,:,0,:]
imgs[:,:,:,1] = pcdata[:,:,0,:]

# now rescale using the venc
imgs[:, :, :, 1] = (imgs[:, :, :, 1] - 2048.0) * (venc / 2048.0)

print('mag imgs.min =', imgs[:, :, :, 0].min())
print('mag imgs.max =', imgs[:, :, :, 0].max())

print('pc imgs.min =', imgs[:, :, :, 1].min())
print('pc imgs.max =', imgs[:, :, :, 1].max())

print('\n')

# ~~~~~~~~~~~~~~~~~  Now use morphsnakes to delineate the lumen on the first mag image ~~~~~~~~~~~~~~~~~

crop = 0
if crop > 0:
    myimg = imgs[int((nrows-crop)/2):int((nrows+crop)/2), int((ncols-crop)/2):int((ncols+crop)/2), 0, 0]
else:
    myimg = imgs[:,:,0,0]

# assume the vessel is at the center of the FOV
# cx = int(ncols/2)
# cy = int(nrows/2)
# cx = cy = int(crop/2)

# have the user click on the center of each vessel of interest
coords = []
print('Click on center of vessel(s), press a key to end')

def onclick(event):
    global ix, iy
    ix, iy = event.ydata, event.xdata   # note transposition due to imshow being transposed

    print('event.name= ',event.name)

    print('x = %d, y = %d' % ( ix, iy))

    # assign global variable to access outside of function
    global coords
    coords.append((iy, ix))

    return

def onpress(event):

    print('event.name= ',event.name)
    fig.canvas.mpl_disconnect(cid1)
    fig.canvas.mpl_disconnect(cid2)
    ppl.close(33)

    return


fig = ppl.figure(num=33,figsize=(16,16))
fig.clf()
ppl.imshow(myimg.transpose(1,0), cmap=ppl.cm.gray, interpolation='bicubic', origin = 'lower')

cid1 = fig.canvas.mpl_connect('button_press_event', onclick)
cid2 = fig.canvas.mpl_connect('key_press_event', onpress)

ppl.show(33)

# waiting here until user finishes selecting vessels

num_vessels = len(coords)
vel = np.zeros((num_vessels,nfrms))

print('coords = ',coords)

fig = ppl.figure(num=32,figsize=(16,16))
fig.clf()
ax1 = ppl.subplot(2, 2, 1)
ax1.set_xticks([])
ax1.set_yticks([])
ppl.imshow(myimg, cmap=ppl.cm.gray, interpolation='bicubic')

masksum = np.zeros((nrows,ncols))

for ptnum,pt in enumerate(coords):
    # Morphological ACWE. Initialization of the level-set.
    macwe = morphsnakes.MorphACWE(myimg, smoothing=1, lambda1=1, lambda2=2)
    macwe.levelset = circle_levelset(myimg.shape, pt, 2)

    # Visual evolution.
    numiter = 10
    print('Snaking at (%d,%d)...'% (pt))

    # finalset = morphsnakes.evolve_visual(macwe, num_iters=numiter, background=myimg)
    ax1.contour(macwe.levelset, [0.5], colors='r')

    ax2 = ppl.subplot(2, 2, 2)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2b = ppl.imshow(macwe.levelset)


    # Iterate.
    for i in range(numiter):
        # Evolve.
        macwe.step()
        # print('on interation %d' % (i))
        # Update figure.
        # del ax1.collections[0]
        ax1.contour(macwe.levelset, [0.5], colors='g')
        ax2b.set_data(macwe.levelset)
        fig.canvas.draw()
        ppl.pause(0.05)


    # ~~~~~~~~~~~~~  Use the convex hull of the 'finalset' to extract values from the velocity images

    mask = convex_hull_image(macwe.levelset)

    # dilate the mask a bit
    mask2 = binary_dilation(mask)

    masksum = masksum + mask2

    ax2b.set_data(masksum)
    fig.canvas.draw()


    for frm in range(nfrms):
        # tmpimg = imgs[:,:,frm,1]
        if crop > 0:
            tmpimg = \
                imgs[int((nrows - crop) / 2):int((nrows + crop) / 2), 
                     int((ncols - crop) / 2):int((ncols + crop) / 2),
                     frm,
                     1]
        else:
            tmpimg = imgs[:,:,frm,1]

        vel[ptnum,frm] = mean(tmpimg[mask2 > 0])

    # print('mean vel in frame %d = %5.1f mm/sec' % (frm, vel[frm]))

    if np.sum(vel[ptnum,:]) < 0:
        vel[ptnum,:] *= -1

    print('vessel %d mean velocity over heart cycle: %5.1f' % (ptnum,mean(vel[ptnum,:])))

    ppl.subplot(2, 1, 2)

    ppl.plot(list(range(nfrms)), vel[ptnum,:])
    ppl.xlabel('Frame')
    ppl.ylabel('Mean luminal velocity [mm/s]')
    ppl.title('Velocity in vessel')
    fig.canvas.draw()


print('Close the figure to continue...')
ppl.show(32)


# save the mask as a Nifti image
pchdr = pcimgs.header
pixel_size = pchdr.get_zooms()
print('get_zooms(): ',pixel_size)

nibimg = nib.Nifti1Image(np.ushort(masksum), None, pchdr)
hdr = nibimg.header
hdr.set_xyzt_units('mm', 'sec')
# print(hdr)
nib.save(nibimg, os.path.join(imgpath,'mask2d_rs18.nii.gz'))

# here is how to save it as a 3D with it replicated nfrms times
mask3d = np.zeros((nrows,ncols,1,nfrms))
for i in range(nfrms):
    mask3d[:,:,0,i] = masksum


print('pchdr.shape = ',pcdata.shape)
print('mask2.shape = ',mask2.shape)
print('masksum.shape = ',masksum.shape)
print('mask3d.shape = ',mask3d.shape)

nibmask3d = nib.Nifti1Image(np.ushort(mask3d), None, pchdr)
hdr3d = nibmask3d.header

hdr3d.set_xyzt_units('mm', 'sec')
# print(hdr3d)
nib.save(nibmask3d, os.path.join(imgpath, 'mask3d_rs18.nii.gz'))

# save the velocity info to JSON as a dict containing a list for each vessel, use a list comprehension to format it
jdict = {}

outfilename = os.path.join(imgpath, 'velocity_rs18.json')
with open(outfilename, 'wt') as outfile:
    for vnum in range(num_vessels):
        jdict[vnum] = ["%.2f" % member for member in list(vel[vnum,:])]

    json.dump(jdict, outfile, indent=4, separators=(',', ':'))


# now load the saved masksum and see how it looks

maskn = nib.load(imgpath+'mask3d_rs18.nii.gz')

maskndata = maskn.get_data()

fig = ppl.figure(num=31,figsize=(16,16))
fig.clf()
ppl.imshow(maskndata[:,:,0,0], cmap=ppl.cm.gray, interpolation='bicubic')
ppl.show(31)

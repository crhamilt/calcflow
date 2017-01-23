#!/usr/bin/env python3
# calcflow.py:  calculate flow in a blood vessel

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


def circle_levelset(shape, center, sqradius):
    """Build a binary function with a circle as the 0.5-levelset."""
    grid = np.mgrid[list(map(slice, shape))].T - center
    phi = sqradius - np.sqrt(np.sum(grid.T**2, 0))
    u = np.float_(phi > 0)
    return u

imgpath='/bme006/hamilton/Projects/2017-Alzheimers-Synergy/TestScans//ALZHEART_TEST_121516_HUMAN_PHANTOM/20161215/'

# read velocity images first, since they have venc in them
pcdir = imgpath+'MR0018/'
dspc = dicom.read_file(pcdir+'000001.DCM')
print("reading first pc image ", pcdir + '000001.DCM')
pxmag = dspc.pixel_array
print('dspc.pixel_array.shape =', dspc.pixel_array.shape)
nrows, ncols = dspc.pixel_array.shape
print('dspc.CardiacNumberOfImages =', dspc.CardiacNumberOfImages)
nfrms = int(dspc.CardiacNumberOfImages)
pixel_size = dspc.PixelSpacing
slice_thickness = dspc.SliceThickness

# get the venc
vencstr0 = dspc[0x51, 0x1014].value
# Siemens stores the venc as "vNNN_through"
uspos = vencstr0.find("_")
vencstr1 = vencstr0[1:uspos]
print('vencstr1 = ', vencstr1, ' cm/s')
venc = float(vencstr1) * 10  # convert cm/s to mm/s
print('venc = ', venc, ' mm/s')

# setup nparray as 4D
imgs = np.zeros((nrows, ncols, nfrms, 2))


magdir = imgpath+'MR0017/'

print('Loading images...\n')

for imnum in range(1, nfrms + 1):
    if imnum < 10:
        fnpc = "%s/00000%1d.DCM" % (pcdir, imnum)
        fnmag = "%s/00000%1d.DCM" % (magdir, imnum)
    elif imnum < 100:
        fnpc = "%s/0000%2d.DCM" % (pcdir, imnum)
        fnmag = "%s/0000%2d.DCM" % (magdir, imnum)
    else:
        fnpc = "%s/000%3d.DCM" % (pcdir, imnum)
        fnmag = "%s/000%3d.DCM" % (magdir, imnum)

    # print("reading ", fnpc)
    ds = dicom.read_file(fnpc)
    imgs[:, :, imnum - 1, 1] = ds.pixel_array
    # print("reading ", fnmag)
    ds = dicom.read_file(fnmag)
    imgs[:, :, imnum - 1, 0] = ds.pixel_array


# now rescale using the venc
imgs[:, :, :, 1] = (imgs[:, :, :, 1] - 2048.0) * (venc / 2048.0)

print('mag imgs.min =', imgs[:, :, :, 0].min())
print('mag imgs.max =', imgs[:, :, :, 0].max())

print('pc imgs.min =', imgs[:, :, :, 1].min())
print('pc imgs.max =', imgs[:, :, :, 1].max())

print('\n')
print('before transposing imgs.shape:',imgs.shape)

imgs = np.swapaxes(imgs,0,2)
imgs = np.swapaxes(imgs,1,2)  # have to do it in 2 steps?

print('after transposing imgs.shape:',imgs.shape)
print('\n')

nibimg4d = nib.Nifti1Image(np.ushort(imgs), np.diag([slice_thickness, pixel_size[0], pixel_size[1], 1.0]))
hdr4d = nibimg4d.header
hdr4d.set_xyzt_units('mm', 'sec')
hdr4d.__setitem__('toffset', venc)       # store the VENC in toffset

print(hdr4d)
nib.save(nibimg4d, os.path.join(pcdir, 'img4d.nii.gz'))

# ~~~~~~~~~~~~~~~~~  Now use morphsnakes to delineate the lumen on the first mag image ~~~~~~~~~~~~~~~~~

# myimg = imgs[0, :, :, 0]
crop = 40
myimg = imgs[0, int((nrows-crop)/2):int((nrows+crop)/2), int((ncols-crop)/2):int((ncols+crop)/2), 0]

# assume the vessel is at the center of the FOV
# cx = int(ncols/2)
# cy = int(nrows/2)
# cx = cy = int(crop/2)

# have the user click on the center of each vessel of interest
coords = []
print('Click on center of vessel(s), press a key to end')

def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata

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


fig = ppl.figure(33)
fig.clf()
ppl.imshow(myimg, cmap=ppl.cm.gray, interpolation='Nearest')

cid1 = fig.canvas.mpl_connect('button_press_event', onclick)
cid2 = fig.canvas.mpl_connect('key_press_event', onpress)

ppl.show(33)

# waiting here until user finishes selecting vessels

num_vessels = len(coords)
vel = np.zeros((num_vessels,nfrms))

print('coords = ',coords)

fig = ppl.figure(32)
fig.clf()
ax1 = ppl.subplot(2, 2, 1)
ax1.set_xticks([])
ax1.set_yticks([])
ppl.imshow(myimg, cmap=ppl.cm.gray, interpolation='Nearest')

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

    ax2b.set_data(mask2)
    fig.canvas.draw()


    for frm in range(nfrms):
        # tmpimg = imgs[frm,:,:,1]
        tmpimg = \
            imgs[frm, 
                 int((nrows - crop) / 2):int((nrows + crop) / 2), 
                 int((ncols - crop) / 2):int((ncols + crop) / 2), 
                 1]

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
ppl.show()


# save the mask as a Nifti image
nibimg = nib.Nifti1Image(np.ushort(mask2), np.diag([pixel_size[0], pixel_size[1], slice_thickness, 1.0]))
hdr = nibimg.header
hdr.set_xyzt_units('mm', 'sec')
print(hdr)
nib.save(nibimg, 'mask.nii.gz')

# here is how to save it as a 3D with it replicated nfrms times
mask3d = [mask2]*nfrms

nibmask3d = nib.Nifti1Image(np.ushort(mask3d), np.diag([slice_thickness, pixel_size[0], pixel_size[1], 1.0]))
hdr3d = nibmask3d.header

hdr3d.set_xyzt_units('mm', 'sec')
print(hdr3d)
nib.save(nibmask3d, os.path.join(pcdir, 'mask3d.nii.gz'))

# save the velocity info to JSON as a dict containing a list for each vessel, use a list comprehension to format it
jdict = {}

outfilename = os.path.join(pcdir, 'velocity.json')
with open(outfilename, 'wt') as outfile:
    for vnum in range(num_vessels):
        jdict[vnum] = ["%.2f" % member for member in list(vel[vnum,:])]

    json.dump(jdict, outfile, indent=4, separators=(',', ':'))

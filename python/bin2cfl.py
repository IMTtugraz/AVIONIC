#!/usr/bin/env python

import os
import ismrmrd
import ismrmrd.xsd
import numpy as np
import scipy.io as sio
from ismrmrdtools import show, transform
import argparse
import struct
import time


parser = argparse.ArgumentParser()

parser.add_argument("-f", dest="filename", required=True,
                    help="h5 input file", metavar="FILE")

parser.add_argument("-d", dest="dimensions",nargs=3, required=True,
                    help="image dimensions (n,m,nframes)")
#
# output
parser.add_argument("-o", dest="output", required=False,
                    help="bin/mat/cfl output file", default="out")
parser.add_argument("-c", dest="writeCFL", action='store_true', default=False,
                    help="write cfl file")
parser.add_argument("-m", dest="writeMAT", action='store_true', default=False,
                    help="write mat file")


args = parser.parse_args()

# Load file
if not os.path.isfile(args.filename):
    print("%s is not a valid file" % filename)
    raise SystemExit

print("processing ",args.filename)
if args.writeMAT:
  sio.savemat(outname, {'kspace':all_data})


# Read and reshape binary file 
#================================================================================
f=open(args.filename,"rb")
filecontent=f.read()
f.close()
ics=struct.unpack('b',filecontent[1])
flength=struct.unpack('D',filecontent[2:5])
img=struct.unpack(flength[0]*"f",filecontent[6:])


# Save to cfl file --> bart
#================================================================================
#TODO manage dimensions
if args.writeCFL:
  datasqueeze = all_data.transpose(9,8,7,6,0,1,2,3,4,5);
  datasqueeze = datasqueeze.squeeze()
  print "datasqueeze.shape: ",datasqueeze.shape 
  h = open(outname + ".hdr", "w")
  h.write('# Dimensions\n')
  for i in (datasqueeze.shape):
    h.write("%d " % i)
  h.write('\n')
  h.close()
  d = open(outname + ".cfl", "w")
  datasqueeze.T.astype(np.complex64).tofile(d) # tranpose for column-major order
  d.close()
  del datasqueeze

#================================================================================
# write data and mask to binary file
#================================================================================
if args.writeBIN:
  for slices in range(nslices):
    print "writing slice ", slices, "to binary file"
    #TODO consider not only slices
    datawrite = all_data[0,0,:,0,0,slices,:,0,:,:];
    datawrite = datawrite.squeeze();
    print "datawrite shape", datawrite.shape
    print "n: ",all_data.shape[9]
    n = all_data.shape[9];
    print "m: ",all_data.shape[8]
    m = all_data.shape[8];
    print "coils: ", ncoils
    print "frames: ", nphases
  
    # reshape data
    datawrite = datawrite.transpose(2,3,1,0);
 
    # define mask corresponding to non-zero entries of data
    maskdata = datawrite[:,:,0,:];
    maskdata = maskdata.squeeze();
    mask = maskdata != 0;
    mask = np.complex64(mask)
    #mask = np.ones((m,n,nphases));
    print "datawrite shape", datawrite.shape, "mask shape", mask.shape

    # setup data
    print "setting up data"

    #Get chop variable
    x = np.linspace(0, n-1,n)
    y = np.linspace(0, m-1,m )
    xx, yy = np.meshgrid(x, y)
    chop = (-1)**(xx+yy)  
  
    #Shift mask and data for matlab fft , recenter image by chop
    for j in range(nphases):
      mask[:,:,j] = transform.ifftshift(mask[:,:,j] );	
      for k in range(ncoils):
        datawrite[:,:,k,j] = transform.ifftshift(np.multiply(chop,datawrite[:,:,k,j]))

    #Rescale data
    datawrite = datawrite*np.sqrt(n*m)

    #Normalize Data
    msum = np.sum(mask,axis=2)
    crec = np.zeros((m,n,ncoils),dtype=np.complex64)
    for j in range(ncoils):
      datatmp=np.sum(np.squeeze(datawrite[:,:,j,:]),axis=2)
      datatmp[msum>0]=datatmp[msum>0]/msum[msum>0]
      crec[:,:,j]=transform.fftn(datatmp)/np.sqrt(n*m)

    u = np.sqrt(np.sum(np.abs(crec)**2,axis=2))
    u = u.flatten()
    datanorm=255/np.median(u[u>0.9*u.max()])
    print "datanorm: ", datanorm
    datawrite = datawrite*datanorm

    # Put in one column for writebin
    datawrite = datawrite.transpose(1,0,2,3)
    mask = mask.transpose(1,0,2)
    datawrite = datawrite.flatten(1);  
    mask = mask.flatten(1);
    mlength = mask.shape[0];
    dlength = datawrite.shape[0];
    isc = 1;
  
    outnamemask = './' + args.output + '_mask' + '_sl' + str(slices) + '.bin'
    outname = './' + args.output + '_sl' + str(slices) + '.bin'
  
    #Save to matlab
    #sio.savemat('datawrite', {'datawrite':datawrite})
    #sio.savemat('mask', {'mask':mask})

    try:
      os.remove(outname)
    except OSError:
      pass
    try:
      os.remove(outnamemask)
    except OSError:
      pass
  
    t0 = time.time()
    print "writing data to binary" 
    maskreal = mask.real
    maskimag = mask.imag
    maskwrite = np.concatenate((maskreal,maskimag))
    maskwrite = np.float64(maskwrite)
    mlength=np.uint32(len(maskwrite)/2)
    f=open(outnamemask,"wb")
    f.write(struct.pack('B',1))
    f.write(struct.pack('i',mlength))
    f.write(struct.pack('d'*len(maskwrite),*maskwrite))
    f.close()

    datareal = datawrite.real
    dataimag = datawrite.imag
    datawrite = np.concatenate((datareal,dataimag))
    datawrite = np.float64(datawrite)
    dlength=np.uint32(len(datawrite)/2)
    isc=1
    print "datawrite shape: ", datawrite.shape, "dlength", dlength

    f = open(outname,"wb")
    f.write(struct.pack('B',1))
    f.write(struct.pack('i',dlength))
    f.write(struct.pack('d'*len(datawrite),*datawrite))
    f.close()
    print "duration: ", (time.time() - t0)

#======================================================================================================
# SOS Reconstruction
#======================================================================================================
if args.reconSOS:
  print "Reconstruction SOS"
  #TODO Normalize
  # all_data = all_data*datanorm
  # all_data=all_data/np.sqrt(n*m)

  # Reconstruct images
  if args.removeOS:
    images = np.zeros((nsets, navg, nphases, nreps, ncontrasts, nslices, rNz, rNy, rNx), dtype=np.float32)
  else:
    images = np.zeros((nsets, navg, nphases, nreps, ncontrasts, nslices, rNz, rNy, 2*rNx), dtype=np.float32)

  for set_ in range(nsets):
    for avg in range(navg):
      for phase in range(nphases):
        for rep in range(nreps):
          for contrast in range(ncontrasts):
            for slice in range(nslices):
              # FFT
              if eNz>1:
                #3D
                if args.writeBIN:
		  im = transform.fftn(np.squeeze(all_data[set_, avg, phase, rep, contrast, slice, :, :, :, :]))
                else:
              	  im = transform.transform_kspace_to_image(all_data[set_, avg, phase, rep, contrast, slice, :, :, :, :], [1,2,3])
        
	      else:
                #2D
                if args.writeBIN:
		  im = transform.fftn(np.squeeze(all_data[set_, avg, phase, rep, contrast, slice, :, 0, :, :]))
 		else:
                  im = transform.transform_kspace_to_image(all_data[set_, avg, phase, rep, contrast, slice, :, 0, :, :], [1,2])
              
	      # Sum of squares
              im = np.sqrt(np.sum(np.abs(im) ** 2, 0))
            
              # Stuff into the output
              if eNz>1:
                #3D
                images[set_, avg, phase, rep, contrast, slice, :, :, :] = im
              else:
                #2D
                images[set_, avg, phase, rep, contrast, slice, 0, :, :] = im

  outname = './' + args.output + 'sos'
  sio.savemat(outname, {'images':images})
  
# Show an image
#  show.imshow(np.squeeze(images[0,0,0,0,0,:,:,:]))


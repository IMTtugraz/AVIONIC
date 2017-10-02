#!/usr/bin/env python

import os
import ismrmrd
import ismrmrd.xsd
import numpy as np
import scipy.io as sio
from ismrmrdtools import show, transform, coils, grappa, sense
import argparse
import struct
import time
import hdf5storage


#================================================================================
# Define parsing arguments
#================================================================================
parser = argparse.ArgumentParser()

parser.add_argument("-f", dest="filename", required=True,
                    help="h5 input file", metavar="FILE")
parser.add_argument("-n", dest="noisename", required=False,
                    help="h5 noise input file", metavar="FILE")

parser.add_argument("-p", dest="removeOS", action='store_true', default=False,
                    help="remove over-sampling")
parser.add_argument("-s", dest="reconSOS", action='store_true', default=False,
                    help="reconstruct sos")
parser.add_argument("-v", dest="verbose", action='store_true', default=False,
                    help="verbose output")
# output
parser.add_argument("-o", dest="output", required=False,
                    help="bin/mat/cfl output file", default="out")
parser.add_argument("-l", dest="writeCFL", action='store_true', default=False,
                    help="write cfl file")
parser.add_argument("-m", dest="writeMAT", action='store_true', default=False,
                    help="write mat file")
parser.add_argument("-b", dest="writeBIN", action='store_true', default=False,
                    help="write binary files for each slice for ICTGV Reconstruction")

parser.add_argument("-c", dest="calcCOILS", action='store_true', default=False,
                    help="write walsh estimated coil senstivities (matlab)")



args = parser.parse_args()

# Load file
if not os.path.isfile(args.filename):
    print("%s is not a valid file" % filename)
    raise SystemExit

print("processing ",args.filename)
dset = ismrmrd.Dataset(args.filename, 'dataset', create_if_needed=False)
header = ismrmrd.xsd.CreateFromDocument(dset.read_xml_header())
enc = header.encoding[0]
#================================================================================
#
#
#================================================================================
# Process the noise data
#================================================================================
if args.noisename in locals():
   if os.path.isfile(args.noisename) :
      print "loading noise data" 
      noise_reps = noise_dset.number_of_acquisitions()
      a = noise_dset.read_acquisition(0)
      noise_samples = a.number_of_samples
      num_coils = a.active_channels
      noise_dwell_time = a.sample_time_us

      noise = np.zeros((num_coils,noise_reps*noise_samples),dtype=np.complex64)
      for acqnum in range(noise_reps):
        acq = noise_dset.read_acquisition(acqnum)
    
        # TODO: Currently ignoring noise scans
        if not acq.isFlagSet(ismrmrd.ACQ_IS_NOISE_MEASUREMENT):
          raise Exception("Errror: non noise scan found in noise calibration")

        noise[:,acqnum*noise_samples:acqnum*noise_samples+noise_samples] = acq.data
    
      noise = noise.astype('complex64')
   
   #Calculate prewhiterner taking BWs into consideration
   a = dset.read_acquisition(firstacq)
   data_dwell_time = a.sample_time_us
   noise_receiver_bw_ratio = 0.79
   dmtx = coils.calculate_prewhitening(noise,scale_factor=(data_dwell_time/noise_dwell_time)*noise_receiver_bw_ratio)
#================================================================================
#
#
#================================================================================
# assemble information
#================================================================================
print "trajectory: ",enc.trajectory

# Matrix size
eNx = enc.encodedSpace.matrixSize.x
eNy = enc.encodedSpace.matrixSize.y
eNz = enc.encodedSpace.matrixSize.z
rNx = enc.reconSpace.matrixSize.x
rNy = enc.reconSpace.matrixSize.y
rNz = enc.reconSpace.matrixSize.z
print "eNx = ",eNx," / eNy = ",eNy," / eNz = ",eNz," / rNx = ",rNx," / rNy = ",rNy," / rNz = ",rNz
# Field of View
eFOVx = enc.encodedSpace.fieldOfView_mm.x
eFOVy = enc.encodedSpace.fieldOfView_mm.y
eFOVz = enc.encodedSpace.fieldOfView_mm.z
rFOVx = enc.reconSpace.fieldOfView_mm.x
rFOVy = enc.reconSpace.fieldOfView_mm.y
rFOVz = enc.reconSpace.fieldOfView_mm.z
print "eFOVx = ",eFOVx," / eFOVy = ",eFOVy," / eFOVz = ",eFOVz," / rFOVx = ",rFOVx," / rFOVy = ",rFOVy," / rFOVz = ",rFOVz

# Number of Slices, Reps, Contrasts, Phase, ...
ncoils = header.acquisitionSystemInformation.receiverChannels
print "#coils: ",ncoils

# averages
if enc.encodingLimits.average != None:
    navg = enc.encodingLimits.average.maximum + 1
else:
    navg = 1
print "#avg: ", navg
 
# sets
if enc.encodingLimits.set_ != None:
    nsets = enc.encodingLimits.set_.maximum + 1
else:
    nsets = 1
print "#sets: ", nsets
 
# phases - (e.g. temporal cardiac phases)
if enc.encodingLimits.phase != None:
    nphases = enc.encodingLimits.phase.maximum + 1
else:
    nphases = 1
print "#phases: ", nphases
      
# slices
if enc.encodingLimits.slice != None:
    nslices = enc.encodingLimits.slice.maximum + 1
else:
    nslices = 1
print "#slices: ", nslices

# repetitions
if enc.encodingLimits.repetition != None:
    nreps = enc.encodingLimits.repetition.maximum + 1
else:
    nreps = 1
print "#repetitions: ", nreps

# contrasts
if enc.encodingLimits.contrast != None:
    ncontrasts = enc.encodingLimits.contrast.maximum + 1
else:
    ncontrasts = 1
print "#contrasts: ", ncontrasts

# segments
if enc.encodingLimits.segment != None:
    nsegments = enc.encodingLimits.segment.maximum + 1
else:
    nsegments = 1
print "#segments: ", nsegments


# ---------------------------------------------------------------------
# In case there are noise scans in the actual dataset, we will skip them. 
# ---------------------------------------------------------------------
firstacq=0
for acqnum in range(dset.number_of_acquisitions()):
    acq = dset.read_acquisition(acqnum)
    
    # TODO: Currently ignoring noise scans
    if acq.isFlagSet(ismrmrd.ACQ_IS_NOISE_MEASUREMENT):
        print "Found noise scan at acq ", acqnum
        continue
    else:
        firstacq = acqnum
        print "Imaging acquisition starts acq ", acqnum
        break
#================================================================================
#
#
#================================================================================
# Sort data 
#================================================================================

#Initialiaze a storage array
#if args.removeOS:
#  print "remove os"
#  all_data = np.zeros( (nsegments,nsets, navg, nphases, nreps, ncontrasts, nslices, ncoils, rNz, rNy, eNx/2 ), dtype=np.complex64 )
#  all_data_calib = np.zeros( (nsegments,nsets, navg, nphases, nreps, ncontrasts, nslices, ncoils, rNz, rNy, eNx/2 ), dtype=np.complex64 )
#else:
#  all_data = np.zeros( (nsegments,nsets, navg, nphases, nreps, ncontrasts, nslices, ncoils, rNz, rNy, eNx ), dtype=np.complex64 ) 
#  all_data_calib = np.zeros( (nsegments,nsets, navg, nphases, nreps, ncontrasts, nslices, ncoils, rNz, rNy, eNx ), dtype=np.complex64 ) 

if args.removeOS:
  print "remove os"
  all_data = np.zeros( (nsegments,nsets, navg, nphases, nreps, ncontrasts, nslices, ncoils, eNz, eNy, eNx/2 ), dtype=np.complex64 )
  all_data_calib = np.zeros( (nsegments,nsets, navg, nphases, nreps, ncontrasts, nslices, ncoils, eNz, eNy, eNx/2 ), dtype=np.complex64 )
else:
  all_data = np.zeros( (nsegments,nsets, navg, nphases, nreps, ncontrasts, nslices, ncoils, eNz, eNy, 2*eNx ), dtype=np.complex64 ) 
  all_data_calib = np.zeros( (nsegments,nsets, navg, nphases, nreps, ncontrasts, nslices, ncoils, eNz, eNy, 2*eNx ), dtype=np.complex64 ) 


print "all_data.shape()", all_data.shape
 
# Loop through the rest of the acquisitions and stuf

for acqnum in range(firstacq,dset.number_of_acquisitions()):
    if ((acqnum % 1000)==0 & args.verbose):
       print "",acqnum," of ",dset.number_of_acquisitions()	


    acq = dset.read_acquisition(acqnum)
    
         # TODO: this is where we would apply noise pre-whitening
         # acq_data_prw = coils.apply_prewhitening(acq.data,dmtx)

         # Stuff into the buffer
    seg = acq.idx.segment
    avg = acq.idx.average
    set_ = acq.idx.set
    rep = acq.idx.repetition
    contrast = acq.idx.contrast
    slice = acq.idx.slice
    phase = acq.idx.phase
    xcenter = acq.center_sample
    y = acq.idx.kspace_encode_step_1 
    z = acq.idx.kspace_encode_step_2
        #print "y: ",y, "| z:",z;

     #TODO: for ASL number of averages (navg) did not coincide vith acq.idx.average
    if y < rNy and z < rNz and seg < nsegments and avg < navg and set_ < nsets and rep < nreps and contrast < ncontrasts and slice < nslices and phase < nphases:
       databuff = np.zeros((ncoils,eNx),dtype=np.complex64) 
       print "eNx: ",eNx,"acq.data.shape[1]:", acq.data.shape[1], "/ xcenter: ", xcenter, "databuff.shape: ", databuff.shape	      
      
      #databuff[:,eNx/2-xcenter:eNx/2-xcenter+acq.data.shape[1]] = acq.data
    if xcenter == 0:
      databuff = acq.data;
    else:
      if (xcenter-acq.data.shape[1]/2) < 0:
        databuff[:,0:acq.data.shape[1]]=acq.data;
      else:
        #databuff[:,xcenter-acq.data.shape[1]/2:xcenter+acq.data.shape[1]/2] = acq.data 
        databuff = acq.data 
      
      #databuff = acq.data 
    if args.removeOS:
        print "removeOS"
        xline = transform.transform_kspace_to_image(databuff, [1])
        x0 = (eNx - rNx) / 2
        x1 = (eNx - rNx) / 2 + rNx/2
        xline = xline[:,x0:x1]
        databuff.resize(rNx,acq.active_channels,acq.trajectory_dimensions)
        acq.center_sample = rNx/2
        # need to use the [:] notation here to fill the data
        databuff = transform.transform_image_to_kspace(xline, [1])
        #print "databuff.shape=", databuff.shape

    print "seg", seg, " / avg", avg, " / set_", set_ ," / rep", rep , " / contrast", contrast , " / slice_", slice ," / xcenter", xcenter, " / y", y, " / z", z
    
    if acq.isFlagSet(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION):
      print "calibration data: acq:", acqnum
      all_data_calib[seg,set_,avg, phase, rep, contrast, slice, :, z, y, : ] = databuff; 
    elif acq.isFlagSet(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING): 
      print "calibration data and imaging: acq:", acqnum
      all_data_calib[seg,set_,avg, phase, rep, contrast, slice, :, z, y, : ] = databuff; 
    elif acq.isFlagSet(ismrmrd.ACQ_IS_DUMMYSCAN_DATA):
      print "dummy scan data: acq:", acqnum
      #all_data[seg,set_,avg, phase, rep, contrast, slice, :, z, y, : ] = databuff; 
    elif acq.isFlagSet(ismrmrd.ACQ_IS_HPFEEDBACK_DATA): 
      print "hp feedback data data: acq:", acqnum
      #all_data[seg,set_,avg, phase, rep, contrast, slice, :, z, y, : ] = databuff; 
    elif acq.isFlagSet(ismrmrd.ACQ_IS_NAVIGATION_DATA): 
      print "navigation  data: acq:", acqnum
      #all_data[seg,set_,avg, phase, rep, contrast, slice, :, z, y, : ] = databuff; 
    elif acq.isFlagSet(ismrmrd.ACQ_IS_PHASECORR_DATA): 
      print "phase corr  data: acq:", acqnum
      #all_data[seg,set_,avg, phase, rep, contrast, slice, :, z, y, : ] = databuff; 
    elif acq.isFlagSet(ismrmrd.ACQ_IS_RTFEEDBACK_DATA): 
      print "rt feedback data: acq:", acqnum
      #all_data[seg,set_,avg, phase, rep, contrast, slice, :, z, y, : ] = databuff; 
    elif acq.isFlagSet(ismrmrd.ACQ_IS_SURFACECOILCORRECTIONSCAN_DATA): 
      print "surface coil correction data: acq:", acqnum
      all_data[seg,set_,avg, phase, rep, contrast, slice, :, z, y, : ] = databuff; 
    else:
      print "just data acq:", acqnum
      all_data[seg,set_,avg, phase, rep, contrast, slice, :, z, y, : ] = databuff; 
     
     # print "databuff.shape", databuff.shape, " / all_data.shape", all_data.shape
      # all_data[avg, set_, phase, rep, contrast, slice, :, z, y, eNx/2-xcenter:eNx/2-xcenter+acq.data.shape[1] ] = acq.data
      # all_data[seg,set_,avg, phase, rep, contrast, slice, :, z, y, : ] = databuff; 
    #else:
       #print "argghhhh"
         #   continue
    
#================================================================================
#
#
#================================================================================
# Save to matlab
#================================================================================
outname = './' + args.output

if args.writeMAT:
    for repnum in range(nreps):
      dataout=all_data[0,0,0,0,repnum,0,0,:,:,:,:] #+ all_data_calib[0,0,0,0,repnum,0,0,:,:,:,:];  
      hdf5storage.savemat(outname+str(repnum) ,{'data':dataout},appendmat=True,format='7.3');
    

    #hdf5storage.savemat(outname,{'data':all_data,'data_calib':all_data_calib},appendmat=True,format='7.3')
    #sio.savemat(outname, {'data':all_data,'data_calib':all_data_calib})
  
  #pattern = all_data != 0;
  #sio.savemat(outname + 'pat',{'pattern',pattern})  
#================================================================================
#
#
#================================================================================
# Save to cfl file --> bart
#================================================================================
#TODO manage dimensions
if args.writeCFL:
  all_data = np.sum(all_data,axis=0)
  #datasqueeze = all_data.transpose(9,8,7,6,0,1,2,3,4,5);
  datasqueeze = all_data.squeeze()
  datasqueeze = datasqueeze.transpose(3,2,1,0);
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
#
#
#================================================================================
# write data and mask to binary file
#================================================================================
if args.writeBIN:
 
  # all_data = (nsegments,nsets, navg, nphases, nreps, ncontrasts, nslices, ncoils, rNz, rNy, eNx/2)
  # sum over segments
  all_data = np.sum(all_data,axis=0)
  
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
 
    # write metadata for reconstruction
    f = open(outname+"meta.hdr","w")
    f.write('# Dimensions\n')
    f.write("n:%d \n" % n )
    f.write("m:%d \n" % m )
    f.write("nphases:%d \n" % nphases )
    f.write("ncoils:%d \n" % ncoils )
    f.write('\n')
    f.close()
   
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
#================================================================================
#
#
#================================================================================
# Coil combination
#================================================================================
if args.calcCOILS:
  if eNz > 1:
    if navg > 1:
      coil_images = np.sum(all_data,1)
      coil_images = transform.fftn(np.squeeze(coil_images[1,1,1,1,1,1,1,:,:,:,:]))
      (csm,rho) = coils.calculate_csm_inati_iter(coil_images)
  else:
    coil_images = transform.transform_kspace_to_image(np.squeeze(np.mean(all_data,0)),(1,2))
    (csm,rho) = coils.calculate_csm_inati_iter(coil_images)
  
  outname = './' + args.output + '_csm'
  sio.savemat(outname, {'csm':csm})
  outname = './' + args.output + '_rho'
  sio.savemat(outname, {'rho':rho})
#================================================================================
#
#
#================================================================================
# SOS Reconstruction
#================================================================================
if args.reconSOS:
  print "Reconstruction SOS"
  #TODO Normalize
  # all_data = all_data*datanorm
  # all_data=all_data/np.sqrt(n*m)

  # Reconstruct images
  if args.removeOS:
    images = np.zeros((nsets, navg, nphases, nreps, ncontrasts, nslices, eNz, eNy, eNx/2), dtype=np.float32)
  else:
    images = np.zeros((nsets, navg, nphases, nreps, ncontrasts, nslices, eNz, eNy, eNx), dtype=np.float32)

  for set_ in range(nsets):
    for avg in range(1):
      for phase in range(nphases):
        for rep in range(nreps):
          for contrast in range(ncontrasts):
            for slice in range(nslices):
              # FFT
              if eNz>1:
  
                print "set_:",set_," avg:", avg, " phase:", phase," rep:", rep , "contrast:", contrast, " slice:", slice
                #3D
                if args.writeBIN:
		  im = transform.fftn(np.squeeze(all_data[set_, avg, phase, rep, contrast, slice, :, :, :, :]))
                else:
		  datas=np.sum(all_data,2); # sum over averages
                  datas=np.squeeze(datas[set_,phase, rep, contrast, slice, :, :, :, :])
                  # Reconstruct in x
                  datas = transform.transform_image_to_kspace(datas,[1]);#ftshift(transform.ifftn(transform.fftshift(datas,1),1),1);
                  # Chop if needed
                  if (eNx == rNx):
                    im = datas;
                  else:
                    ind1 = np.floor((eNx - rNx)/2)+1;
                    ind2 = np.floor((eNx - rNx)/2)+rNx;
                    im = datas[ind1:ind2,:,:,:];
                   
                  # Reconstruct in y then z
                  im = transform.transform_image_to_kspace(im,[2]);#ftshift(transform.ifftn(transform.fftshift(im,2),2),2);
                  if im.shape[3]>1:
                    im = transform.transform_image_to_kspace(im,[3]);#ftshift(transform.ifftn(transform.fftshift(im,3),3),3);                                                                                                                                                                                                           end
                  # Combine SOS across coils
                  #im = sqrt(sum(abs(im).^2,4));
                  #im = transform.fftn(np.squeeze(all_data[set_, avg, phase, rep, contrast, slice, :, :, :, :]))
                  #im = transform.transform_kspace_to_image(np.squeeze(all_data[set_, avg, phase, rep, contrast, slice, :, :, :, :]), (1,2,3))
                  #outname = './' + args.output + 'soschan'
                  #sio.savemat(outname, {'im':np.squeeze(im[1,1,1,1,1,1,:,:,:,:])})
            
                # Sum of squares
                im = np.sqrt(np.sum(np.abs(im) ** 2,axis=0))
           
	      else:
                #2D
                if args.writeBIN:
		  im = transform.fftn(np.squeeze(all_data[set_, avg, phase, rep, contrast, slice, :, 0, :, :]))
 		else:
                  im = transform.transform_kspace_to_image(all_data[set_, avg, phase, rep, contrast, slice, :, 0, :, :], [1,2])
             
		
	      # Sum of squares
              im = np.sqrt(np.sum(np.abs(im) ** 2,axis=0))
           
	   
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


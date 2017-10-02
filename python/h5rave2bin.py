#!/usr/bin/env python

import os
import ismrmrd
import ismrmrd.xsd
import numpy as np
import scipy.io as sio
from ismrmrdtools import transform, coils
import argparse
import struct
import time
import hdf5storage

#================================================================================
# Helper functions
#================================================================================
def writebin(data,filename):
  try:
    os.remove(filename)
  except OSError:
    pass
  
  print "writing", filename, "to binary file"
  data = data.flatten(1);  
  isc = np.iscomplex(data[0]);
  if isc:
    datawrite = np.float64(np.concatenate((data.real,data.imag)));
    dlength = np.uint32(len(datawrite)/2)
  else:
    datawrite = np.float64(data);
    dlength = np.uint32(len(datawrite))
 
  f = open(filename,"wb")
  f.write(struct.pack('B',isc))
  f.write(struct.pack('i',dlength))
  f.write(struct.pack('d'*len(datawrite),*datawrite))
  f.close()

def sortspokes(all_data,nframes,spf):
  print "sorting spokes"
  datasqueeze = all_data.squeeze()
  datasqueeze = datasqueeze.transpose(3,2,1,0);

  # entangle slab encoding
  datasqueeze = transform.transform_kspace_to_image(datasqueeze,[2]);
  
  datasqueeze = datasqueeze.transpose(0,1,3,2);

  data_= np.zeros((2*eNx,spf,ncoils,nparts,nframes),dtype=np.complex64);
  print "nframes=",nframes
  for j in range(nframes):
    data_[:,:,:,:,j] = datasqueeze[:,j*spf:(j+1)*spf,:,:];
  data_ = data_.transpose(0,1,2,4,3);
  print " ... done"
  return data_;

def sortspokes_movavg(all_data,nframes,spf,window):
  print "sorting spokes"
  datasqueeze = all_data.squeeze()
  datasqueeze = datasqueeze.transpose(3,2,1,0);

  # entangle slab encoding
  datasqueeze = transform.transform_kspace_to_image(datasqueeze,[2]);
  
  datasqueeze = datasqueeze.transpose(0,1,3,2);

  data_= np.zeros((2*eNx,spf,ncoils,nparts,nframes),dtype=np.complex64);
  print "nframes=",nframes
  for j in range(nframes):
    data_[:,:,:,:,j] = datasqueeze[:,j*window:j*window+spf,:,:];
  data_ = data_.transpose(0,1,2,4,3);
  print " ... done"
  return data_;


def writecfl(data,filename):
  h = open(filename + ".hdr", "w")
  h.write('# Dimensions\n')
  for i in (data.shape):
    h.write("%d " % i)
  h.write('\n')
  h.close()
  d = open(filename + ".cfl", "w")
  data.T.astype(np.complex64).tofile(d) # tranpose for column-major order
  d.close()

def calc_ga_trajectory(nspokes,nframes,spf,nread):
  # define golden angle kspace trajectory and density compensation
  # golden-ratio sampling
  #     Winkelmann S, Schaeffter T, Koehler T, Eggers H, Doessel O.
  #     An optimal radial profile order based on the Golden Ratio
  #     for time-resolved MRI. IEEE TMI 26:68--76 (2007)
  print "calculating trajectory and density compensation"
  ga = 3.0 - np.sqrt(5.0);
  base = (2.0-ga)/2.0;
  angle = np.pi*np.linspace(0,nspokes-1,nspokes)*base;
  read = np.linspace(-0.5,0.5,2*eNx);

  traj = np.outer(read,np.sin(angle))+1j*np.outer(read,np.cos(angle)) 
 
  # densitiy compensation
  w = np.tile(np.abs(read),[nframes*spf,1]);
  w = np.transpose(w,(1,0));

  # assemble to frames
  traj_= np.zeros((2*eNx,spf,nframes),dtype=np.complex64);
  w_ = np.zeros((nread,spf,nframes),dtype=np.float64);  
  for j in range(nframes):
    traj_[:,:,j] = traj[:,j*spf:(j+1)*args.spf];
    w_[:,:,j] = w[:,j*spf:(j+1)*args.spf];
 
  w_ = w_/np.max(w_)*2.5*eNx/(nframes*spf*np.pi);

  print "density scale = ",np.max(w_)

  traj = traj_.transpose(1,0,2);
  traj = traj.reshape(spf*nread, nframes)
  traj = np.float64(np.dstack((traj.real,traj.imag)));
  traj = traj.transpose(0,2,1)

  w = np.float64(w_.flatten(1));
  return traj, w


def calc_ga_trajectory_slidwin(nspokes,nframes,spf,nread,window):
  # define golden angle kspace trajectory and density compensation
  # golden-ratio sampling
  #     Winkelmann S, Schaeffter T, Koehler T, Eggers H, Doessel O.
  #     An optimal radial profile order based on the Golden Ratio
  #     for time-resolved MRI. IEEE TMI 26:68--76 (2007)
  print "calculating trajectory and density compensation"
  ga = 3.0 - np.sqrt(5.0);
  base = (2.0-ga)/2.0;
  angle = np.pi*np.linspace(0,nspokes-1,nspokes)*base;
  read = np.linspace(-0.5,0.5,2*eNx);

  traj = np.outer(read,np.sin(angle))+1j*np.outer(read,np.cos(angle)) 
 
  # densitiy compensation
  w = np.tile(np.abs(read),[nframes*spf,1]);
  w = np.transpose(w,(1,0));

  # assemble to frames
  traj_= np.zeros((2*eNx,spf,nframes),dtype=np.complex64);
  w_ = np.zeros((nread,spf,nframes),dtype=np.float64);  
  for j in range(nframes):
    traj_[:,:,j] = traj[:,j*window:j*window+spf];
    w_[:,:,j] = w[:,j*window:j*window+spf]
  
  traj = traj_.transpose(1,0,2);
  traj = traj.reshape(spf*nread, nframes)
  traj = np.float64(np.dstack((traj.real,traj.imag)));
  traj = traj.transpose(0,2,1)

  w = np.float64(w_.flatten(1));
  return traj, w


################################################################################
#                                 MAIN 
################################################################################

#================================================================================
# Define parsing arguments
#================================================================================
parser = argparse.ArgumentParser()

parser.add_argument("-f", dest="filename", required=True,
                    help="h5 input file", metavar="FILE")
parser.add_argument("-n", dest="noisename", required=False,
                    help="h5 noise input file", metavar="FILE")
parser.add_argument("-s", dest="spf", type=int, required=True, help="spokes per frame")
parser.add_argument("-p", dest="maxspokes", type=int, required=True, help="spokes per frame")
parser.add_argument("-m", dest="threephases", action='store_true', default=False,
                    help="devide data into three phases (pre,inter,post contrast) with defined spokesperframe")
parser.add_argument("-w", dest="slidwin", type=int, required=False, help="sliding window", default=0)


parser.add_argument("-o", dest="output", required=False,
                    help="bin/mat/cfl output file", default="out")
parser.add_argument("-l", dest="writeCFL", action='store_true', default=False,
                    help="write cfl file")
parser.add_argument("-t", dest="writeMAT", action='store_true', default=False,
                    help="write mat file")
parser.add_argument("-b", dest="writeBIN", action='store_true', default=False,
                    help="write binary files for each slice for ICTGV Reconstruction")
parser.add_argument("-i", dest="writeINFOonly", action='store_true', default=False,
                    help="write only measurement data info to file")


args = parser.parse_args()

# Load file
if not os.path.isfile(args.filename):
    print("%s is not a valid file" % args.filename)
    raise SystemExit

print("processing ",args.filename," with ",args.spf," spokes-per-frame")
dset = ismrmrd.Dataset(args.filename, 'dataset', create_if_needed=False)
header = ismrmrd.xsd.CreateFromDocument(dset.read_xml_header())
enc = header.encoding[0]
seqpar = header.sequenceParameters;
sysinfo = header.acquisitionSystemInformation;

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
          raise Exception("Error: non noise scan found in noise calibration")

        noise[:,acqnum*noise_samples:acqnum*noise_samples+noise_samples] = acq.data
    
      noise = noise.astype('complex64')
   
   #Calculate prewhiterner taking BWs into consideration
   a = dset.read_acquisition(firstacq)
   data_dwell_time = a.sample_time_us
   noise_receiver_bw_ratio = 0.79
   dmtx = coils.calculate_prewhitening(noise,scale_factor=(data_dwell_time/noise_dwell_time)*noise_receiver_bw_ratio)


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

# partitions 
if enc.encodingLimits.kspace_encoding_step_2 != None:
    nparts = enc.encodingLimits.kspace_encoding_step_2.maximum + 1
else:
    nparts = 1
print "#partitions: ", nparts


# spokes
if enc.encodingLimits.kspace_encoding_step_1 != None:
    nspokes = enc.encodingLimits.kspace_encoding_step_1.maximum + 1
else:
    nspokes = 1
print "#spokes: ", nspokes
print "#maxspokes: ", args.maxspokes
nspokes = min(args.maxspokes,nspokes);

if args.slidwin==0:
  print "no sliding window reconstruction"
  nframes = np.int(np.floor(nspokes/args.spf));
else:
  nframes = np.int(np.floor(nspokes/(args.slidwin+1)));

print "#nframes: ", nframes

outname = './' + args.output
nread=2.0*eNx;
# ---------------------------------------------------------------------
# write metadata for reconstruction
# ---------------------------------------------------------------------
f = open(outname+"meta.hdr","w")
f.write('# Dimensions\n')
f.write("nread=%d \n" % nread )
f.write("nspokes=%d \n" % nspokes )
f.write("nframes=%d \n" % nframes )
f.write("spf=%d \n" % args.spf )
f.write("ncoils=%d \n" % ncoils )

f.write("nsets=%d \n" % nsets )
f.write("nphases=%d \n" % nphases )
f.write("nslices=%d \n" % nslices )
f.write("nreps=%d \n" % nreps )
f.write("ncontrasts=%d \n" % ncontrasts )
f.write("nsegments=%d \n" % nsegments )
f.write("nparts=%d \n" % nparts )
f.write("nwindow=%d \n" % args.slidwin )



f.write("eNx=%d \n" % eNx )
f.write("eNy=%d \n" % eNy )
f.write("eNz=%d \n" % eNz )
f.write("rNx=%d \n" % rNx )
f.write("rNy=%d \n" % rNy )
f.write("rNz=%d \n" % rNz )

f.write("eFOVx=%d \n" % eFOVx )
f.write("eFOVy=%d \n" % eFOVy )
f.write("eFOVz=%d \n" % eFOVz )
f.write("rFOVx=%d \n" % rFOVx )
f.write("rFOVy=%d \n" % rFOVy )
f.write("rFOVz=%d \n" % rFOVz )

# sequence parameters
f.write("TR=%d \n" % seqpar.TR[0])
f.write("TE=%d \n" % seqpar.TE[0])
#f.write("TI=%d \n" % seqpar.TI[0])
f.write("FA=%d \n" % seqpar.flipAngle_deg[0])
#f.write("echospacing=%d \n" % np.double(seqpar.echo_spacing[0]))

# system parameters
#f.write("systemModel=%s \n" % sysinfo.systemModel)
#f.write("systemVendor=%s \n" % sysinfo.systemVendor)
#f.write("institution=%s \n" % sysinfo.institutionName)
#f.write("station=%s \n" % sysinfo.stationName)
#f.write("relReceiveNBW=%d \n" % sysinfo.relativeReceiverNoiseBandwidth)

f.write('\n')
f.close()

if args.writeINFOonly:
    exit(0)
# ---------------------------------------------------------------------
# In case there are noise scans in the actual dataset, we will skip them. 
# ---------------------------------------------------------------------
firstacq=0
for acqnum in range(dset.number_of_acquisitions()):
    acq = dset.read_acquisition(acqnum)
    
    # TODO: Currently ignoring noise scans
    if acq.isFlagSet(ismrmrd.ACQ_IS_NOISE_MEASUREMENT):
        acq = dset.read_acquisition(acqnum)
        
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
        #cal_data[seg,set_,avg, phase, rep, contrast, slice, :, z, y, : ] = acq.data;#databuff; 


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
all_data = np.zeros( (nsegments,nsets, navg, nphases, nreps, ncontrasts, nslices, ncoils, nparts, nspokes,2*eNx ), dtype=np.complex64 ) 
 
print "all_data.shape()", all_data.shape
 
# Loop through the rest of the acquisitions and stuff
for acqnum in range(firstacq,dset.number_of_acquisitions()):
    if ((acqnum % 1000)==0):
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
    if y < args.maxspokes:
      all_data[seg,set_,avg, phase, rep, contrast, slice, :, z, y, : ] = acq.data;#databuff; 
    else:
      break;

print "... done"


#================================================================================
# Preprocess data 
#================================================================================
if args.slidwin==0:
  data_=sortspokes(all_data,nframes,args.spf);
else:
  data_=sortspokes_movavg(all_data,nframes,args.spf,args.slidwin)

#================================================================================
# get trajectory and density compensation
#================================================================================
if args.slidwin==0:
  trajwrite, wwrite = calc_ga_trajectory(nspokes,nframes,args.spf,nread)
else:
  trajwrite, wwrite = calc_ga_trajectory_slidwin(nspokes,nframes,args.spf,nread,args.slidwin)

#================================================================================
# write data and mask to binary file
#================================================================================
# write data
for slices in range(nparts):
  print "writing slice ", slices, "to file"
  datawrite = data_[:,:,:,:,slices];
  datawrite = datawrite.squeeze(); 
  datawrite = datawrite.transpose(1,0,2,3)
  datawrite = datawrite.reshape(args.spf*nread, ncoils, nframes)*(10**6)
  #print "datawrite shape", datawrite.shape
 
  if args.writeMAT:
    hdf5storage.savemat('./' + args.output + '_sl' + str(slices+1) + '_data' , {'data':datawrite},appendmat=True,format='7.3')
    hdf5storage.savemat('./' + args.output + '_sl' + str(slices+1) + '_w', {'w':wwrite},appendmat=True,format='7.3')
    hdf5storage.savemat('./' + args.output + '_sl' + str(slices+1) + '_k', {'k':trajwrite},appendmat=True,format='7.3')

  if args.writeCFL:
    writecfl(datawrite,'./' + args.output + '_sl' + str(slices+1) + '.bin');
    writecfl(wwrite,'./' + args.output + '_w.bin'); 
    writecfl(trajwrite,'./' + args.output + '_k.bin'); 
 
  if args.writeBIN:
    writebin(datawrite,'./' + args.output + '_sl' + str(slices+1) + '.bin');
    writebin(wwrite,'./' + args.output + '_w.bin'); 
    writebin(trajwrite,'./' + args.output + '_k.bin'); 
  
